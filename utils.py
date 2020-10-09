import math
import os
import random

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    VerticalFlip,
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
)
from torch.utils.data import Dataset, IterableDataset

from lrs_scheduler import WarmRestart, warm_restart
from model import ModelForward


def seed(seed=2020):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2020)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2020).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


seed()

image_size = 120, 512
img_augmentation = Compose(
    [
        Resize(height=image_size[0], width=image_size[1]),
        OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
        OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3), ], p=0.5, ),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1,
        ),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ]
)
img_augmentation_test = Compose(
    [
        Resize(height=image_size[0], width=image_size[1]),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ]
)


class DatasetPreparer:
    def __init__(self, train_path, images_path):
        self.train_path = train_path
        self.images_path = images_path

    def get_train_test(self):
        train = pd.read_csv(self.train_path)
        train['UID'] = self.images_path + '/' + train['UID'] + '.jpeg'
        list_images = [self.images_path + '/' + x for x in os.listdir(self.images_path)]
        test = [uid for uid in list_images if uid not in train['UID'].tolist()]
        return train, pd.Series(test, name='UID')

    def submission(self, prediction, index, round=False):
        if round:
            prediction = np.round(prediction)
        prediction = np.clip(prediction, 1, 7)
        index = index.apply(lambda x: x[7:-5])
        data = {'UID': index, 'growth_stage': prediction.ravel()}
        submission = pd.DataFrame(data)
        return submission


class Data(Dataset):
    def __init__(self, df, subset=None, data_augmentation=False, test=False):
        self.df = df
        self.subset = subset
        if subset == 'high_quality':
            self.df = df[df['label_quality'] == 2]
        elif subset == 'low_quality':
            self.df = df[df['label_quality'] == 1]
        self.data_augmentation = data_augmentation
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.test:
            uid, label, quality = self.df.iloc[idx]
        else:
            uid = self.df.iloc[idx]
        img = Image.open(uid)
        img = img.convert('RGB')
        img = np.asarray(img)
        img = img_augmentation(image=img)['image'] if self.data_augmentation else img_augmentation_test(image=img)[
            'image']
        img = np.transpose(img, (2, 0, 1))

        if self.test:
            return img
        else:
            return img, np.array([label], dtype=np.float32)


class Model(pl.LightningModule):
    def __init__(self, dropout=0.3, model_name='tf_efficientnet_b3_ns', T_max=10):
        super().__init__()
        self.model_forward = ModelForward(image_size=image_size, dropout=dropout, model_name=model_name)
        self.criterion = torch.nn.MSELoss()
        self.T_max = T_max

    def forward(self, x):
        return self.model_forward(x)

    def alpha_weight(self):
        if self.current_epoch < 45:
            return 0
        if self.current_epoch > 75:
            return 3
        return (3 - 0) / (75 - 45) * (self.current_epoch - 44)

    def predict(self, data_loader):
        outputs = []
        self.eval()
        with torch.no_grad():
            for x in data_loader:
                x = x.cuda()
                out = self(x)
                outputs.append(out)
        return torch.cat(outputs, dim=0).cpu().detach().numpy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('MSE_t', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('val_loss', loss, prog_bar=False, on_epoch=True, on_step=False, logger=True)
        return result

    def validation_epoch_end(self, outputs):
        rmse = torch.sqrt(outputs['val_loss'].mean())
        result = pl.EvalResult(checkpoint_on=rmse)
        result.log('RMSE', rmse, prog_bar=True, logger=True)
        return result

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=self.T_max, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]


class WarmRestartCallback(pl.Callback):
    def __init__(self):
        pass

    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if pl_module.current_epoch < (trainer.max_epochs - 4):
            pl_module.scheduler = warm_restart(pl_module.scheduler, T_mult=2)


class SemiSupervisedData(IterableDataset):
    def __init__(self, labeled_data, unlabeled_data, pretrain_epochs, data_augmentation=False):
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.pretrain_epochs = pretrain_epochs
        self.data_augmentation = data_augmentation
        self.current_epoch = 0

        self.labeled_data.drop(columns=['label_quality'], inplace=True)
        self.unlabeled_data.drop(columns=['label_quality', 'growth_stage'], inplace=True)
        self.unlabeled_data = self.unlabeled_data.sample(frac=1, random_state=1233)

        self.ratio_unlabeled_labeled = len(self.unlabeled_data) / len(self.labeled_data)
        self.checkpoint = 0

    def get_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = np.asarray(img)
        img = img_augmentation(image=img)['image'] if self.data_augmentation else img_augmentation_test(image=img)[
            'image']
        img = np.transpose(img, (2, 0, 1))
        return img

    def iter_labeled(self):
        self.labeled_data = self.labeled_data.sample(frac=1, random_state=1233)
        for i in range(self.labeled_data.shape[0]):
            uid, label = self.labeled_data.iloc[i]
            img = self.get_image(uid)
            yield img, np.array([label], dtype=np.float32)

    def iter_semi_sup(self):
        i = 0
        for i in range(math.ceil(self.unlabeled_data.shape[0] / self.ratio_unlabeled_labeled)):
            uid = self.unlabeled_data.iloc[(self.checkpoint + i) % len(self.unlabeled_data), 0]
            img = self.get_image(uid)
            yield img, np.array([-1], dtype=np.float32)
            if (self.checkpoint + i) % len(self.unlabeled_data) + 1 >= self.unlabeled_data.shape[0]:
                self.unlabeled_data = self.unlabeled_data.sample(frac=1, random_state=1233)

        self.checkpoint = (self.checkpoint + i) % len(self.unlabeled_data) + 1

        for img, label in self.iter_labeled():
            yield img, label

    def __iter__(self):
        self.current_epoch += 1
        if self.current_epoch <= self.pretrain_epochs:
            return self.iter_labeled()

        return self.iter_semi_sup()

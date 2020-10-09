import gc

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from utils import seed, DatasetPreparer, Data, Model, WarmRestartCallback

seed()

dp = DatasetPreparer('Train.csv', 'Images')
train, test = dp.get_train_test()
labeled_data = train[train['label_quality'] == 2]
unlabeled_data = train[train['label_quality'] == 1]

kfold = StratifiedKFold(n_splits=5, random_state=12382, shuffle=True)
for i, (X_train, X_val) in enumerate(kfold.split(labeled_data, labeled_data['growth_stage'])):
    seed()
    X_train, X_val = labeled_data.iloc[X_train], labeled_data.iloc[X_val]
    batch_size = 32

    train_loader = DataLoader(Data(X_train, data_augmentation=True),
                              batch_size=batch_size)
    val_loader = DataLoader(Data(X_val, data_augmentation=False), batch_size=batch_size)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    mc = pl.callbacks.ModelCheckpoint(filepath=f'fold={i}-' + '{epoch}-{RMSE:.5f}',
                                      save_top_k=3,
                                      save_weights_only=True)
    wrc = WarmRestartCallback()

    model = Model()
    trainer = pl.Trainer(gpus=1, precision=16, checkpoint_callback=mc, callbacks=[wrc, lr_monitor],
                         progress_bar_refresh_rate=5, max_epochs=70)
    trainer.fit(model, train_loader, val_loader)

    del model
    gc.collect()
    torch.cuda.empty_cache()

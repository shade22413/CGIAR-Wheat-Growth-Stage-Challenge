import sys

import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import Model, DatasetPreparer, Data, seed

test_time_augmentation = True
dp = DatasetPreparer('Train.csv', 'Images')
train, test = dp.get_train_test()
test_loader = DataLoader(Data(test, data_augmentation=False, test=True), batch_size=32)
test_loader.dataset.data_augmentation = test_time_augmentation

paths = sys.argv[1:]
folds_predictions = []
for model_path in paths:
    seed()
    model = Model().cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'])
    if test_time_augmentation:
        predictions = np.stack([model.predict(test_loader) for _ in range(5)]).mean(0)
    else:
        predictions = model.predict(test_loader)
    folds_predictions.append(predictions)
folds_predictions = np.stack(folds_predictions).mean(0)

submission = dp.submission(folds_predictions, test)
submission.to_csv('submission_folds.csv', index=False)
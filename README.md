# CGIAR-Wheat-Growth-Stage-Challenge

[Competition Link](https://zindi.africa/competitions/cgiar-wheat-growth-stage-challenge/leaderboard)

The goal of this competition is to predict the wheat growth stage using images. There are 7 growth stages (from 1 to 7). There are 2 types of labels : Expert labels (reliable) and Normal labels (less reliable). The Test set was annotated by experts. So, to train my model, I only used Expert labeled data.

## Methodology
### Model/Architecture
* I used EfficientNetB3 as a backbone.
* I added a fully connected layer on top of it (512, 256, 1) with a dropout of 0.3.
* I treated the problem as a regression problem. The chosen loss function is MSE.
* The chosen optimizer is ADAM with default parameters. 
* Image size = (512, 120)
* I used cosine annealing as a learning rate scheduler.

### Data augmentation
- RandomBrightnessContrast
- MotionBlur/MedianBlur/GaussianBlur
- Horizontal/Vertical Flip
- ShiftScaleRotate

### Training/Inference
I splitted the dataset into 5-Folds stratified with respect to 'Growth Stage'. For each split, a model was trained. For inference, we make a prediction using each one of the 5 models and then, we average them.

### Setup
- Download Images.zip and Train.csv from the link above and extract Images.zip in a folder named 'Images'.
- Run 'train_folds.py' to train the models.
- Run 'submission_folds.py BEST_FOLD1_PATH BEST_FOLD2_PATH BEST_FOLD3_PATH BEST_FOLD4_PATH BEST_FOLD5_PATH' to create the submission file.
- Submit.

### Result
Private Leaderboard Score : 0.44 (RMSE)

## Experiment Log

### Exp1 (02/23/2020)
 - Dataset
1. Rafd dataset
2. Used face_cropper.py to only crop the frontal images

 - Config
1. Run 3000 iterations
2. Didn't change hyperparameters in data_loader.

 - Result
1. Results are good.


### Exp2 (03/03/2020)
 - Dataset
1. Rafd dataset
2. Manually cropped face areas from original photos, three crops for each photo.
3. Used all photos, including frontal and all profile faces.

 - Config
1. Run 15000 iterations
2. Silenced random crop (this is not needed since input images are already 128x128), and replaced mean and std with custom values.

 - Result
1. Results are not good.
2. Color of sample images are warpped because of custom mean and std.
3. All generated faces are horizontally flipped, if given face is profile. It is weird.


### Exp3 (03/04/2020)
 - Dataset
1. Rafd dataset, the same as in *Exp2*.
2. Removed profile face shot at 0 and 180 degrees.

 - Config
1. Propose to run 120000 iterations.
2. Silenced random crop, use mean and std as it is.

 - Result
1. Sample results are really good.


### Exp4 (03/14/2020)
 - Dataset
1. AffectNet dataset.
3. Only use 8 emotion categories, remove grayscale images, remove those opencv/face_detect fails to detect.
2. 196902 for train, 2701 for test.

 - Config
1. Propose to run 200000 iterations.
2. Silenced random crop, use mean and std as it is.

 - Result
1. Some of the sample results are good.


### Exp5 (03/18/2020)
 - Idea
1. Train a regressor (together with classifier) to test the effectiveness of regression.

 - Dataset
1. AffectNet dataset. The same as in *Exp4*.
2. Have removed open mouths, reduced data from 190,000 to 170,000. Didn't apply in this experiment.
3. Have regulated different emotion categories, by removing overlapping regions on va plane. Didn't apply in this experiment.
4. Dataset and model logging folder in `AffectNet`.

 - Config
1. Almost the same hyper-parameters as in training StarGAN.
2. Set weights on regression and classification equally to be 1 and 1.

 - Result
1. Result on training set is good, but bad on validation set. One reason of bad performance on val dataset is data distribution is very different from val to train.


### Exp6 (03/19/2020)
 - Idea
1. Basically the same idea as of *Exp5*, except adding additional validation steps during training to track overfit.

 - Dataset
1. The same dataset as of *Exp5*.

 - Config
1. Set weights on regression and classification to be 5 and 1, respectively.

 - Result
1. Results on training set is good, but bad on validation set.

### Exp7 (03/20/2020)
 - Idea
1. Manually remove overlap of emotion categories on va plane. Start from faces_good3.pkl, end with faces_good4.pkl. Also resplit train/val and save to training4.csv and validation4.csv.
2. Reduced number of images from 193089 to 137578.

 - Dataset
1. Use training4.csv and validation4.csv directly.

 - Result

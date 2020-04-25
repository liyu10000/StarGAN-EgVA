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

 - Config
1. Set weights on regression and classification to be 5 and 1, respectively.

 - Result
recall [0.81649485 0.92515593 0.49468085 0.31034483 0.34693878 0.0952381 0.48630137 0.07317073]
precision [0.62460568 0.87084149 0.65957447 0.48214286 0.5862069  0.66666667 0.66981132 0.5       ]
mean recall 0.44354067768567706
mean precision 0.6324811718392941


### Exp8 (03/20/2020)
 - Idea
1. The same setup as that in *Exp7*, except the weight on classfication loss is 0, in order to check that performance of sole regression.

 - Result
recall [0.00412371 0.0031185  0.12765957 0.14942529 0.         0.02380952 0.         0.63414634]
precision [0.66666667 0.07142857 0.11538462 0.02471483 0.         0.25        nan 0.02138158]
mean recall 0.11778536769450684
mean precision nan


### Exp9 (03/20/2020)
 - Idea
1. The same setup as that in *Exp7*, except the weight on regression loss is 0, in order to check that performance of sole classification.

 - Dataset
1. There is slight change to the dataset: reset the segmenting line between fear and surprise. Now that it is a vertical line
2. training4.csv

 - Result
recall [0.68972746 0.93802521 0.65898618 0.35185185 0.24       0.20588235 0.43410853 0.06060606]
precision [0.6476378  0.87634936 0.63274336 0.54285714 0.48       0.26923077 0.47863248 0.22222222]
mean recall 0.4473984551303103
mean precision 0.5187091416462234


### Exp10 (03/21/2020)
 - Idea
1. Train StarGAN on AffectNet dataset, with classification and regression

 - Dataset
1. training4.csv
2. validation4.csv

 - Config
1. set weights on classification and regresssion to be 1, 5.

 - Result
1. Generated results are mainly controlled by cat label, cannot see any effect of va regulation.


### Exp11 (03/27/2020)
 - Idea
1. Train StarGAN on AffectNet dataset, with classification and regression

 - Dataset
1. training4.csv
2. validation4.csv

 - Config
1. set weights on classification and regresssion to be 0.5, 20.

 - Result
1. Results are better than *Exp10*. There is "gradual" shift from one emotion to another.


### Exp12 (03/27/2020)
 - Idea
1. Train StarGAN on AffectNet dataset, with classification and regression
2. Infer emotion category from va values at testing

 - Dataset
1. training3.csv
2. validation3.csv

 - Config
1. set weights on classification and regresssion to be 0.5, 20.

 - Result
1. Results are better than *Exp10*. There is "gradual" shift from one emotion to another.
2. Results are better than *Exp13* and *Exp14*, when inferring category from va.


### Exp13 (04/16/2020)
 - Idea
1. Train StarGAN on AffectNet dataset, with only regression, as a comparison to *Exp12*

 - Dataset
1. training3.csv
2. validation3.csv

 - Config
1. set weights on classification and regresssion to be 0, 40.

 - Result
1. There is difference, at testing, when using all zero emotion category than using corresponding category.
2. Probably have to completely remove the module of classification.


### Exp14 (04/20/2020)
 - Idea
1. Remove classification module completely from the network.
2. Train StarGAN on AffectNet dataset, only with regression.

 - Dataset
1. training3.csv
2. validation3.csv

 - Config
1. set weights on regression to be 40.

 - Result
1. Results are actually very good. It's more consistent than *Exp12* with varied cat input, and also more prominent than *Exp12* with zero cat input.


### Exp15 (04/24/2020)
 - Repeat *Exp12* on local machine.

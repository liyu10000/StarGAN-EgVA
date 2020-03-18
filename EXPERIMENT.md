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


### Exp5 (03/14/2020)

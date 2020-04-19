# Machine Learning: Binary classification of CT images into COVD+/COVID-
### Python pyTorch approach to classifying COVID+/COVID- CT images

### Background:
Currently, diagnosis of COVID is based on a positive viral PCR swab. However, due to the lack of test availability and the time it takes to complete a PCR test, clinicians will often treat patients as COVID+ dependent upon their clinical profile and imaging results (chest X-rays and CT scans). After finding the [COVID/Non-COVID data found here](https://github.com/UCSD-AI4H/COVID-CT), I decided it would be fun to explore the data further and fit a decent model to classify CT images as COVID+/COVID-.

#### Goal:
 - Classify new CT images into COVID+/COVID-, trained on the [data found here](https://github.com/UCSD-AI4H/COVID-CT)

#### Data exploration:
 - [X] Explore image types: some are .jpg and some .png
 - [X] Histogram plot number of images in each class
 - [X] Explore image resolution
 - [X] Remove images that are low quality or have warped perspectives
 - [X] Remove images from source with overlays e.g. classification text, boxes, arrows
 - [X] Compare mean pixel brightness of both groups, look at the stdev, anything abhorrent?

#### Preprocessing:
 - [X] Fit images to smaller resolution
 - [X] Get grayscale of image
 - [X] Export to processed data folders

#### Augmentation:
 - [X] To mitigate deficiencies in our training data, we can mirror the images to increase the size of our dataset, randomise the brightness, and randomly crop the image

#### Training:
 - [X] Get an equal sample size for both classes so we don't overfit the model to one class
 - [X] Split data into our `training` and `validation` subsets
 - [ ] Fit model

#### Validation:
 - `TBC: Results`

#### Evaluation:
 - `TBC: Results`
 - `TBC: Limitations`
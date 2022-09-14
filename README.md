# ML_Project_1_Food-Vision
Food Vision Project

## Objective: Use transfer learning to beat [DeepFood Machine Learning Paper](https://arxiv.org/abs/1606.05675). Best accuracy of 77.4%.
Note: see the Jupyter Notebook for any additional information

## Data
The Food101 dataset used is imported directly from TensorFlow datasets. Steps were taken to visualize the different images in the dataset as can be seen in the notebook.

### Preprocessing
Steps:
1. Resizing all images to size `(224,224,3)`
2. Change datatypes from `uint8` to `float32` for access to mixing point calculations
3. Normalization is done in the first layer of the backbone model. (**EfficientNetB0**)
4. All data is batched into batches of 32 and pre-fetched for faster loading

## Model
### Layers:
1. Input Layer with shape `(224,224,3)`
2. Base Model: **EfficientNetB0** (All layers frozen for feature extraction)
3. Global Average Pooling 2D Layer
4. Dense Output Layer with `softmax` activation 

### Feature Extraction Compilation:
* Loss Function: Sparse Categorical Cross Entropy (All labels are in integer format when downloading data from TensorFlow datasets)
* Optimizer: Adam with default learning rate of `0.001`
* Metrics: `accuracy`

### Feature Extraction Fitting:
Note: The model was trained using mixed point numbers for faster training
* Epochs: 3
* 15% of the validation dataset was used for evaluation during training
* Callbacks used: Tensorboard and Model Checkpoint

Feature Extraction Accuracy: **70.97%**

### Fine-Tuning Compilation:
All layers in the base model were unfrozen for fine-tuning. All hyperparameters remained the same except the learning rate.
* Optimizer: Adam with learning rate of `0.0001` (Reduced for fine-tuning)

### Fine-Tuning Fitting:
* Epochs: 100 (Used a larger number for early stopping call back)
* 15% of validation dataset used for evaluation during training
* Callbacks used: Tensorboard, Model Checkpoint, and early stopping

## Final Accuracy of **77.7%** over less than 10 epochs



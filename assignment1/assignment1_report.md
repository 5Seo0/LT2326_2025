# Assignment 1 - Chinese Character Detection Report

## Part 1 - Data Preparation

I started by exploring the provided dataset and annotation format from the dataset. Initially, I attempted to downscale images to 256x256 to reduce memory consumption. However, at this resolution, most Chinese characters became barely recognizable due to the loss of detail. At this small size, my initial linear+flatten model also caused an out-of-memory issue and could only train on a tiny subset of the dataset.

To resolve these issues, I restructured the model to include convolutional layers and max pooling, which made it possible to process the original image size and train using a larger portion of the dataset.

For data visualization, I used the sample code provided on the dataset website to visualize annotations using bounding box polygons. To train my models, I created corresponding binary masks for each image, where pixel values inside Chinese character bounding boxes were set to 1 and all other pixels to 0. These masks were used as targets in a pixel-wise binary classification task.

The dataset was divided into training, validation, and test sets from the available subset on the server. I ensured that all images included in my split had both the image file and matching annotation entries.

## Part 2 - Model Architectures

I implemented two models with different decoder strategies.

Model 1 uses F.interpolate to upsample feature maps after convolutional and pooling layers. This model is simpler in design and uses fewer parameters, relying on bilinear interpolation to scale feature maps back to input size.

Model 2 uses nn.ConvTranspose2d for learned upsampling. This model uses transpose convolutions to reconstruct the spatial resolution of the output and potentially learn more complex decoding patterns.

Both models were trained using BCEWithLogitsLoss with a manually specified pos_weight to account for class imbalance in the pixel masks. I trained both models for 25 epochs using the same training and validation splits.

Model 1 reached its best validation loss of 0.0026 at epoch 19 and maintained that loss until the end of training.

Model 2 also reached a best validation loss of 0.0026 at epoch 24, generally slower per epoch.

## Part 3 - Evaluation and Results

For evaluation, I used both visual inspection and numerical metrics.

I defined an evaluate_mse function to calculate the average mean squared error (MSE) over the test dataset.
- Model 1 average MSE: 1.749e-6
- Model 2 average MSE: 4.250e-6

This suggests that Model 1 achieved better pixel-level alignment with ground truth masks. Although the numerical difference in MSE is relatively small, it is consistent with the validation loss results and visualizations.

For qualitative analysis, I implemented a function to visualize a test image alongside its ground truth mask and the predicted probability map from the model. The visual inspection showed that Model 1 tends to produce brighter and more localized heatmaps that align well with Chinese character regions, while Model 2’s outputs were slightly blurrier in certain cases.

Both models successfully learned to predict Chinese character regions from input images, using only bounding box-derived masks. However, the model using F.interpolate for upsampling (Model 1) produced slightly better results in both quantitative metrics (MSE and validation loss) and qualitative inspection.

Although ConvTranspose2d is capable of learning more complex upsampling patterns, it appears that in this specific task and dataset size, the simplicity and smoothness of interpolation-based upsampling provided slightly more  accurate results. The performance advantage of Model 1 may also be due to reduced parameter count, leading to better generalization in this moderately sized training setup.

Given the marginal differences, further experiments with deeper architectures, regularization, or larger data subsets could help clarify which upsampling method scales better. For this assignment, however, Model 1 with interpolation provided more efficient results.
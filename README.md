# Cat-vs.-Dog-Image-Classification

### Description
- Solving the famous Cat versus Dog problem: given an image, use a CNN model to classify whether it is a dog image or a cat image.
- Utilize the '**datasets**' and '**transforms**' modules from the torchvision library for image preprocessing.

### Metrics Evaluation
- Model performance will be assessed based on **precision**, **accuracy**, and **recall** metrics.
- Metrics should be computed on the validation set.
- To meet project criteria, all three metrics must surpass 0.6.

### Model Hyper Parameters
- Learning Rate: 0.001
  - A low learning rate was chosen to help prevent overshooting the optimal solution.
- Batch Size: 8
- Epoch Size: 8
  - Smaller epoch size was chosen to reduce overfitting.

### Model Structure
- **Layer 1: Convolution -> ReLU -> Pool**
  - Convolution: Input channels=3, Output channels=64, Kernel size=3
  - Batch Normalization
  - ReLU activation function
  - Max Pooling: Kernel size=2
- **Layer 2: Convolution -> ReLU -> Pool**
  - Convolution: Input channels=64, Output channels=128, Kernel size=3
  - Batch Normalization
  - ReLU activation function
  - Max Pooling: Kernel size=2
- **Layer 3: Convolution -> ReLU -> Pool**
  - Convolution: Input channels=128, Output channels=256, Kernel size=3
  - Batch Normalization
  - ReLU activation function
  - Max Pooling: Kernel size=2
- **Layer 4: Convolution -> ReLU -> Pool**
  - Convolution: Input channels=256, Output channels=512, Kernel size=3
  - Batch Normalization
  - ReLU activation function
  - Max Pooling: Kernel size=2
- **Layer 5: Linear -> ReLU -> Linear -> ReLU -> Linear**
  - Fully connected layer: Input size=512, Output size=512
  - ReLU activation function
  - Dropout with probability 0.5
  - Fully connected layer: Input size=512, Output size=2 (for binary classification)
 
### Results
![Alt text](CatVDogCNN/Results.jpeg?raw=true "Metrics Scores")


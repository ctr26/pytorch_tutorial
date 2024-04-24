- This is a tutorial for learning pytorch from scratch.


We will start with a simple classifier (easiier) and then move on to a more complex model that can be used for segmentation (harder) .

Start with exercise 1_ml_buggy, and try to fix the bugs

Breakpoints are included in the code to help guide the fixes.

Learning points:
  - Building a simple MLP with pytorch
  - Identifying and fixing bugs in the code
  - Logits vs probabilities
  - The logic of the training loop
  - Metrics on classification tasks

With exercise 2 you will learn how to use pytorch to train a resnet18 model on CIFAR10 dataset.
Learning points are:
  - Building a resnet18 with pytorch
  - Using a pretrained model
  - Training the model
  - More compelex data augmentation


With exercise 3 you will learn how to use pytorch to train a Unet on the SBDataset
Learning points are: 
  - Building a Unet with Monai
  - Creating transforms to preprocess the data to work with the Unet
  - Training the Unet
  - Using a segmentation appropriate loss function
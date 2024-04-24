
PyTorch Tutorial: From Simple Classifiers to Complex Models
===========================================================

Overview
--------

This tutorial will guide you through the basics of PyTorch, starting with a simple multilayer perceptron (MLP) model and progressing to more complex models used for image segmentation. Each exercise is designed to build on the skills learned in the previous one, enhancing your understanding of deep learning fundamentals and PyTorch's functionality.

Getting Started
---------------

To begin, ensure you have PyTorch installed along with necessary libraries like torchvision, tqdm, sklearn, numpy, and monai for advanced exercises. Each exercise includes breakpoints to assist you in identifying and fixing specific issues, enhancing both your debugging skills and understanding of model architectures.

To get started, ensure you have the following libraries installed:

```bash
pip install torch torchvision tqdm scikit-learn numpy monai
```

Or with conda environment:

```bash
conda env create -f environment.yml -p ./env
conda activate ./env
```

GPU support is recommended for the advanced exercises, as they involve training deep neural networks on large datasets.
If you don't have a GPU, you can use Google Colab or Kaggle notebooks to run the exercises.

### Exercise 1: Fixing a Buggy MLP Model (Beginner)

*   **Objective**: Learn to build and debug a basic MLP using PyTorch.
    
*   **Key Learning Points**:
    
    *   Constructing a simple MLP.
        
    *   Debugging common issues in PyTorch models.
        
    *   Understanding the difference between logits and probabilities.
        
    *   Implementing an effective training loop.
        
    *   Evaluating classification metrics.
        

### Exercise 2: Training a ResNet18 on CIFAR10 Dataset (Intermediate)

*   **Objective**: Enhance your skills by training a more complex model on a standard dataset.
    
*   **Key Learning Points**:
    
    *   Building and understanding ResNet18 architecture.
        
    *   Utilising pre-trained models to improve performance.
        
    *   Implementing training routines for deep networks.
        
    *   Applying complex data augmentation techniques.
        

### Exercise 3: Image Segmentation with U-Net on the SBDataset (Advanced)

*   **Objective**: Apply your knowledge to train a U-Net for image segmentation.
    
*   **Key Learning Points**:
    
    *   Constructing a U-Net model using Monai.
        
    *   Preprocessing data with transforms suitable for segmentation tasks.
        
    *   Training a segmentation model effectively.
        
    *   Utilising segmentation-specific loss functions.
        


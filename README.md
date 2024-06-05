# Bilinear-Pooling-with-Hierarchical-Structure-for-Precise-Visual-Recognition
Computer vision is continuously advancing in the field of accurate visual representation. Deep learning architectures such as ResNet50 and pooling algorithms have been crucial to this process. In this study, a pioneering approach called Bilinear Pooling with Hierarchical Structure is introduced to enhance the visual recognition accuracy. Using ResNet50 the method incorporates hierarchical pooling using the ReLU activation function. This new combination improves both feature representation and the accuracy with which complex visual details are captured. Through the hierarchical organization of pooling the layers,  multi-scale contextual information is  efficiently extracted.   Through training, it is shown that the proposed framework is effective in producing notable performance improvements. The results highlight the potential benefits of this method.
-------

### Screenshots

<p align="center">
  <img src="Demo/Screenshot 2024-06-06 020855.png" alt="image"/>
</p>

<p align="center">
  <img src="Demo/Screenshot 2024-06-06 020941.png" alt="image"/>
</p>

<hr>

### Introduction
The project aims to enhance fine-grained visual recognition accuracy.

Integrate Hierarchical Bilinear Pooling (HBP) with ResNet architecture.

Leverage ResNet's capabilities and pre-trained parameters.

Provide accessibility to the codebase for broader research and adaptation.

<hr>

### Objectives
• Improving the efficiency of the model using Two-Step Training Strategy

• Increasing accuracy through Spatial Relationship Learning

• Improved Fine-Grained Recognition using hierarchical bilinear pooling

• Hierarchical pooling to capture complex relationships

• Enhancing the feature representation of the model

• Efficiently extracting multi-scale contextual information

• Improving the performance of the model through training

<hr>

### ARCHITECTURE diagram
<p align="center">
  <img src="Demo/Screenshot 2024-06-06 023627.png" alt="image"/>
</p>

<hr>

## Requirements
- python 2.7
- pytorch 0.4.1

## Train

Step 1. 
- Download the resnet pre-training parameters.

- Download the CUB-200-2011 dataset.
[CUB-download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

Step 2. 
- Set the path to the dataset and resnet parameters in the code.

Step 3. Train the fc layer only.
- python train_firststep.py


    	


Step 4. Fine-tune all layers. It gets an accuracy of around 86% on CUB-200-2011 when using resnet-50.
- python train_finetune.py

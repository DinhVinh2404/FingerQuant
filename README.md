<div align="center">
<h2>FingerQuant</h2>
<h4>Research on a Quantization Method to Optimize a Deep Learning Model for Finger Recognition</h4>
</div>

## Abstract
According to current statistics, smartphones have become widespread, and almost all of them integrate a touchscreen as the primary interface for direct interaction. Although the concept of direct touch seems natural to users, the vocabulary of touch input is limited compared to traditional hardware devices such as keyboards and mice. If different finger inputs on the display can be distinguished, then the same input performed with different fingers could trigger different functions—similar to using multiple buttons on a computer mouse or keyboard. The goal of this project is to optimize the size of a simple finger classification model so that it can be easily deployed on devices without consuming too many resources.


## Usage
### 1. Use the available FP model or train a new model with your own dataset.
In the `train` directory, the model takes as input capacitive touchscreen data, which has been preprocessed into grayscale matrices of size 8×8.

### 2. Installation.
```
python >= 3.7.13
numpy >= 1.21.6
torch >= 1.11.0
torchvision >= 0.12.0
```

### 3. Run experiments
You can run ```run_script.py``` with different settings by modifying the quantization bit-width in the file.

The weights and activations of convolution and linear layers are quantized sequentially, layer by layer, until the last layer of the model is processed.  
The optimization objective consists of two main components:  

1. **Normalization Loss** – the instantaneous output difference of the current layer being quantized, measured between the quantized model and the FP model using Mean Squared Error (MSE).  
2. **Prediction Difference Loss** – the difference in the final predictions between the quantized model and the FP model, measured using Kullback–Leibler (KL) Divergence.

## Experimental Results

### Task: Distinguishing Between Two Thumbs

| Model             | Original Accuracy | Quantized Accuracy | Accuracy Drop |
|-------------------|------------------:|-------------------:|--------------:|
| CapFingerId       | 92.13%            | 90.29%             | 1.84%         |
| ThumbClassifier1  | 91.50%            | 91.38%             | 0.12%         |
| ThumbClassifier2  | 93.36%            | 93.13%             | 0.23%         |

### Task: Distinguishing Between Ten Fingers

| Model             | Original Accuracy | Quantized Accuracy | Accuracy Drop |
|-------------------|------------------:|-------------------:|--------------:|
| CapFingerId       | 62.03%            | 61.35%             | 0.68%         |
| ThumbClassifier1  | 48.96%            | 48.92%             | 0.04%         |
| ThumbClassifier2  | 59.02%            | 57.17%             | 1.85%         |

## Thanks
The research and implementation in this project are based on [**PD-Quant: Post-Training Quantization Based on Prediction Difference Metric**](https://arxiv.org/pdf/2212.07048).


# RetinaNets & EVNets

### Abstract

> Convolutional neural networks (CNNs) excel at object recognition but struggle with robustness to image corruptions, limiting their real-world applicability. Recent work has shown that incorporating a neural front-end block simulating primate V1, can improve CNN robustness. This study expands on that approach by introducing two novel CNN families: RetinaNets and EVNets, which incorporate a RetinaBlock designed to simulate retinal and LGN visual processing. We evaluate these models alongside VOneNets adapted for Tiny ImageNet, using ResNet18 and VGG16 as base architectures. Our results demonstrate that the RetinaBlock simulates retinal response properties and RetinaNets improve robustness against corruptions when compared to the base model, with an overall relative accuracy gain of 12.6% on Tiny ImageNet-C. EVNets, which combine the RetinaBlock and VOneBlock, show even greater improvements with a 19.4% relative gain. These enhancements in robustness are consistent across different back-end architectures, though accompanied by slight decreases in clean image accuracy. Our findings suggest that simulating multiple stages of early visual processing in CNN front-ends can provide cumulative benefits for model robustness.

### How to Run

On Unix-based machines:

```
pip install --update pip
git clone https://github.com/lucaspiper99/retinanets-evnets.git
cd retinanets-evnets
pip -m venv myvenv
source /myvenv/bin/activate
pip install -r requirements.txt
```

### BibTex

...
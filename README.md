# RetinaNets & EVNets

### Abstract

> Convolutional neural networks (CNNs) excel at object recognition but struggle with robustness to image corruptions, limiting their real-world applicability. Recent work has shown that incorporating a neural front-end block simulating primate V1, can improve CNN robustness. This study expands on that approach by introducing two novel CNN families: RetinaNets and EVNets, which incorporate a RetinaBlock designed to simulate retinal and LGN visual processing. We evaluate these models alongside VOneNets adapted for Tiny ImageNet, using ResNet18 and VGG16 as base architectures. Our results demonstrate that the RetinaBlock simulates retinal response properties and RetinaNets improve robustness against corruptions when compared to the base model, with an overall relative accuracy gain of 12.6% on Tiny ImageNet-C. EVNets, which combine the RetinaBlock and VOneBlock, show even greater improvements with a 19.4% relative gain. These enhancements in robustness are consistent across different back-end architectures, though accompanied by slight decreases in clean image accuracy. Our findings suggest that simulating multiple stages of early visual processing in CNN front-ends can provide cumulative benefits for model robustness.

## How to run

### Setup virtual envoirenment (Unix-based machines)

```
pip install --update pip
git clone https://github.com/lucaspiper99/retinanets-evnets.git
cd retinanets-evnets
pip -m venv myvenv
source /myvenv/bin/activate
pip install -r requirements.txt
```

### Train on Tiny ImageNet

1. Download Tiny ImageNet publicly available at <http://cs231n.stanford.edu/tiny-imagenet-200.zip> to sibling directory `../tiny-imagenet-200`.
2. Run:

```
train.py [-h] [--hyperparameters HYPERPARAMETERS] [--data_path DATA_PATH] [--out_dir OUT_DIR] [--use_checkpoint] [--preload] [--num_workers NUM_WORKERS] [--prefetch_factor PREFETCH_FACTOR] [--seed SEED] [--model_arch MODEL_ARCH] [--model_family {base,retinanet,vonenet,evnet}] [--device DEVICE]

  -h, --help            show this help message and exit
  --hyperparameters HYPERPARAMETERS
                        Path to yaml file with hyperparameters. Defaults to "./hyperparameters.yml". 
  --data_path DATA_PATH
                        Path to the data. Defaults to "../tiny-imagenet-200".
  --out_dir OUT_DIR     Path to the output directory where the model will be saved. Defaults of "./output".
  --use_checkpoint      Whether to load the last checkpoint.
  --preload             Whether to preload the data in memory.
  --num_workers NUM_WORKERS
                        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 8.
  --prefetch_factor PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker. Defaults to 2.
  --seed SEED           Seed for the random number generators. Defaults to 42.
  --model_arch {resnet18,vgg16}
                        EVNet backend model architecture. Defaults to "resnet18".
  --model_family {base,retinanet,vonenet,evnet}
                        Model family. Defaults to "base".
  --device DEVICE       Device to use when training the model. Defaults to "cuda:0".
  ```

### Test on Tint ImageNet-C

1. Download Tiny ImageNet-C publicly available at <https://github.com/hendrycks/robustness> under Creative Commons Attribution 4.0 International to sibling directory `../tiny-imagenet-c`
2. Run:

```
test_corruptions.py [-h] [--data_path DATA_PATH] [--use_checkpoint] [--preload] [--num_workers NUM_WORKERS] [--prefetch_factor PREFETCH_FACTOR] [--model_arch MODEL_ARCH] [--model_family {base,retinanet,vonenet,evnet}] [--device DEVICE]

  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the data. Defaults to "../tiny-imagenet-c".
  --preload             Whether to preload the data in memory.
  --num_workers NUM_WORKERS
                        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 8.
  --prefetch_factor PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker. Defaults to 2.
  --model_arch {resnet18,vgg16}
                        EVNet backend model architecture. Defaults to "resnet18".
  --model_family {base,retinanet,vonenet,evnet}
                        Model family. Defaults to "base".
  --device DEVICE       Device to use when training the model. Defaults to "cuda:0".
  ```

## BibTex
```
@misc{piper2024explicitlymodelingprecorticalvision,
      title={Explicitly Modeling Pre-Cortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness}, 
      author={Lucas Piper and Arlindo L. Oliveira and Tiago Marques},
      year={2024},
      eprint={2409.16838},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.16838}, 
}
```

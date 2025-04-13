
# Towards Disentangled and Controllable Deep Metric Learning with Human-Like Concept Decomposition
This is the implementation of “Towards Disentangled and Controllable Deep Metric Learning with Human-Like Concept Decomposition” in Pytorch.

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)


## Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))

2. Extract the tgz or zip file into `./data/`

## Training 

We trained the model on three popular benchmarks: CUB-200-2011, Stanford Cars 196 and SOP:
```
sh run/CUB.sh
```
```
sh run/Cars.sh
```
```
sh run/SOP.sh
```


## Ackowledgement

We thank the following repos providing helpful components in our work:

- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)

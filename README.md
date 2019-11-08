# [Re] Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting

Reimplementation of NeurIPS'19: "Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting" by Shu et al.  

## Setups

* Linux
* Python 3.7.4
* PyTorch 1.2.0
* Torchvision 0.4.0


### Running the Meta-Weight-Net on benchmark datasets (CIFAR-10 or CIFAR-100)

```
nohup python main.py --cifar_type 10 --model_type MWN --experiment_type 'Imbalance' --factor 200 --seed 12345 > Logs/log_file_10_MWN_Imbalance_200_12345.txt &
```


## Acknowledgments

We would like to thank https://github.com/akamaster/pytorch_resnet_cifar10 for the ResNet-32 implementation and https://github.com/szagoruyko/wide-residual-networks for the Wide-ResNet implementation.

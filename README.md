# U-2-NET
An implementation of U-2-NET ([U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)) using PyTorch.

Based on the [PyTorch version](https://github.com/NathanUA/U-2-Net) by NathanUA, PDillis, vincentzhang, and chenyangh.

# U-NET
An implementation of U-NET ([U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)) using PyTorch.

# Application
This implementation is to reproduce the results of human portrait drawing. So we use [APDrawingGAN dataset](https://github.com/yiranran/APDrawingGAN) to train and test.

# Requirement
We recommand the following version of the library:

Python 3.8  
PyTorch 1.7  
numpy 1.19  
pandas 1.1  
scikit-image 0.17.2


# Usage for portrait generation
The pretrained models and the split dataset can be downloaded at [this link](https://pan.baidu.com/s/18EKG2vzxJ9a2AujTj6C28w) (extraction code: ***xdu5***).

For the dataset, you can use split version provided by us or download the original dataset and use split.py to split. The dataset should under ```./dataset/``` path and split into ```'train_data', 'train_mask', 'test_data', 'test_mask'```. Or you can modify the code yourself.

If you simply want to see the results, download ```results_150.zip``` to see the results after trained for 150 epochs.

For using the pretrained models, please download the models and put them under ```pretrained_models/``` then open ```net_test.py``` and change the name to 'u2net' or 'unet' to load different network respectively. These two models are both trained only 150 epochs. And if you want to test your own images, please put them into ```'./dataset/test_data' and './dataset/test_mask'``` and use '.png' images.

And if you want to train the model yourself, you can run the ```u2net_train_test.py``` to train 'u2net' or ```unet_train_test.py``` to train 'unet', if you don't have enough GPUs, you can mannualy change the batchsize in the aforementioned files.

# Results demo
Up: original image, original mask.  
Down: unet's results, u2net's results.  
<p>
    <img src='demo/img_1695.png' width="49%"/>
    <img src='demo/img_1695_mask.png' width="49%"/>
</p>
<p>
    <img src='demo/img_1695_unet.png' width="49%"/>
    <img src='demo/img_1695_u2net.png' width="49%"/>
</p>

# Citation

Dataset:
```
@article{Yi2019APDrawingGANGA,
  title={APDrawingGAN: Generating Artistic Portrait Drawings From Face Photos With Hierarchical GANs},
  author={Ran Yi and Yongjin Liu and Y. Lai and Paul L. Rosin},
  journal={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
  pages={10735-10744}
}
```

U-2-NET:
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```

U-NET:
```
@article{Ronneberger2015UNetCN,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={O. Ronneberger and P. Fischer and T. Brox},
  journal={ArXiv},
  year={2015},
  volume={abs/1505.04597}
}
```

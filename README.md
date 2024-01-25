# ESP-PCT: Enhanced VR Semantic Performance  through Efficient Compression of Temporal and Spatial Redundancies in Point Cloud Transformers

This is Pytorch implementation of our paper "ESP-PCT: Enhanced VR Semantic Performance  through Efficient Compression of Temporal and Spatial Redundancies in Point Cloud Transformers".
## Datasets

| Data Type | Google Links | Baidu Links                                                 |
|------------------|-------------------|-------------------------------------------------------------------------|
| mmWave raw IQ, mmWave Point Cloud, Kinect | [Google Drive](https://drive.google.com/drive/folders/1Hk5WnxPbsS_3Ilbs9x-ut50CIZU6m9Ns?usp=sharing) | [Baidu Drive ](https://pan.baidu.com/s/1k01fexEuOZIGssj_GvlOwQ?pwd=mpjx) |

## Pre-trained Models

| Backbone | Accuracy	                | Checkpoints Google Links | Checkpoints Baidu Links                                                 |
|---|--------------------------|-------------------|-------------------------------------------------------------------------|
| Point Transformer   | 98.8 (k=96,$\eta$=0.82) | [Google Drive](https://drive.google.com/file/d/19yV-4ChD2RLsMBLYPeEPjnDua8wJQtbZ/view?usp=sharing) (v599) | [Baidu Drive ](https://pan.baidu.com/s/1k01fexEuOZIGssj_GvlOwQ?pwd=mpjx) (v599) |
- What are contained in the checkpoints:

```
**.pth
├── epoch: indicate many iterations of the training loop have been completed.
├── model: state dictionaries of the model
├── optimizer: a dictionary that contains information about the optimizer’s hyperparameters
├── loss: a scalar that represents the average loss of the model on the training data.
```

## Requirements
- python 3.11.4
- pytorch 2.0.1
- torchvision 0.15.2a0

## Data Preparation
- The PointCloud dataset should be prepared as follows:
```
matData
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```

## Visualization
- Visualization of Point Cloud on VR user body, i.e., across wall.

```
matlab -r "data_visualation"
```

## Evaluate Pre-trained Models
- Get accuracy of each stage, see the terminal output
```
python train_me.py
```
- Visualize Training Process
```
Open a terminal (or a command prompt in Windows) and change to the directory where your log files are located, or specify a directory that contains the log files.

Enter the following command to start the TensorBoard server:
tensorboard --logdir=<directory_name>

Enter the following URL in your browser to open the TensorBoard main page:
http://localhost:6006
```

## Train
- Train ESP-PCT
```
python train_me.py
```


## Acknowledgment
Our code of Point Transformer is from [here](https://github.com/POSTECH-CVLab/point-transformer). Our code of Point 4D Transformer is from [here](https://github.com/hehefan/P4Transformer). Our code of Self-Supervised4D is from [here](https://github.com/dongyh20/C2P). Thanks to these authors. 


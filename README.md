# Pytorch code for EGCT

**EGCT：Enhanced Graph Convolutional Transformer for 3D Point Cloud Representation Learning**

### Requirement

This code is tested on Python 3.6 and Pytorch 1.1.0

### 1、Part Segmentation 

#### Dataset

Download the ShapeNetPart dataset (xyz, normals and labels) from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and put it under `data`

#### Train:

`python train.py`

#### Test:

`python test.py --log train --checkpoint epoch.pkl`

#### Visualize:

`python test.py --log train --checkpoint epoch.pkl --output ./results`


### 2、Classification

#### Train:

`python train.py`

#### Test:

`python train.py --eval 1`

#### Dataset

Download the ModelNet40 dataset (xyz and labels) from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and put it under `data`

### 3、Indoor scene segmentation 

#### Dataset

Download the S3DIS dataset from [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2) and put it under `data`

Compile the C++ extension modules for python located in `cpp_wrappers`. Open a terminal in this folder, and run:

`sh compile_wrappers.sh`

#### Train:

`python train.py`

#### Test:

`python test.py --log ./results/train --model epoch.tar`

#### Visualize:

`python visualize.py --model epoch.tar`

## Reference

Wang, Y.; Sun, Y.; Liu, Z.; Sarma, S.E.; Bronstein, M.M.; Solomon, J.M. Dynamic Graph CNN for Learning on Point Clouds 2019. 38. https://doi.org/10.1145/3326362.

Zhou, H.; Feng, Y.; Fang, M.; Wei, M.; Qin, J.; Lu, T. Adaptive Graph Convolution for Point Cloud Analysis. In Proceedings of the Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4965–4974.

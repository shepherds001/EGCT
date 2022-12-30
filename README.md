# Pytorch code for EGCT

**EGCT：Enhanced Graph Convolutional Transformer for 3D Point Cloud Representation Learning**

### Requirement

This code is tested on Python 3.6 and Pytorch 1.1.0

### Part Segmentation 

#### Dataset

Download the ShapeNetPart dataset (xyz, normals and labels) from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and put it under `data`

#### Train:

`python train.py`

#### Test:

`python test.py --log train --checkpoint epoch_099.pkl`

#### Visualize:

`python test.py --log train --checkpoint epoch_099.pkl --output ./results`


### Classification

#### Train:

`python train.py`

#### Test:

`python train.py --eval 1`

#### Dataset

Download the ModelNet40 dataset (xyz and labels) from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and put it under `data`

## Reference

Wang, Y.; Sun, Y.; Liu, Z.; Sarma, S.E.; Bronstein, M.M.; Solomon, J.M. Dynamic Graph CNN for Learning on Point Clouds 2019. 38. https://doi.org/10.1145/3326362.

Zhou, H.; Feng, Y.; Fang, M.; Wei, M.; Qin, J.; Lu, T. Adaptive Graph Convolution for Point Cloud Analysis. In Proceedings of the Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4965–4974.

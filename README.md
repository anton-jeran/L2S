# Listen2Scene: Interactive material-aware binaural sound propagation for reconstructed 3D scenes

This is the official implementation of our end-to-end binaural audio rendering approach ([**Listen2Scene**](https://arxiv.org/pdf/2302.02809.pdf)) for virtual reality (VR) and augmented reality (AR) applications. Our Neural Sound Rendering results is available [**here**](https://anton-jeran.github.io/Listen2Scene/).


## Requirements

```
Python3.6
pip3 install numpy
pip3 install torch
pip3 install torchvision
pip3 install python-dateutil
pip3 install soundfile
pip3 install pandas
pip3 install scipy
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip3 install librosa
pip3 install easydict
pip3 install cupy-cuda102
pip3 install wavefile
pip3 install torchfile
pip3 install pyyaml==5.4.1
pip3 install pymeshlab
pip install openmesh
pip3 install gdown
pip3 install matplotlib
pip3 install IPython
pip3 install pydub
```
Please note that, in the above requirements we installed and tested on cupy library and torch-geometric library compatible with CUDAv10.2. For different CUDA versions, you can find the appropriate installation commands here.

```
1) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
2) https://docs.cupy.dev/en/stable/install.html

```

## Download Listen2Scene Dataset

To download the Listen2Scene dataset run the following command  
```
source download_data.sh
```
 You also can directly download it from the following link
 
```
https://drive.google.com/uc?id=1FnBadVRQvtV9jMrCz_F-U_YwjvxkK8s0
```



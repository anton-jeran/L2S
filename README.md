# Listen2Scene: Interactive material-aware binaural sound propagation for reconstructed 3D scenes

This is the official implementation of our end-to-end binaural audio rendering approach ([**Listen2Scene**](https://arxiv.org/pdf/2302.02809.pdf)) for virtual reality (VR) and augmented reality (AR) applications. Our Neural Sound Rendering results is available [**here**](https://anton-jeran.github.io/Listen2Scene/).


## Requirements

```
Python3.9.7
pip3 install numpy
pip3 install wheel
pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install python-dateutil
pip3 install soundfile
pip3 install pandas
pip3 install scipy
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip3 install librosa
pip3 install easydict
pip3 install cupy-cuda11x
pip3 install wavefile
pip3 install torchfile
pip3 install pyyaml==5.4.1
pip3 install pymeshlab
pip install openmesh
pip3 install gdown
pip3 install matplotlib
pip3 install IPython
pip3 install pydub
pip3 install torch-geometric==2.1.0
```
Please note that, in the above requirements we installed and tested on cupy library and torch-geometric library compatible with CUDAv11.7. For different CUDA versions, you can find the appropriate installation commands here.

```
1) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
2) https://docs.cupy.dev/en/stable/install.html

```

**Note** - If you have issues with loading the trained model, downgrade torch-geometric (**pip3 install torch-geometric==2.1.0**)

## Download Listen2Scene Dataset

To download the Listen2Scene dataset run the following command  
```
source download_data.sh
```
 You also can directly download it from the following link
 
```
https://drive.google.com/uc?id=1FnBadVRQvtV9jMrCz_F-U_YwjvxkK8s0
```


## Evaluation

Download the trained model, sample 3D indoor real environment meshes from [**ScanNet dataset**](https://github.com/ScanNet/ScanNet), and sample source-receiver paths files using the following command.

```
source download_files.sh
```

Generate embedding with different receiver and source locations for five different real 3D indoor scenes. For 5 different real indoor scenes, we have stored sample source-receiver locations in a CSV format inside the **Paths** folder. Columns 2-4 give the 3D cartesian coordinates of the source and receiver positions. Column 1 with negative values corresponds to source positions and Column 1 with non-negative values corresponds to listener positions. 

```
python3 embed_generator.py
```

Generate binaural IRs corresponding to each embedding file inside **Embeddings** folder using the following command.

```
python3 evaluate.py
```

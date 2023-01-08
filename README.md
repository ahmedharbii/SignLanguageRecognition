# SignLanguage_YOLOv7
 Deploying a YOLOv7 detecting American Sign Language (ASL) on Intel RealSense


# Scripts:
Training YOLOv7:

Open the notebook:
```
Sign_Language_Training_on_Custom_Data.ipynb
```

```
!python train.py --batch 8 --epochs 200 --data /yolov7/American-Sign-Language-Letters/data.yaml \
--weights '/yolov7/runs/train/exp11/weights/best.pt' \
--device 0 --img 640 640 --cfg '/cfg/training/yolov7.yaml'
```

Training Modified ResNet18: 
```
image_classification.ipynb
```

Testing yolov7 - Camera or Image upload:
```
test_yolo.py
```
Testing Modified Resnet - Camera or Image upload:
```
test_resnet.py
```


# Required Libraries:
(Note some of them might not be needed depending on the application)

inside yolov7 jupyter notebook, run the command to install dependencies from Requirements.txt.



 
 # pyrealsense2
 ```
 pip install pyrealsense2
```
# Install requests
 ```
 pip3 install requests
 ```

# Install Pillows for PIL image
```
pip install Pillow==6.1
```
# Pandas
```
pip install pandas
```
# Pytorch
```
pip3 install torch torchvision torchaudio
```
# tqdm
```
pip install tqdm
```

# pyyaml
```
pip install pyyaml
```

# Misc
```
python -m pip install -U matplotlib
pip install seaborn
pip install scipy
```

# For exporting Model:
```

pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
pip install pycuda

pip install protobuf<4.21.3
pip install onnxruntime-gpu
pip install onnx>=1.9.0
pip install onnx-simplifier>=0.3.6

pip install nvidia-pyindex
pip install onnx-graphsurgeon

```

# To use Jupyter Notebooks
```
pip install jupyter
pip install ipywidgets widgetsnbextension pandas-profiling
```
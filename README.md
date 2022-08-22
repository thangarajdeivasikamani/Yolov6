# Yolov6  Custom Object detection : Trafic data

This repo contain the jupyter notebook to train the custom model using YOLOV6 and also detect the objects based on YOLOV6 Flask application

# Data preparation:
  Download the data set from Kaggle https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset

- Create the folder structure as below
- images and labels folder
- under the images create train,test,val
- under the labels create train,test,val
- copy images and labels respective folder
- inside the Test folder copy few images from train and val
- update the dataset.yaml file as per above folder path.
- Copy the dataset.yaml file to Yolo6 folder or while training give the actual path

Steps:
1. Mount the drive

```
 from google.colab import drive
 drive.mount('/content/drive')

```
 Create the folder like 
  Yolov6_custom

2. Upload the dataset into colab Yolov6_custom folder

3. git clone the YOLO V6

```
!git clone https://github.com/meituan/YOLOv6
%cd YOLOv6
!pip install -r requirements.txt

```
4. Navigate to YOLOv6
5. Install the requirements

!pip install -r requirements.txt

6.Prepare the data.yaml

- Example End folder should not /
- There should be space b/w train: /
- If it's full path we need to specify / not ./

![image](https://user-images.githubusercontent.com/46878296/185957424-361f7b1f-5299-4a87-b024-2ef2c92eb88a.png)

7. Navigate to

```
 %cd /content/drive/MyDrive/YOLO_6_Custom/YOLOv6

```




# Training

```
 !python tools/train.py --batch 32 --conf configs/yolov6s.py --epochs 200 --data  /content/drive/MyDrive/YOLO_6_Custom/dataset/trafic_data/data_1.yaml --device 0

```
# Resume

```
!python tools/train.py --resume

```

# Evaluate YOLOv6 Model Performance

```

!python tools/eval.py --data /content/drive/MyDrive/YOLO_6_Custom/dataset/trafic_data/data_1.yaml --weights /content/drive/MyDrive/YOLO_6_Custom/YOLOv6/runs/train/exp/weights/best_ckpt.pt --device 0

```
# Inference:

![image](https://user-images.githubusercontent.com/46878296/185958368-ca9e5bd4-99eb-4592-8f88-80d01cc06b9d.png)


```
 !python tools/infer.py --weights /content/drive/MyDrive/YOLO_6_Custom/YOLOv6/runs/train/exp/weights/best_ckpt.pt --yaml /content/drive/MyDrive/YOLO_6_Custom/dataset/trafic_data/data_1.yaml --source /content/drive/MyDrive/YOLO_6_Custom/dataset/trafic_data/images/test --device 0 

```
![image](https://user-images.githubusercontent.com/46878296/185960726-b4d5d6a3-34df-4ad3-825d-8b5dbacbcd38.png)


# FLASK API

- 1.Create the virtual envoriment or conda envoriment
- 2.Activate the envoriment
- 3.clone the YOLOV6 & install the requirements


```
 !git clone https://github.com/meituan/YOLOv6
  %cd YOLOv6
  %pip install -qr requirements.txt # install dependencie

```

- 4.Copy the custom weight into yolov6\include\predictor_yolo_detector
- 5.Clone the repo
Open the root directory in visual code or python IDE
Run the flask application as below

$ python clientApp.py

![image](https://user-images.githubusercontent.com/46878296/185961412-bc05c001-df0f-49e2-9b60-94ec39ffd443.png)



# Result

![image](https://user-images.githubusercontent.com/46878296/185969180-850107aa-e4d4-40b1-a514-f3060c0d98f5.png)

![image](https://user-images.githubusercontent.com/46878296/185970333-907d3dbe-0931-418e-9a38-d186146da22f.png)



# Video Inference

- clientApp.py do the modification as below
- uncomment the fileformat & provide the required link
- comment the fileformat = '' Line

![image](https://user-images.githubusercontent.com/46878296/185969671-881a2dbf-9d7a-431a-be22-89205730165b.png)


- Download the output.mp4 sample video and check


Reference or Original work:

https://github.com/meituan/YOLOv6
https://colab.research.google.com/notebooks/markdown_guide.ipynb#scrollTo=Lhfnlq1Surtk


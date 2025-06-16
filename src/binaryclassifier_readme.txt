collecting various images and preparing a dataset

data/
├── knee_xray/(475 images)
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── not_knee_xray/(472 images)
    ├── imageA.png
    ├── imageB.png
    └── ...

renaming the images in the dataset using bianaryclassifier.py file 


creating environment 
py -3.11 -m venv venv311

activating environment
venv311\Scripts\activate

update pip
python.exe -m pip install --upgrade pip

installing 
pip install tensorflow matplotlib scikit-learn

trained the basic model 
python binaryclassifier.py

testing model 
python binaryclassifier_testing_script.py imgpath

model is saved as knee_xray_classifier.h5
accuracy: 0.9962 - loss: 0.0127 - val_accuracy: 1.0000 - val_loss: 0.0160





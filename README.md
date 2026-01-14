# Masterplan Image Classifier

This project is an AI-based system that classifies real estate masterplan images.  
It helps identify layouts such as plots, roads, blocks, parks, and amenities.

## Features
- Classifies masterplan images automatically  
- Supports deep learning (CNN-based models)  
- Easy dataset structure for training and testing  
- High accuracy with labeled data  

## Tech Stack
- Python  
- TensorFlow / PyTorch  
- OpenCV  
- NumPy, Pandas  

## Folder Structure
masterplan-classifier/
│── data/
│ ├── train/
│ ├── test/
│── models/
│── notebooks/
│── app.py
│── requirements.txt
│── README.md

markdown
Copy code

## How to Run
1. Install dependencies  
pip install -r requirements.txt

markdown
Copy code
2. Train the model  
python train.py

csharp
Copy code
3. Test on images  
python app.py

markdown
Copy code

## Use Case
- Real estate platforms  
- Builders and developers  
- Property management systems  

## Output
The model predicts whether an image is a masterplan and classifies its layout type.

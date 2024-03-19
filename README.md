# README for Hate Speech Detection Project
Download the zip file from this repositry and the glove 6B file from stanford Glov or this link (https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt?resource=download)

## Overview
This project includes a set of Python scripts designed to train a machine learning model for hate speech detection, serve predictions using a FastAPI web application, and make requests to the web service. The project utilizes text data, pre-trained GloVe word embeddings, and TensorFlow for model training.

## Directory Structure

- Data: Contains the dataset and pre-trained GloVe embeddings.
  - `annotations_metadata.csv`: CSV file with text file identifiers and labels.
  - `sampled_train/`: Folder containing training text files.
  - `sampled_test/`: Folder containing test text files.
  - `glove.6B.100d.txt`: File containing GloVe word embeddings.
- Models: Contains the saved TensorFlow model files.
  - `Test6.h5`: The saved TensorFlow model from training.
- train_model.py: Script for training the hate speech detection model.
- app.py: FastAPI web application for serving model predictions.
- predict_model.py: Script for sending requests to the FastAPI web service.

## train_model.py

This script trains a TensorFlow model for hate speech detection. It performs the following steps:
- Cleans and preprocesses the text data.
- Loads and preprocesses the GloVe word embeddings.
- Trains a neural network model using the preprocessed text data.
- Saves the trained model to the `Models/` directory.

## app.py

This FastAPI web application serves predictions from the trained model. It supports POST and GET requests for analyzing text for hate speech. The application loads the trained model and a tokenizer fitted on the training data to preprocess incoming text for predictions.

## predict_model.py

This script demonstrates how to send requests to the FastAPI web service to get hate speech predictions for provided text. It includes examples of sending requests using both POST and GET methods.

## Setup and Execution

### Prerequisites
- Python 3.8 or later.
- TensorFlow 2.x.
- NLTK for text processing.
- FastAPI and Uvicorn for the web service.

### Installation
1. Install required Python packages:
```shell
pip install pandas numpy tensorflow nltk fastapi uvicorn
```
2. Download the GloVe embeddings from the provided link and place it in the `Data/` directory.

### Training the Model
Execute the `train_model.py` script to train and save the model. Ensure the dataset is correctly placed in the `Data/` directory.

```shell
python train_model.py
```

### Running the Web Service
Start the FastAPI web application using Uvicorn:

```shell
uvicorn app:app --reload
```

### Making Predictions
Use the `predict_model.py` script or utilize cURL / PowerShell commands as demonstrated in `app.py` to send requests to the running web service and receive hate speech predictions.

## Additional Notes
- Ensure all data paths in the scripts are correct according to your local setup.
- The model training script and FastAPI application require the GloVe embeddings and dataset to be in the specified directory structure.

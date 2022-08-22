# emotion-detection-and-reaction

Repository for Hot Topic Seminar Camera-based Multimodal Emotion Detection

## Objective

The objective of this project is to train and deploy a microservice that detects the current emotion of a user based on his voice and facial expression from his webcam and microphone inputs.

## Methodology

 - two separate models predict on the datastreams in parallel. The final class predictions are then combined.

## Training

 - chosen models: convolutional neural networks (CNNs) on both data streams
 - different datasets and feature sets
 - improved by augmentation techniques (see jupyter training files for more details)

## Deployment

 - deployed on a localhost using the Flask framework and javascript

## Installation instructions

 - git clone the repo
 - install the dependencies from requirements.txt
 - run the app.py file
 - open localhost 127.0.0.1:5000/ on your favorite web browser. 
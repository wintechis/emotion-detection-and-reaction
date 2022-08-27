# emotion-detection-and-reaction

Repository for Hot Topic Seminar Camera-based Multimodal Emotion Detection

## Objective

The objective of this project is to train and deploy a microservice that detects the current emotion of a user based on his voice and facial expression from his webcam and microphone inputs.

## Methodology

two separate models predict on the datastreams in parallel. The final class predictions are then combined.

## Training

 - chosen models: convolutional neural networks (CNNs) on both data streams
 - different datasets for audio: 
	- <a href="https://zenodo.org/record/1188976">Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)<a>
	- <a href="https://github.com/CheyneyComputerScience/CREMA-D">Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)<a>
	- <a href="https://tspace.library.utoronto.ca/handle/1807/24487">Toronto emotional speech set (TESS)<a>
	- <a href="http://kahlan.eps.surrey.ac.uk/savee/Database.html">Surrey Audio-Visual Expressed Emotion (SAVEE) Database<a>
	- <a href="http://emodb.bilderbar.info/docu/">Berlin Database of Emotional Speech (emo-db)<a>
 - and feature sets. for audio: Zero Crossing Rate, Chroma, MFCC, Root Mean Square Value, MelSpectrogram
 - improved by augmentation techniques: added noise, streched, shifted and pitched audio files (see jupyter training files for more details)

## Deployment Tech Stack

 deployed on a localhost using the Flask framework and javascript/ Ajax

## Requirements and installation instructions

 - a <a href="https://www.python.org/downloads/">python installation<a> is necessary (at least V3.7)
 - a working webcam and microphone is necessary. The input ports should be detected automatically.
 - git clone the repo
 - install the dependencies from requirements.txt
 - run the app.py file
 - open localhost <a href="https://127.0.0.1:5000/">127.0.0.1:5000/<a> on your favorite web browser. 
 
 ## Features
 
 - both models can be used standalone or together
 - 
 
 
## License

This project is licensed under the MIT License. See LICENSE for more details
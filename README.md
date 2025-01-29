# Image Colorization | Autoencoders & ResNet

This repository contains a project for image colorization using **Autoencoders** and **ResNet**. The goal of this project is to predict RGB color channels from grayscale images.

## Overview

This project utilizes an **Autoencoder-based approach** combined with **ResNet** for feature extraction. The model was trained on paired datasets of grayscale and color images. The input images were of size **256x256 with a single channel (grayscale)**, and the output images were of size **256x256 with three channels (RGB)**.

### Key Techniques:
- **LAB Color Space**: The model extracts the L channel (lightness) from LAB images as input and predicts the AB color channels.
- **ResNet Embeddings**: A ResNet classifier provides additional feature embeddings to enhance color prediction.
- **Fusion of Features**: The embeddings are combined with encoder outputs to improve colorization.
- **Autoencoder Architecture**: The model consists of an encoder-decoder structure for reconstructing color information.

## Dataset

The dataset consists of two folders:
- **Gray Images**: Contains the grayscale input images (256x256, 1 channel).
- **Color Images**: Contains the corresponding color images (256x256, 3 channels).

## Model Architecture

The model follows a hybrid approach:
- **Encoder**: Extracts important features from the grayscale input.
- **Fusion Layer**: Combines encoder output with ResNet embeddings.
- **Decoder**: Upsamples and reconstructs the image while adding color information.

## Data
- [Kaggle Dataset](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)
- [Kaggle Notebook](https://www.kaggle.com/code/alaagaberh/image-colorization-autoencoders-efficientnet)

## Deployment
- The model is deployed using **Streamlit** for an interactive user experience.

## Training Details
- The model is trained using **mean squared error loss** and the **Adam optimizer**.
- **Data augmentation** is applied for better generalization.
- The training process includes **checkpointing and early stopping** for optimal results.

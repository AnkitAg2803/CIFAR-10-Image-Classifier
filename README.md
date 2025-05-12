# 🧠 CIFAR-10 Image Classifier with Streamlit Deployment

This project demonstrates how to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow/Keras, save the trained model (`.h5`), and deploy it using **Streamlit** for live predictions via a web interface.

---

## 📚 Project Overview

- ✅ Train a CNN on CIFAR-10 image dataset
- ✅ Save the best performing model as `best_cifar10_model.h5`
- ✅ Extract class labels using TensorFlow Datasets + Pandas
- ✅ Save sample test images to disk
- ✅ Deploy model in a Streamlit web app
- ✅ Accept uploaded image, classify and return predicted label

---

## 📁 Dataset

The model is trained on the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

### Class Labels:
- `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

---

## 📦 Features

- Upload an image to classify it into one of the 10 CIFAR-10 categories
- Automatically resizes and preprocesses input image
- Uses trained deep learning model for real-time predictions
- Interactive and responsive web UI using Streamlit

---

## 🖼️ CIFAR-10 Class Labels

Automatically extracted via TensorFlow Datasets:


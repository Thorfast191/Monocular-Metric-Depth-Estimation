#!/bin/bash

# Install kaggle
pip install -q kaggle

# Make kaggle directory
mkdir -p ~/.kaggle

# Copy kaggle.json (must be uploaded manually first)
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d klemenko/kitti-depth

# Unzip to correct location
mkdir -p /content/kitti
unzip -q kitti-depth.zip -d /content/drive/MyDrive/kitti

# Project Title

# Terrain Recognition Project

## Overview

The Terrain Recognition Project is a computer vision-based system developed to identify and classify different types of terrain from images or video feeds. This project has applications in robotics, autonomous vehicles, and environmental monitoring.

## Table of Contents

- [Project Title](#project-title)
- [Terrain Recognition Project](#terrain-recognition-project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Dataset Overview](#dataset-overview)
- [Contributions](#Contributions)

## Project Description

The Terrain Recognition Project includes the following key features:

- Accurate classification of terrain types, including grass, pavement, sand, rocks, and more.
- Utilizes state-of-the-art computer vision techniques, such as deep learning and Convolutional Neural Networks (CNNs).
- Provides an easy-to-use system for both real-time and batch processing of images or video streams.
- Serves as a foundational tool for further research and development in related fields.

## Requirements

To run this project, you need the following software and hardware:

- Python 3.11.6
- TensorFlow 2.14.0
- OpenCV
- Numpy
- Matplotlib (for visualization)
- CUDA-compatible GPU (recommended for faster model training)

## Installation

To set up the project on your machine, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/terrain-recognition/terrain-recognition.git
   cd terrain-recognition

## Dataset Overview

This dataset is composed of 5000 image sets. Each set represents a random 512x512 pixel crop of the Earth and is composed of a Terrain map, a Height map, and a Segmentation map. Image sets are numbered 0000 through 5000 with a suffix of '_t' for terrain, '_h' for height, and '_i2' for segmentation. Crops were dynamically adjusted based on latitude to compensate for map projection distortion and maintain a relatively consistent land feature size across the dataset (1px ~ 400m).

Terrain Maps
Terrain Maps are colored based on land type with relief shading. Images are standard uint8 png format.

Height Maps
Height Maps encode altitude information through pixel value (0 being sea level). Note that these images are single channel, uint16.

Segmentation Maps
These were generated based on the terrain and height maps using unsupervised clustering and classification techniques for local pixel regions. In total, 7 terrain categories were defined for segmentation and associated with representative colorings as follows:

(17, 141, 215): Water
(225, 227, 155): Grassland
(127, 173, 123): Forest
(185, 122, 87): Hills
(230, 200, 181): Desert
(150, 150, 150): Mountain
(193, 190, 175): Tundra
Maps were then median filtered to remove noise and smooth out the features into larger blobs. Each segmentation map was created with some randomized parameters to create more variety across the dataset and ensure that image sets which happened to overlap would not have exactly the same segmentation map

## Contributions 
1. [SAHIL GOYAL](https://github.com/sahilgoyal7214)
2. [Vaibhav Sharma](https://github.com/vaibhav7766)
3. [Jayaditya Shukla](https://github.com/Jayaditya177)
4. [Suyash Tambe](https://github.com/suyashtambe)
5. [Tejas Tambe](https://github.com/AIMaster17)

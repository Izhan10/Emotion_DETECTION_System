# Emotion Detection System

## Introduction

This project implements a **real-time emotion detection system** that classifies facial expressions into **seven emotion categories** using a compact convolutional neural network (CNN). The model is trained on the **FER-2013** dataset and uses optimized NumPy-based operations for lightweight inference.

## Features

- 🎯 **7-class emotion classification**: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
- 📹 **Real-time detection**: Process live webcam feeds with minimal latency
- 🧠 **Compact CNN architecture**: Custom NumPy-based implementation for fast inference
- 👤 **Robust face detection**: Uses SSD (Single Shot MultiBox Detector) for accurate face localization
- ⚡ **No TensorFlow required**: Lightweight implementation using NumPy and OpenCV

## Dependencies

- Python 3
- [OpenCV](https://opencv.org/) - Computer vision library for image processing and face detection
- [NumPy](https://numpy.org/) - Numerical computations
- [h5py](https://www.h5py.org/) - Loading pre-trained model weights
- [Pillow](https://pillow.readthedocs.io/) - Image processing

To install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Emotion_DETECTION_System/
├── src/
│   ├── emotions.py              # Main script for real-time emotion detection
│   ├── nn.py                    # Custom CNN implementation with NumPy
│   ├── model.h5                 # Pre-trained model weights
│   ├── deploy.prototxt          # SSD model configuration
│   └── res10_300x300_ssd_iter_140000.caffemodel  # SSD face detector weights
├── imgs/                        # Sample images and results
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Usage

### Quick Start - Real-time Detection

1. **Clone the repository**:
```bash
git clone https://github.com/Izhan10/Emotion_DETECTION_System.git
cd Emotion_DETECTION_System
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download pre-trained weights** (if not included):
   - Download `model.h5` from [this link](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing)
   - Place it in the `src/` directory

4. **Run the emotion detector**:
```bash
cd src
python emotions.py --mode display
```

- The webcam feed will open in a window
- Detected faces will be highlighted with bounding boxes
- Emotion predictions will be displayed above each face
- Press `q` to quit

## How It Works

### Algorithm

1. **Face Detection**: 
   - Uses SSD (Single Shot MultiBox Detector) with a pre-trained ResNet-10 model
   - Detects faces in each frame of the webcam feed
   - Filters detections by confidence threshold (0.5)

2. **Preprocessing**:
   - Extracts detected face region
   - Converts to grayscale
   - Resizes to **48×48 pixels** (model input size)
   - Normalizes pixel values to [0, 1] range

3. **Emotion Classification**:
   - Passes preprocessed image through the CNN
   - Network outputs softmax scores for 7 emotion classes
   - Returns the emotion with the maximum score

### Model Architecture

The CNN consists of:
- **Convolutional layers**: Extract spatial features from face images
- **Max pooling layers**: Reduce dimensionality and computational cost
- **Dense layers**: Classify learned features into emotion categories
- **ReLU activation**: Non-linearity for feature learning
- **Softmax output**: Probability distribution over 7 emotions

**Input**: 48×48 grayscale image  
**Output**: 7-dimensional probability vector (one per emotion class)

## Emotion Classes

| ID | Emotion | ID | Emotion |
|----|---------|----|---------| 
| 0  | Angry   | 3  | Happy   |
| 1  | Disgusted | 4 | Neutral |
| 2  | Fearful | 5  | Sad     |
| 6  | Surprised | - | -     |

## Performance

- The custom NumPy-based CNN achieves competitive accuracy on the FER-2013 test set
- Lightweight implementation enables real-time performance on standard hardware
- Face detection confidence threshold set to 0.5 for balanced precision/recall

## Technical Details

### Face Detection
- **Detector**: ResNet-10 based SSD
- **Input size**: 300×300 pixels
- **Framework**: OpenCV DNN module (Caffe format)

### Emotion CNN
- **Framework**: Custom NumPy implementation
- **Model format**: HDF5 (.h5)
- **Weight loading**: h5py library
- **Inference**: Pure NumPy operations (conv2d, pooling, dense layers)

## References

- "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li, X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu, M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and Y. Bengio. arXiv 2013.
- [FER-2013 Dataset](https://www.kaggle.com/deadskull7/fer2013) - Kaggle dataset used for training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

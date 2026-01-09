# Cat vs Dog Image Classifier

A web application that classifies images of cats and dogs using deep learning and transfer learning techniques.

## About

I built this project to learn about deploying machine learning models as web applications. The classifier uses a pre-trained MobileNetV2 model and achieves strong accuracy on the binary classification task.

## Features

- Upload an image and get instant predictions
- See confidence scores for each prediction
- Clean, simple interface built with Streamlit
- Uses transfer learning with MobileNetV2

## Tech Stack

- Python 3.12
- TensorFlow 2.15
- Streamlit
- Pillow for image processing
- Plotly for visualizations

## Installation

Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/cat-dog-classifier.git
cd cat-dog-classifier
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run cat_dog_classifier_app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

The application uses transfer learning with MobileNetV2, a lightweight convolutional neural network pre-trained on ImageNet. I fine-tuned it for binary classification (cat vs dog) and deployed it using Streamlit for the web interface.

When you upload an image:
1. The image is resized to 224x224 pixels
2. Pixel values are normalized
3. The model makes a prediction
4. Results are displayed with confidence scores

## Model Details

- Base architecture: MobileNetV2
- Input size: 224x224x3
- Output: Binary classification (Cat or Dog)
- Framework: TensorFlow/Keras

## Project Structure

```
cat-dog-classifier/
├── cat_dog_classifier_app.py    # Main application
├── models/
│   └── cat_dog_model.keras      # Trained model
├── requirements.txt
└── README.md
```

## What I Learned

Working on this project helped me understand:
- How to implement transfer learning with pre-trained models
- Deploying ML models as web applications
- Image preprocessing and normalization
- Building interactive UIs with Streamlit

## Future Improvements

- Add support for more animal classes
- Implement batch image processing
- Add confidence threshold controls
- Deploy to cloud platform

## Contact

Feel free to reach out if you have questions or suggestions!

- Email: Patel.mun25@gmail.com

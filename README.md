# 🌿 Lightweight Plant Leaf Disease Detector with Grad-CAM

A deep learning-powered web application designed to diagnose plant diseases from leaf images, utilizing Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the areas of the leaf that influenced the model's decision.

![App Header](https://raw.githubusercontent.com/AbidHasanRafi/lightweight-plant-leaf-disease-detector/main/assets/header.png)

## Key Features

- **Accurate Disease Detection**: Diagnoses 20 common plant diseases across multiple species with high precision.
- **Model Explainability**: Utilizes Grad-CAM to visually highlight the areas of the leaf that most influenced the model’s predictions.
- **Responsive Design**: Ensures smooth user experience on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's preferred theme.
- **Confidence Metrics**: Displays prediction probabilities to showcase the model's certainty.

## Model Architecture

The application leverages a custom lightweight Convolutional Neural Network (CNN) designed for efficiency and performance:
- **Depthwise Separable Convolutions**: Reduces computational cost while maintaining accuracy.
- **Attention Mechanism**: Enhances feature learning and improves model performance.
- **Compact Architecture**: The model consists of only 128,000 parameters, making it highly efficient.

![Grad-CAM Visualization](https://raw.githubusercontent.com/AbidHasanRafi/lightweight-plant-leaf-disease-detector/main/assets/grad-cam.png)

## Quick Start

To set up and run the application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AbidHasanRafi/lightweight-plant-leaf-disease-detector.git
   cd lightweight-plant-leaf-disease-detector
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

## Project Structure

```
lightweight-plant-leaf-disease-detector/
├── README.md
├── app.py
├── lightweight_plant_cnn.pth
└── requirements.txt
```

## Usage Instructions

1. **Upload Model**: The pretrained model (`lightweight_plant_cnn.pth`) is included in the repository.
2. **Upload Leaf Image**: Choose a plant leaf image (JPG/PNG format) for diagnosis.
3. **View Results**: The app will display the diagnosis along with the model's confidence score.
4. **Analyze Attention**: Use the Grad-CAM visualization to see which areas of the leaf were most influential in the prediction.

![Results Display](https://raw.githubusercontent.com/AbidHasanRafi/lightweight-plant-leaf-disease-detector/main/assets/result.png)

## Technologies Used

- **PyTorch**: For model development and inference.
- **Streamlit**: Framework for building interactive web applications.
- **Grad-CAM**: Technique for visualizing model attention and explaining predictions.
- **Matplotlib**: Library used for generating visualizations.

## Supported Diseases

The model can detect the following 20 diseases across multiple plant species:

- **Apple**: Scab, Black Rot, Cedar Rust
- **Bell Pepper**: Bacterial Spot
- **Corn**: Common Rust, Northern Blight
- **Grape**: Black Rot, Leaf Blight
- **Potato**: Early Blight, Late Blight
- **Tomato**: Bacterial Spot, Leaf Curl Virus

## Contact

For questions or suggestions, please feel free to reach out:

- **Md. Abid Hasan Rafi**: [GitHub](https://github.com/AbidHasanRafi) · [Website](https://abidhasanrafi.github.io) · [Email](mailto:ahr16.abidhasanrafi@gmail.com)
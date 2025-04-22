# 🌿 Lightweight Plant Leaf Disease Detector

A deep learning-powered web application designed to diagnose plant diseases from leaf images, utilizing Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the areas of the leaf that influenced the model's decision.

![App Header](https://raw.githubusercontent.com/AbidHasanRafi/lightweight-plant-leaf-disease-detector/main/assets/header.png)

## Key Features

- **Accurate Disease Detection**: Diagnoses 20 common plant diseases across multiple species with high precision.
- **Model Explainability**: Utilizes Grad-CAM to visually highlight the areas of the leaf that most influenced the model’s predictions.
- **Responsive Design**: Ensures smooth user experience on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's preferred theme.
- **Confidence Metrics**: Displays prediction probabilities to showcase the model's certainty.

## Dataset

The model is trained using a dataset that contains images of plant leaves, labeled with their corresponding diseases. You can test the app using the images from the dataset. The dataset can be accessed from the following Kaggle link:

[Plant Village Dataset (Updated)](https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated)

## Model Architecture
```python
# model architecture
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.features[:-1](x)
        attention = self.features[-1](features)
        attended = features * attention
        return self.classifier(attended)
```

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
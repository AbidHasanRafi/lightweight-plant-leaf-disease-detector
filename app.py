import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# class names
CLASS_NAMES = [
    'Apple_Apple Scab', 'Apple_Black Rot', 'Apple_Cedar Apple Rust',
    'Bell Pepper_Bacterial Spot', 'Cherry_Powdery Mildew',
    'Corn (Maize)_Cercospora Leaf Spot', 'Corn (Maize)_Common Rust',
    'Corn (Maize)_Northern Leaf Blight', 'Grape_Black Rot',
    'Grape_Esca (Black Measles)', 'Grape_Leaf Blight',
    'Peach_Bacterial Spot', 'Potato_Early Blight', 'Potato_Late Blight',
    'Strawberry_Leaf Scorch', 'Tomato_Bacterial Spot',
    'Tomato_Early Blight', 'Tomato_Late Blight',
    'Tomato_Septoria Leaf Spot', 'Tomato_Yellow Leaf Curl Virus'
]

# load model function
@st.cache_resource
def load_model(model_path):
    try:
        model = PlantDiseaseCNN(num_classes=len(CLASS_NAMES)).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        # forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # zero gradients
        self.model.zero_grad()
        
        # backward pass for specific class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        
        # get pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # weight the activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

def create_combined_figure(original_image, heatmap, alpha=0.5):
    """Create a responsive combined figure with original and heatmap"""
    # convert to numpy arrays
    img_array = np.array(original_image)
    
    # resize heatmap to match image dimensions
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(original_image.size, Image.BILINEAR))
    
    # normalize heatmap
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    
    # create figure with dark mode compatibility
    plt.style.use('dark_background' if st.get_option("theme.base") == "dark" else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.05})
    
    # display original image
    ax1.imshow(img_array)
    ax1.set_title("Original Image", fontsize=12, pad=10, fontweight='bold', 
                 color='white' if st.get_option("theme.base") == "dark" else 'black')
    ax1.axis('off')
    
    # display original image with heatmap overlay
    ax2.imshow(img_array)
    heatmap_display = ax2.imshow(heatmap_normalized, cmap='inferno', alpha=alpha)
    ax2.set_title("Model Attention", fontsize=12, pad=10, fontweight='bold',
                 color='white' if st.get_option("theme.base") == "dark" else 'black')
    ax2.axis('off')
    
    # add colorbar with smaller size
    cbar = fig.colorbar(heatmap_display, ax=ax2, fraction=0.046, pad=0.01)
    cbar.ax.tick_params(labelsize=8, colors='white' if st.get_option("theme.base") == "dark" else 'black')
    
    plt.tight_layout()
    return fig

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def main():
    st.set_page_config(
        layout="wide", 
        page_title="Plant Disease Diagnosis", 
        page_icon="🌿",
        initial_sidebar_state="expanded"
    )
    
    # custom CSS for dark mode compatibility
    st.markdown("""
    <style>
    /* Main content - works in both light and dark modes */
    .main {padding: 2rem;}
    .block-container {padding-top: 2rem;}
    
    /* Cards - dark mode compatible */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
    }
    
    /* Text colors that work in both modes */
    .card h1, .card h2, .card h3, .card p {
        color: var(--text-color) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #4caf50;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #81c784;
        border-radius: 10px;
        padding: 2rem 1rem;
        background-color: transparent;
    }
    
    /* Theme variables */
    :root {
        --background-color: white;
        --text-color: black;
        --border-color: #e0e0e0;
    }
    
    [data-theme="dark"] {
        --background-color: #1e1e1e;
        --text-color: white;
        --border-color: #444;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main {padding: 1rem;}
        .sidebar .sidebar-content {padding: 1rem;}
    }
    </style>
    """, unsafe_allow_html=True)
    
    # app header
    st.markdown("""
    <div class="card">
        <h1 style="margin-bottom: 0.5rem; color: var(--text-color) !important;">🌿 Plant Disease Diagnosis</h1>
        <p style="color: var(--text-color); margin-bottom: 0;">Upload a plant image to detect diseases and visualize model attention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # sidebar - Model upload
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h3>1. Upload Model</h3>
            <p style="color: var(--text-color);">Please upload your trained model file (.pth)</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_file = st.file_uploader(
            "Choose a model file", 
            type="pth",
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div class="card">
            <h3>About</h3>
            <p style="color: var(--text-color);">This app uses deep learning to:</p>
            <ul style="color: var(--text-color);">
                <li>Classify plant diseases</li>
                <li>Visualize model attention</li>
                <li>Explain predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # main content
    if model_file is not None:
        # save uploaded model to temporary file
        model_path = "temp_model.pth"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        
        model = load_model(model_path)
        
        if model is not None:
            st.success("Model loaded successfully!", icon="✅")
            
            # initialize Grad-CAM
            target_layer = model.features[-2]  # Layer before attention
            grad_cam = GradCAM(model, target_layer)
            
            # image upload section
            st.markdown("""
            <div class="card">
                <h2>2. Upload Plant Image</h2>
                <p style="color: var(--text-color);">Upload an image of a plant leaf for disease diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
            
            image_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if image_file is not None:
                image = Image.open(image_file).convert('RGB')
                
                # process image
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    top_prob, top_class = torch.max(probabilities, 1)
                
                # generate Grad-CAM heatmap
                heatmap = grad_cam(input_tensor, top_class.item())
                
                # visualization section
                st.markdown("""
                <div class="card">
                    <h2>Image Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # combined visualization
                fig = create_combined_figure(image, heatmap)
                st.pyplot(fig, use_container_width=True)
                
                # results section
                col1, col2 = st.columns([1, 2], gap="large")
                
                with col1:
                    st.markdown(f"""
                    <div class="card">
                        <h3>🔍 Diagnosis Results</h3>
                        <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-color);"><b>Predicted Disease:</b></p>
                        <p style="font-size: 1.3rem; color: #4caf50; font-weight: bold; margin-top: 0;">{CLASS_NAMES[top_class.item()]}</p>
                        <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-color);"><b>Confidence:</b></p>
                        <p style="font-size: 1.3rem; color: #4caf50; font-weight: bold; margin-top: 0;">{top_prob.item()*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="card">
                        <h3>📊 Top Predictions</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    probs = probabilities.squeeze().cpu().numpy()
                    top_indices = np.argsort(probs)[-5:][::-1]
                    
                    for i in top_indices:
                        label = CLASS_NAMES[i]
                        percent = probs[i]*100
                        st.markdown(f"<span style='color: var(--text-color);'><b>{label}</b></span>", unsafe_allow_html=True)
                        st.progress(float(probs[i]), text=f"{percent:.2f}%")
                
                # interpretation section
                st.markdown("""
                <div class="card">
                    <h3>🔎 Understanding the Results</h3>
                    <p style="color: var(--text-color);">The attention map shows which areas of the image most influenced the model's prediction:</p>
                    <div style="display: flex; justify-content: center; margin: 1rem 0;">
                        <div style="text-align: center; margin: 0 1rem;">
                            <div style="width: 20px; height: 20px; background-color: #d62728; display: inline-block; border-radius: 3px;"></div>
                            <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-color);">Most important</p>
                        </div>
                        <div style="text-align: center; margin: 0 1rem;">
                            <div style="width: 20px; height: 20px; background-color: #ff7f0e; display: inline-block; border-radius: 3px;"></div>
                            <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-color);">Important</p>
                        </div>
                        <div style="text-align: center; margin: 0 1rem;">
                            <div style="width: 20px; height: 20px; background-color: #1f77b4; display: inline-block; border-radius: 3px;"></div>
                            <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-color);">Less important</p>
                        </div>
                    </div>
                    <p style="color: var(--text-color);">The color intensity corresponds to the relative importance of each region in the diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # clean up temporary model file
        if os.path.exists(model_path):
            os.remove(model_path)
    else:
        st.info("Please upload a trained model (.pth file) to begin diagnosis", icon="ℹ️")

if __name__ == "__main__":
    main()
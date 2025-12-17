"""
Streamlit App for Glaucoma Detection
Upload an image (NPZ format or regular image) and get glaucoma prediction
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os
from utils.model import create_model

# Set page config
st.set_page_config(
    page_title="Glaucoma Detection",
    page_icon="👁️",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_model(checkpoint_path, use_metadata=True):
    """Load model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check checkpoint format
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Standard checkpoint format (best_model.pt)
        model = create_model(
            num_classes=2,
            pretrained=False,
            use_metadata=use_metadata,
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict format (best_glaucoma_2d.pth)
        # Try to create model and load state dict
        try:
            # Try with metadata first
            model = create_model(
                num_classes=2,
                pretrained=False,
                use_metadata=use_metadata,
                device=device
            )
            # Try loading as direct state dict
            if isinstance(checkpoint, dict):
                try:
                    model.load_state_dict(checkpoint, strict=False)
                except Exception as e1:
                    # If that fails, try without metadata
                    if use_metadata:
                        model = create_model(
                            num_classes=2,
                            pretrained=False,
                            use_metadata=False,
                            device=device
                        )
                        model.load_state_dict(checkpoint, strict=False)
                    else:
                        raise e1
            else:
                st.error(f"Unknown checkpoint format for {checkpoint_path}")
                return None
        except Exception as e:
            st.warning(f"Could not load {checkpoint_path} with standard model. Error: {e}")
            return None
    
    model.eval()
    return model, device

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image: numpy array or PIL Image
        target_size: target size (height, width)
    
    Returns:
        Preprocessed tensor (1, 1, 224, 224)
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))  # Convert to grayscale
    
    # Ensure it's a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Handle different input shapes
    if len(image.shape) == 3:
        # If RGB, convert to grayscale
        if image.shape[2] == 3:
            image = np.mean(image, axis=2)
        else:
            image = image[:, :, 0]
    
    # Normalize to [0, 1]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Convert to tensor and add channel dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
    
    # Resize to target size
    if image_tensor.shape[1] != target_size[0] or image_tensor.shape[2] != target_size[1]:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # Add batch dimension: (1, 1, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def preprocess_npz(npz_file):
    """
    Preprocess NPZ file
    
    Args:
        npz_file: uploaded file object
    
    Returns:
        image_tensor, metadata (age, md)
    """
    # Load NPZ file
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract RNFL thickness map
    if 'rnflt' in data:
        rnflt = data['rnflt'].astype(np.float32)
    else:
        st.error("NPZ file does not contain 'rnflt' key")
        return None, None
    
    # Normalize
    rnflt = (rnflt - rnflt.min()) / (rnflt.max() - rnflt.min() + 1e-8)
    
    # Convert to tensor
    rnflt_tensor = torch.from_numpy(rnflt).unsqueeze(0)  # (1, H, W)
    
    # Resize to 224x224
    if rnflt_tensor.shape[1] != 224 or rnflt_tensor.shape[2] != 224:
        rnflt_tensor = F.interpolate(
            rnflt_tensor.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # Add batch dimension
    rnflt_tensor = rnflt_tensor.unsqueeze(0)  # (1, 1, 224, 224)
    
    # Extract metadata if available
    age = float(data.get('age', 0.0))
    md = float(data.get('md', 0.0))
    metadata = torch.tensor([[age, md]], dtype=torch.float32)
    
    return rnflt_tensor, metadata

def predict(model, image_tensor, metadata=None, device='cpu'):
    """
    Make prediction using the model
    
    Args:
        model: loaded model
        image_tensor: preprocessed image tensor (1, 1, 224, 224)
        metadata: optional metadata tensor (1, 2) with [age, md]
        device: device to run inference on
    
    Returns:
        probabilities, prediction
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # If model expects metadata but none provided, use zeros
        if hasattr(model, 'use_metadata') and model.use_metadata:
            if metadata is None:
                # Create zero metadata tensor if model expects it
                metadata = torch.zeros((1, 2), dtype=torch.float32)
            metadata = metadata.to(device)
            output = model(image_tensor, metadata)
        else:
            # Model doesn't use metadata
            output = model(image_tensor)
        
        probs = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1).item()
        prob_glaucoma = probs[0, 1].item()
        prob_no_glaucoma = probs[0, 0].item()
    
    return prob_glaucoma, prob_no_glaucoma, predicted

# Main app
def main():
    st.title("👁️ Glaucoma Detection App")
    st.markdown("Upload an image (NPZ format or regular image) to detect glaucoma")
    
    # Load model
    checkpoint_path = "checkpoints/best_model.pt"
    
    model = None
    device = None
    use_metadata = True
    
    # Try to load model
    if os.path.exists(checkpoint_path):
        try:
            model, device = load_model(checkpoint_path, use_metadata=True)
            if model is None:
                st.error(f"Failed to load model from {checkpoint_path}")
                return
        except Exception as e:
            st.error(f"Could not load {checkpoint_path}: {e}")
            return
    else:
        st.error(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    st.success("Model loaded successfully!")
    
    # File upload
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['npz', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        # Display uploaded file info
        st.header("Uploaded File")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File type:** {uploaded_file.name.split('.')[-1].upper()}")
        
        # Process file based on type
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'npz':
            # Process NPZ file
            try:
                image_tensor, metadata = preprocess_npz(uploaded_file)
                if image_tensor is None:
                    st.error("Failed to process NPZ file")
                    return
                
                # Display image
                with col2:
                    img_display = image_tensor[0, 0].cpu().numpy()
                    st.image(img_display, caption="RNFL Thickness Map", use_container_width=True)
                
                # Show metadata from NPZ
                if metadata is not None:
                    age_val = metadata[0, 0].item()
                    md_val = metadata[0, 1].item()
                    st.info(f"**Age:** {age_val:.1f} years | **MD:** {md_val:.2f} dB")
                
            except Exception as e:
                st.error(f"Error processing NPZ file: {e}")
                return
        else:
            # Process regular image
            try:
                image = Image.open(uploaded_file)
                
                # Display image
                with col2:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                # No metadata for regular images
                metadata = None
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                return
        
        # Make prediction
        st.header("Prediction")
        
        # Use metadata only if available from NPZ file
        meta_to_use = metadata if metadata is not None else None
        
        try:
            prob_glaucoma, prob_no_glaucoma, predicted = predict(
                model, image_tensor, meta_to_use, device
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Glaucoma Probability",
                    f"{prob_glaucoma*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "No Glaucoma Probability",
                    f"{prob_no_glaucoma*100:.2f}%"
                )
            
            with col3:
                prediction = "Glaucoma" if predicted == 1 else "No Glaucoma"
                color = "🔴" if predicted == 1 else "🟢"
                st.metric("Prediction", f"{color} {prediction}")
            
            # Progress bar
            st.progress(prob_glaucoma)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()


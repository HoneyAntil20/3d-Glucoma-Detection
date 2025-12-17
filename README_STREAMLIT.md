# Glaucoma Detection Streamlit App

A Streamlit web application for detecting glaucoma from retinal images.

## Features

- Upload images in NPZ format (with RNFL thickness maps) or regular image formats (PNG, JPG, etc.)
- Uses two trained models for prediction:
  - `best_model.pt` - Full model with metadata support
  - `best_glaucoma_2d.pth` - Alternative model
- Displays predictions with confidence scores
- Shows consensus prediction when multiple models are available

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

1. Make sure the checkpoint files are in the `checkpoints/` directory:
   - `checkpoints/best_model.pt`
   - `checkpoints/best_glaucoma_2d.pth`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown (typically `http://localhost:8501`)

## Usage

1. **Upload an image:**
   - For NPZ files: Upload a `.npz` file containing RNFL thickness maps. The app will automatically extract the image and metadata (age, MD).
   - For regular images: Upload PNG, JPG, or other image formats. You can optionally provide age and mean deviation (MD) in the sidebar.

2. **View predictions:**
   - The app will show predictions from all loaded models
   - Each prediction includes:
     - Glaucoma probability
     - No glaucoma probability
     - Final prediction (Glaucoma/No Glaucoma)
   - If multiple models are loaded, a consensus prediction is also shown

## File Formats

### NPZ Format
NPZ files should contain:
- `rnflt`: RNFL thickness map (numpy array, typically 225x225)
- `age`: Patient age (float)
- `md`: Mean deviation (float)
- Optional: `glaucoma`, `tds`, etc.

### Regular Images
- Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP
- Images will be converted to grayscale and resized to 224x224
- Metadata (age, MD) can be provided in the sidebar

## Notes

- The app automatically handles different checkpoint formats
- If a model fails to load, the app will continue with the successfully loaded models
- Predictions are made using softmax probabilities
- The consensus prediction averages probabilities from all models


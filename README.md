👁️ Glaucoma Detection Using Deep Learning

This project presents an end-to-end deep learning system for glaucoma detection using RNFL (Retinal Nerve Fiber Layer) thickness maps derived from retinal imaging.
The system supports training, evaluation, visualization, and real-time inference via a Streamlit web application.

📌 Project Highlights

🔬 Binary Classification: Glaucoma vs No Glaucoma

🧠 Deep Learning Model (PyTorch)

📊 Uses RNFL Thickness Maps as primary input

🧾 Optional clinical metadata (Age, Mean Deviation – MD)

🌐 Streamlit Web App for interactive predictions

📈 Balanced Prediction Visualization for qualitative analysis

📁 Modular scripts for training, testing, and evaluation

📷 Sample Balanced Prediction Visualization

The figure below shows balanced glaucoma and non-glaucoma predictions with correctness highlighted:

🟢 Green border → Correct prediction

🔴 Red border → Incorrect prediction

Heatmap → Normalized RNFL thickness

Correct: 9/10 (90%)


This visualization is generated using:

python visualize_balanced_predictions.py

📂 Project Structure
├── app.py                         # Streamlit application
├── train.py                       # Model training script
├── evaluate.py                    # Validation & test evaluation
├── test.py                        # Test-only evaluation
├── visualize_balanced_predictions.py  # Prediction visualization
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
├── checkpoints/
│   └── best_model.pt              # Trained model checkpoint
├── dataset-001/
│   ├── train/
│   ├── val/
│   └── test/
└── utils/
    ├── model.py
    ├── dataset.py
    ├── trainer.py
    └── evaluator.py

⚙️ Installation
1️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

🚀 Running the Streamlit App

Make sure the trained model exists:

checkpoints/best_model.pt


Then run:

streamlit run app.py


📍 Open browser at:
http://localhost:8501

🖼️ Input Formats Supported
✅ NPZ Files (Recommended)

Must contain:

rnflt → RNFL thickness map (2D numpy array)

age → Patient age

md → Mean Deviation (visual field)

✅ Image Files

Formats supported:

PNG, JPG, JPEG, TIFF, BMP


Images are automatically converted to grayscale and resized to 224 × 224.

🧠 Model Training

To train the model from scratch:

python train.py \
  --data_dir dataset-001 \
  --batch_size 32 \
  --num_epochs 40 \
  --learning_rate 1e-4


The best model is automatically saved in:

checkpoints/best_model.pt

📊 Model Evaluation
Validation + Test Evaluation
python evaluate.py --data_dir dataset-001

Test Only
python test.py --data_dir dataset-001


Metrics reported:

Accuracy

Precision

Recall

F1-score

AUC-ROC

🎯 Balanced Prediction Visualization

To generate the visualization shown in the image:

python visualize_balanced_predictions.py \
  --data_dir dataset-001 \
  --num_images 10 \
  --split test


Output:

balanced_predictions_test.png

🧪 Technologies Used

Python

PyTorch

NumPy

Matplotlib

Streamlit

Pillow

📌 Key Features Summary

✔ RNFL-based glaucoma detection
✔ Metadata-enhanced classification
✔ Visual explainability via heatmaps
✔ Streamlit-based real-time inference
✔ Academic & project-ready structure

📜 Disclaimer

⚠️ This system is intended for educational and research purposes only.
It is not a substitute for professional medical diagnosis.

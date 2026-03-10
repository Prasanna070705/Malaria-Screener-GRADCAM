# Malaria Screener App

A Streamlit web app for malaria parasite and WBC detection using YOLOv8 and CNN refinement.

## Features
- Upload a single blood smear image
- YOLOv8 detects parasites and WBCs
- CNN refinement classifier reduces false positives
- Annotated results shown in browser

## Usage
1. Clone this repo and add your model files:
   - `malaria-screener-GitHub/yolo_malaria_best.pt`
   - `malaria-screener-GitHub/refinement_cnn_best.pth`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment
- You can deploy on Streamlit Cloud, Hugging Face Spaces, or use GitHub Actions for static hosting.

## Credits
- YOLOv8: Ultralytics
- CNN refinement: Custom PyTorch model

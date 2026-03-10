import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
YOLO_MODEL_PATH = 'malaria-screener-GitHub/yolo_malaria_best.pt'
CNN_MODEL_PATH = 'malaria-screener-GitHub/refinement_cnn_best.pth'

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource
def load_cnn():
    import torch.nn as nn
    class RefinementCNN(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    model = RefinementCNN(num_classes=3)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

CLASS_NAMES = ['parasite', 'wbc', 'artifact']

st.title('Malaria Screener App')
st.write('Upload a blood smear image. The app will detect parasites/WBCs and refine predictions with a CNN.')

uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'tiff'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    yolo = load_yolo()
    cnn = load_cnn()

    # Run YOLO detection
    results = yolo.predict(img_np, conf=0.15, verbose=False)
    boxes = results[0].boxes
    refined_labels = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            refined_labels.append('artifact')
            continue
        crop_resized = cv2.resize(crop, (64, 64))
        crop_tensor = torch.from_numpy(crop_resized.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            logits = cnn(crop_tensor)
            pred = logits.argmax(dim=1).item()
        refined_labels.append(CLASS_NAMES[pred])

    # Annotate image
    annotated = img_np.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        label = refined_labels[i]
        color = (0,255,0) if label=='parasite' else (0,0,255) if label=='wbc' else (255,255,0)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    st.image(annotated, caption='Detections with CNN refinement', use_column_width=True)
    st.write(f'Detections: {len(boxes)}')
    st.write('Class counts:', {k: refined_labels.count(k) for k in CLASS_NAMES})

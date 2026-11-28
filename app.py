import os, io, torch, torch.nn as nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import models, transforms
import numpy as np
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# 1. Load the model
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 4)
model.load_state_dict(torch.load('cnn_classifier.pth', map_location='cpu'))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
CLASSES = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']

def is_likely_mri(img):
    # Convert to grayscale and RGB
    arr = np.array(img.convert('L'), dtype=np.float32)
    img_rgb = np.array(img.convert('RGB'), dtype=np.float32)
    
    # 1. COLOR CHECK
    sat = np.mean((img_rgb.max(2) - img_rgb.min(2)) / (img_rgb.max(2) + 1e-6))
    if sat > 0.15:
        return False, "The image appears to be colored. Please upload a grayscale MRI scan."
    
    # 2. BRIGHTNESS CHECK
    mean_brightness = np.mean(arr)
    if mean_brightness > 245:
        return False, "The image is too bright (likely a blank document or screenshot)."
    if mean_brightness < 5:
        return False, "The image is completely black."
        
    # 3. CONTRAST CHECK
    std_dev = np.std(arr)
    if std_dev < 10: 
        return False, "The image lacks contrast."

    # 4. CORNER CHECK
    h, w = arr.shape
    corner_size = 20
    top_left = arr[:corner_size, :corner_size]
    top_right = arr[:corner_size, -corner_size:]
    bottom_left = arr[-corner_size:, :corner_size]
    bottom_right = arr[-corner_size:, -corner_size:]
    
    corners_mean = np.mean([np.mean(top_left), np.mean(top_right), np.mean(bottom_left), np.mean(bottom_right)])
    if corners_mean > 80: 
        return False, "The image does not have the dark background typical of an MRI scan."

    # 5. THE NUMPY "SOLID BRAIN" CHECK (Rejects screenshots)
    # Extract the central 50% of the image (where the brain always is)
    center_crop = arr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    
    # Count how many pixels in the center are actual tissue (above pure black)
    tissue_pixels = np.sum(center_crop > 40)
    total_center_pixels = center_crop.size
    tissue_density = tissue_pixels / total_center_pixels
    
    # A brain fills up more than 50% of the center of the image.
    # A screenshot has mostly black space in the center with a few thin letters.
    if tissue_density < 0.40:
        return False, "The image does not contain the solid central anatomical mass of a brain MRI."

    return True, "OK"
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        
        # Run the updated Validation Gate
        valid, msg = is_likely_mri(img)
        if not valid:
            return jsonify({'error': msg}), 400
            
        # Run the AI Model
        with torch.no_grad():
            out = model(tf(img).unsqueeze(0))
            prob = torch.softmax(out, dim=1)
            
        pred = prob.argmax(1).item()
        confidence = float(prob.max())
        
        # Catch uncertain predictions
        return jsonify({'label': CLASSES[pred], 'confidence': confidence})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)))
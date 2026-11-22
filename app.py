import os, io, torch, torch.nn as nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import models, transforms
import numpy as np

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
    # Convert to grayscale
    arr = np.array(img.convert('L'), dtype=np.float32)
    
    # 1. Color Check
    img_rgb = np.array(img.convert('RGB'), dtype=np.float32)
    sat = np.mean((img_rgb.max(2) - img_rgb.min(2)) / (img_rgb.max(2) + 1e-6))
    if sat > 0.15:
        return False, "The image appears to be colored. Please upload a grayscale MRI scan."
    
    # 2. UPDATED: Wider Brightness Check
    # Changed from 230/15 to 245/5 to allow darker/brighter valid MRIs
    mean_brightness = np.mean(arr)
    if mean_brightness > 245:
        return False, "The image is too bright (likely a blank document or screenshot)."
    if mean_brightness < 5:
        return False, "The image is completely black. Please upload a proper MRI scan."
        
    # 3. UPDATED: Lower Contrast Check
    # Changed from 25 to 10 so it only catches completely flat images
    std_dev = np.std(arr)
    if std_dev < 10: 
        return False, "The image lacks contrast. Please upload a clearer medical image."
        
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
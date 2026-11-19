import os, io, torch, torch.nn as nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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
    import numpy as np
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    sat = np.mean((arr.max(2) - arr.min(2)) / (arr.max(2) + 1e-6))
    if sat > 0.15:
        return False, "The image appears to be colored. Please upload a grayscale MRI scan."
    mean_bright = np.mean(arr)
    if mean_bright > 230 or mean_bright < 15:
        return False, "Image brightness is abnormal. Please upload a proper MRI scan."
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
        valid, msg = is_likely_mri(img)
        if not valid:
            return jsonify({'error': msg}), 400
        with torch.no_grad():
            out = model(tf(img).unsqueeze(0))
            prob = torch.softmax(out, dim=1)
        pred = prob.argmax(1).item()
        return jsonify({'label': CLASSES[pred], 'confidence': float(prob.max())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)))
from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model
class MasterplanModel:
    def __init__(self, model_path='masterplan_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """Load trained model"""
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.class_names = checkpoint.get('class_names', ['masterplan', 'not_masterplan'])
        return model
    
    def predict(self, image):
        """Predict if image is a masterplan"""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        is_masterplan = self.class_names[predicted.item()] == 'masterplan'
        confidence_score = confidence.item() * 100
        
        # Get both class probabilities
        masterplan_prob = probabilities[0][0].item() * 100 if self.class_names[0] == 'masterplan' else probabilities[0][1].item() * 100
        not_masterplan_prob = 100 - masterplan_prob
        
        return {
            'is_masterplan': is_masterplan,
            'confidence': confidence_score,
            'masterplan_probability': masterplan_prob,
            'not_masterplan_probability': not_masterplan_prob,
            'predicted_class': self.class_names[predicted.item()]
        }

# Initialize model
try:
    predictor = MasterplanModel('masterplan_model.pth')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    predictor = None

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Masterplan Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .upload-section {
            background: #f8f9fa;
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-section:hover {
            background: #e9ecef;
            border-color: #764ba2;
        }
        
        .upload-section.drag-over {
            background: #d4e7ff;
            border-color: #0066cc;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            margin: 10px;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-clear {
            background: #dc3545;
        }
        
        .preview-section {
            display: none;
            margin-bottom: 30px;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            margin: 20px auto;
            display: block;
        }
        
        .results-section {
            display: none;
            margin-top: 30px;
        }
        
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .result-card.not-masterplan {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .result-title {
            font-size: 2em;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .confidence-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1.3em;
            margin-top: 10px;
        }
        
        .details-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .probability-bars {
            margin-top: 20px;
        }
        
        .bar-container {
            margin: 15px 0;
        }
        
        .bar-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .bar {
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .bar-fill {
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .bar-fill.masterplan {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        .bar-fill.not-masterplan {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        }
        
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
        }
        
        .info-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .info-box ul {
            margin-left: 20px;
            color: #333;
        }
        
        .info-box li {
            margin: 5px 0;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ Masterplan Detector</h1>
        <p class="subtitle">Upload any image to detect if it's an architectural masterplan</p>
        
        <div class="upload-section" id="uploadSection">
            <div class="upload-icon">📤</div>
            <h2>Upload Image</h2>
            <p>Drop image here or click to upload</p>
            <p style="color: #666; font-size: 0.9em; margin-top: 10px;">Supports: JPG, PNG, PDF</p>
            <input type="file" id="fileInput" accept="image/*,.pdf">
        </div>
        
        <div class="preview-section" id="previewSection">
            <img id="previewImage" class="preview-image" alt="Preview">
            <div style="text-align: center;">
                <button class="btn" id="analyzeBtn">🔍 Analyze Image</button>
                <button class="btn btn-clear" id="clearBtn">🗑️ Clear</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px;">Analyzing image...</p>
        </div>
        
        <div class="results-section" id="resultsSection">
            <div class="result-card" id="resultCard">
                <div class="result-title" id="resultTitle"></div>
                <div class="confidence-badge" id="confidenceBadge"></div>
            </div>
            
            <div class="details-section">
                <h3>📊 Detailed Scores:</h3>
                <div class="probability-bars">
                    <div class="bar-container">
                        <div class="bar-label">
                            <span>Masterplan Probability:</span>
                            <span id="masterplanProb">0%</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill masterplan" id="masterplanBar" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="bar-container">
                        <div class="bar-label">
                            <span>Not Masterplan Probability:</span>
                            <span id="notMasterplanProb">0%</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill not-masterplan" id="notMasterplanBar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="info-box">
                <h3>💡 What is a Masterplan?</h3>
                <p>In architecture, a master plan is a comprehensive, long-term strategic document that provides a framework for guiding the future growth and development of a large area.</p>
                <br>
                <strong>Examples of Masterplans:</strong>
                <ul>
                    <li>Residential house floor plans</li>
                    <li>Office building layouts</li>
                    <li>Campus development plans</li>
                    <li>Urban planning schematics</li>
                </ul>
                <br>
                <strong>NOT Masterplans:</strong>
                <ul>
                    <li>Regular photographs</li>
                    <li>3D renderings</li>
                    <li>Street maps</li>
                    <li>Landscape images</li>
                </ul>
            </div>
        </div>
    </div>
    
    <script>
        const uploadSection = document.getElementById('uploadSection');
        const fileInput = document.getElementById('fileInput');
        const previewSection = document.getElementById('previewSection');
        const previewImage = document.getElementById('previewImage');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const clearBtn = document.getElementById('clearBtn');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');
        
        // Click to upload
        uploadSection.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('drag-over');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('drag-over');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleFile(file);
        });
        
        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                uploadSection.style.display = 'none';
                previewSection.style.display = 'block';
                resultsSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        // Analyze button
        analyzeBtn.addEventListener('click', async () => {
            loading.style.display = 'block';
            resultsSection.style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                alert('Error analyzing image: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        });
        
        // Clear button
        clearBtn.addEventListener('click', () => {
            fileInput.value = '';
            uploadSection.style.display = 'block';
            previewSection.style.display = 'none';
            resultsSection.style.display = 'none';
        });
        
        function displayResults(result) {
            const resultCard = document.getElementById('resultCard');
            const resultTitle = document.getElementById('resultTitle');
            const confidenceBadge = document.getElementById('confidenceBadge');
            
            if (result.is_masterplan) {
                resultCard.classList.remove('not-masterplan');
                resultTitle.textContent = '✅ YES, THIS IS A MASTERPLAN';
            } else {
                resultCard.classList.add('not-masterplan');
                resultTitle.textContent = '❌ NO, THIS IS NOT A MASTERPLAN';
            }
            
            confidenceBadge.textContent = `Confidence: ${result.confidence.toFixed(1)}%`;
            
            // Update probability bars
            document.getElementById('masterplanProb').textContent = 
                result.masterplan_probability.toFixed(2) + '%';
            document.getElementById('notMasterplanProb').textContent = 
                result.not_masterplan_probability.toFixed(2) + '%';
            
            document.getElementById('masterplanBar').style.width = 
                result.masterplan_probability + '%';
            document.getElementById('notMasterplanBar').style.width = 
                result.not_masterplan_probability + '%';
            
            resultsSection.style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    try:
        # Read and process image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Make prediction
        result = predictor.predict(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if predictor is None:
        print("⚠️ Warning: Model not loaded. Please train the model first using train.py")
    else:
        print("🚀 Starting Masterplan Detector Web App...")
        print("📍 Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
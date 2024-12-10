import os
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from transformers import BlipForConditionalGeneration, AutoTokenizer
from flask import send_from_directory
from werkzeug.utils import secure_filename  # Add this import

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add this route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Assuming Flask is being used in the backend:

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', model_error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', model_error="No selected file")
    
    # Save the image to the 'static/uploads/' folder
    filename = secure_filename(file.filename)  # Use secure_filename here
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Generate the report
    report = generate_report(filepath)  # This is your report generation logic
    return render_template('result.html', report=report, image_path=filepath)


# Load model and tokenizer with error handling
model = None
model_error = None
try:
    checkpoint_path = r"D:\CV Challenge\model.pth"
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    model_error = f"Model loading failed: {str(e)}"

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_report(image_path):
    """Generate a report from an image."""
    if not model:
        return "Error: Model not loaded. Please check the configuration."

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=image,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    report = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return report

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "No file uploaded.", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file.", 400

        if file:
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Generate report
            report = generate_report(filepath)

            return render_template('result.html', image_path=filepath, report=report, model_error=model_error)

    return render_template('index.html', model_error=model_error)


if __name__ == '__main__':
    app.run(debug=True)

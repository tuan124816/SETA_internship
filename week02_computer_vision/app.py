from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from detection import process_image  
import time

# Initialize Flask app and folders

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed image extension.
    """

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_folder(folder, max_age_seconds=300):
    """
    Delete files older than max_age_seconds from the specified folder.

    Args:
        folder (str): Path to the folder to clean up.
        max_age_seconds (int): Age threshold in seconds for deleting files.
    """
    now = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            if now - os.path.getmtime(filepath) > max_age_seconds:
                os.remove(filepath)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Flask route for homepage.

    GET: Render upload form.
    POST: Validate and save uploaded image, process it, and render results.

    Cleans up old images from uploads and processed folders.
    """

    cleanup_folder(app.config['UPLOAD_FOLDER'])
    cleanup_folder(PROCESSED_FOLDER)

    error_message = None
    result_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_path = process_image(filepath)
        else:
            error_message = "Invalid file type. Please upload a PNG, JPG, or JPEG image."

    return render_template('index.html', result_image=result_path, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)

import os
import shutil
from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('best_coco.pt')        # use best_coco.pt

def process_image(image_path):
    """
    Process an image by running object detection and saving the result with bounding boxes.

    Steps:
    - Resize the image if it exceeds max dimension (1024).
    - Run the YOLO model on the resized image.
    - Draw bounding boxes on detected objects.
    - Save the resulting image with '_result.jpg' suffix in the static folder.

    Args:
        image_path (str): Path to the uploaded image.

    Returns:
        str: Relative path to the processed image to be used in the frontend.
    """

    # Open and resize the image to limit max dimension for faster processing
    img = Image.open(image_path)
    max_dim = 1024
    img.thumbnail((max_dim, max_dim))
    resized_path = image_path  # Overwrite original
    img.save(resized_path)

    # Run YOLO detection
    results = model(resized_path)

    # Render result image boxes and labels
    result_img = results[0].plot()

    # Create static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)

    # Save result image with _result.jpg suffix to avoid overwriting
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    result_filename = f"{base_filename}_result.jpg"
    result_path = os.path.join(static_dir, result_filename)

    # Save the image with detections
    cv2.imwrite(result_path, result_img)

    return f'static/{result_filename}'

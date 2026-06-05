import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image as PILImage

from image_processing import find_folder_containing_subfolder, evaluate_image_formats, open_and_get_image_info

app = Flask(__name__)

folder_path = "images"
os.makedirs(folder_path, exist_ok=True)
TARGET_FOLDER = find_folder_containing_subfolder(folder_path)

all_files = os.listdir(folder_path)
image_filenames = [os.path.join(folder_path, filename) for filename in all_files if filename.lower(
).endswith(('.png', '.avif', '.heif', '.webp', '.jpg', '.jpeg'))]

def is_valid_image(file_name):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    return file_name.lower().endswith(valid_extensions)

@app.route('/upload_results')
def display_images():
    image_info = []
    for filename in image_filenames:
        if os.path.exists(filename):
            image_data = open_and_get_image_info(filename)
            image_info.append(image_data)
    return render_template('display_images.html', image_info=image_info)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        image_content = uploaded_file.read()
        file_name = uploaded_file.filename

        if not is_valid_image(file_name):
            return "Selected file is not a valid image format."

        temp_file_path = os.path.join(folder_path, "temp_uploaded_image")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(image_content)

        # Retain color channel properties instead of forcing flat L-mode structures globally
        image = PILImage.open(temp_file_path)
        image_name = os.path.splitext(os.path.basename(file_name))[0]
        
        if image is None:
            return "Error loading the image target."

        raw_png_array = np.array(image)

        # Evaluate performance across codecs using uniform settings
        mse_results, psnr_results, ssim_results, encoding_times, decoding_times, compression_ratios = evaluate_image_formats(
            raw_png_array, image_name, temp_file_path)

        # Cleanup internal temporary tracking file safely
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return render_template(
            'upload.html', 
            mse_results=mse_results, 
            psnr_results=psnr_results, 
            ssim_results=ssim_results, 
            encoding_times=encoding_times, 
            decoding_times=decoding_times, 
            compression_ratios=compression_ratios
        )

    return "No file uploaded."

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error('Internal Server Error: %s', e)
    return 'Internal Server Error', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
import os
import tkinter as tk
import numpy as np
import io as ioo
from tkinter import filedialog
from flask import Flask, render_template, request
from PIL import Image as PILImage

# Import your functions from the image_processing module
from image_processing import find_folder_containing_subfolder, evaluate_image_formats, open_and_get_image_info


app = Flask(__name__)

# Folder containing the image files
folder_path = "images"
TARGET_FOLDER = find_folder_containing_subfolder(folder_path)

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Filter only image files
image_filenames = [os.path.join(folder_path, filename) for filename in all_files if filename.lower(
).endswith(('.png', '.avif', '.heic', '.webp'))]

def is_valid_image(file_name):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    return file_name.lower().endswith(valid_extensions)



@app.route('/upload_results')
def display_images():
    image_info = []

    for filename in image_filenames:
        image_data = open_and_get_image_info(filename)
        image_info.append(image_data)
    print(filename)
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
            return "Selected file is not a valid image."

        temp_file_path = "images/temp_uploaded_image"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(image_content)

        image = PILImage.open(temp_file_path)
        image_name = os.path.basename(file_name)
        image_name = os.path.splitext(os.path.basename(file_name))[0]
        if image is None:
            return "Error loading the image."

        if image.mode != "L":
            image = image.convert("L")

        raw_png_array = np.array(image)

        # Process the results or return them in a response
        mse_results, psnr_results, ssim_results = evaluate_image_formats(
            raw_png_array, image_name, temp_file_path)

        # Pass the evaluation results to the template
        return render_template('upload.html', mse_results=mse_results, psnr_results=psnr_results, ssim_results=ssim_results)

    return "No file uploaded."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

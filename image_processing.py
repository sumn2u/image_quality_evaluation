import subprocess
import numpy as np
import tempfile
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from PIL import Image as PILImage
import io as ioo
from skimage import transform, color
import os
from pathlib import Path
# import avif
from IPython.display import display
import imageio
import pillow_avif
import pillow_heif
import time
import base64


def open_and_get_image_info(filename):
    _, ext = os.path.splitext(filename)
    start_time = time.time()  # Record start time
    print(filename)
    if ext.lower() == '.heif':
        heif_file = pillow_heif.open_heif(filename, convert_hdr_to_8bit=False)
        np_array = np.asarray(heif_file)
        img = PILImage.frombytes(
            heif_file.mode, heif_file.size, np_array, "raw", heif_file.mode, heif_file.stride)
        print("Image data length:", len(heif_file.data))
        print("Image data stride:", heif_file.stride)
    else:
        img = PILImage.open(filename)
        # img.thumbnail((200, 200))  # Resize the image for display

    # Convert the image to base64-encoded data
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Calculate loading time
    loading_time = time.time() - start_time
    print("Loading Time:", loading_time, "seconds\n", filename)
    image_data = {
        "filename": filename,
        "loading_time": loading_time,
        "image_base64": image_base64
    }

    return image_data


def save_image_to_folder(original_image, image_bytes, image_format, filename):
    """Saves the image to a specific folder with the given filename and format."""
    # Determine if the original image is grayscale
    is_grayscale = len(original_image.shape) == 2 or (
        len(original_image.shape) == 3 and original_image.shape[2] == 1)

    IMAGES_FOLDER = "images"
    # Save the original image using imageio
    original_filepath = os.path.join(IMAGES_FOLDER, f"{filename}_original.png")
    imageio.imwrite(original_filepath, original_image)

    # Save the converted image
    filepath = os.path.join(IMAGES_FOLDER, f"{filename}.{image_format}")

    if is_grayscale:
      # If the original image is grayscale and the desired format is PNG,
      # convert the grayscale image to "L" mode and save
      grayscale_image = PILImage.fromarray(original_image)
      grayscale_image.save(filepath, format=image_format)
    else:
        # For other formats or if the image is not grayscale, save as is
      with open(filepath, "wb") as file:
        file.write(image_bytes)


def convert_to_avif(image, image_name):
    """Converts an image to AVIF format."""

    # Convert the image to a numpy array
    image_np = np.asarray(image)

    # Handle case when the image is in RGB format
    if image_np.shape[-1] == 3:
        image_np = np.concatenate(
            (image_np, 255 * np.ones(image_np.shape[:-1] + (1,), dtype=np.uint8)), axis=-1)

    # Create a PIL Image from the numpy array
    pil_image = PILImage.fromarray(image_np)

    # Save the image to a temporary file in AVIF format
    with ioo.BytesIO() as temp_file:
        pil_image.save(temp_file, format="AVIF")
        avif_bytes = temp_file.getvalue()
        save_image_to_folder(image, avif_bytes, 'avif',  image_name)
    return avif_bytes


def convert_to_png(image, image_name):
    """Converts the image to PNG format."""
    image_png = PILImage.fromarray(np.uint8(image))
    png_bytes = ioo.BytesIO()
    image_png.save(png_bytes, format="png")
    save_image_to_folder(image, png_bytes.getvalue(), 'png',  image_name)
    return png_bytes


def convert_to_webp(image, image_name):
    """Converts the image to WebP format."""
    image_webp = PILImage.fromarray(np.uint8(image))
    webp_bytes = ioo.BytesIO()
    image_webp.save(webp_bytes, format="webp")
    save_image_to_folder(image, webp_bytes.getvalue(), 'webp',  image_name)
    return webp_bytes


def find_folder_containing_subfolder(target_subfolder):
    current_path = os.getcwd()  # Get the current working directory

    while current_path != '/':  # Loop until reaching the root directory
        if target_subfolder in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)  # Move up one directory

    return None  # Return None if not found




# Function to convert an image to HEIF format using imageio
def convert_to_heif(image, image_name, image_path):
    pillow_heif.register_heif_opener()
    IMAGES_FOLDER = "images"
    start_time = time.time()

    img = PILImage.open(image_path)
    img.save(os.path.join(IMAGES_FOLDER,
             f"{image_name}.heif"), quality=100, save_all=True)
    encoding_time = time.time() - start_time

    heif_path = os.path.join(IMAGES_FOLDER, f"{image_name}.heif")

    start_time = time.time()
    heif_file = pillow_heif.open_heif(heif_path, convert_hdr_to_8bit=False)
    np_array = np.asarray(heif_file)
    img = PILImage.frombytes(heif_file.mode, heif_file.size,
                             np_array, "raw", heif_file.mode, heif_file.stride)
    heif_bytes = img.tobytes()

    decoding_time = time.time() - start_time
    return heif_bytes, img, encoding_time, decoding_time

# Function to calculate the compression ratio
def calculate_compression_ratio(original_image,  compressed_image_bytes):
    original_image_size = len(original_image.tobytes()) 
    compressed_image_size = len(compressed_image_bytes)
    compression_ratio = original_image_size / compressed_image_size
    return compression_ratio

def convert_to_numpy_array(image):
    """Converts a PIL image to a NumPy array."""
    return np.array(image)


def load_image(image_url):
    """Loads the image from the specified URL."""
    image = ioo.imread(image_url)
    if len(image.shape) == 2:
        # Convert grayscale image to color image (3 channels)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image


def evaluate_image_quality(image, dist_metric, data_range=255):
    """Evaluates the quality of an image using a distortion metric."""
    mse = mean_squared_error(image, dist_metric)
    psnr = peak_signal_noise_ratio(image, dist_metric, data_range=data_range)
    ssim = structural_similarity(
        image, dist_metric, multichannel=True, data_range=data_range)
    return mse, psnr, ssim


def resize_image(image, target_shape):
    """Resizes the image to the target shape using bilinear interpolation."""
    return transform.resize(image, target_shape, mode='reflect', anti_aliasing=True)

def evaluate_image_formats(image, image_name, image_path):
    formats = ["JPEG", "WebP",  "HEIF", "AVIF"]

    mse_results = {format: [] for format in formats}
    psnr_results = {format: [] for format in formats}
    ssim_results = {format: [] for format in formats}
    encoding_times = {format: [] for format in formats}
    decoding_times = {format: [] for format in formats}
    compression_ratios = {format: [] for format in formats}

    for format in formats:
        if format == "JPEG":
            start_time = time.time()
            image_jpeg = PILImage.fromarray(np.uint8(image))
            image_jpeg_bytes = ioo.BytesIO()
            image_jpeg.save(image_jpeg_bytes, format="jpeg")
            encoding_time = time.time() - start_time

            start_time = time.time()
            image_format = PILImage.open(image_jpeg_bytes)
            image_format = image_format.convert("RGB")  # Convert to RGB mode
            decoding_time = time.time() - start_time
            compression_ratio = calculate_compression_ratio(
                image, image_jpeg_bytes.getvalue())

        elif format == "WebP":
            start_time = time.time()
            image_webp = convert_to_webp(image, image_name)
            encoding_time = time.time() - start_time

            start_time = time.time()
            image_format = PILImage.open(image_webp)
            decoding_time = time.time() - start_time
            compression_ratio = calculate_compression_ratio(
                image, image_webp.getvalue())

        elif format == "AVIF":
            start_time = time.time()
            avif_bytes = convert_to_avif(image, image_name)
            encoding_time = time.time() - start_time

            start_time = time.time()
            image_format = PILImage.open(ioo.BytesIO(avif_bytes))
            decoding_time = time.time() - start_time
            compression_ratio = calculate_compression_ratio(
                image, avif_bytes)

        elif format == "HEIF":
            heif_bytes, image_format, encoding_time, decoding_time = convert_to_heif(
                image, image_name, image_path)
            compression_ratio = calculate_compression_ratio(
                image, heif_bytes)

        image_format_array = convert_to_numpy_array(image_format)
        if image_format_array.shape[-1] == 3:
            image_format_array_gray = color.rgb2gray(image_format_array)
        else:
            image_format_array_gray = image_format_array

        target_shape = image_format_array_gray.shape
        image_resized = transform.resize(
            image, target_shape, mode='reflect', anti_aliasing=True)

        mse, psnr, ssim = evaluate_image_quality(
            image_resized, image_format_array_gray)
        mse_results[format].append(mse)
        psnr_results[format].append(psnr)
        ssim_results[format].append(ssim)
        encoding_times[format].append(encoding_time)
        decoding_times[format].append(decoding_time)
        compression_ratios[format].append(compression_ratio)
    return mse_results, psnr_results, ssim_results, encoding_times, decoding_times, compression_ratios


def load_image_from_file(file_path):
    image = ioo.imread(file_path)
    return image, os.path.basename(file_path)



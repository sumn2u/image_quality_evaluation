import subprocess
import numpy as np
import tempfile
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from PIL import Image as PILImage
import io as ioo
from skimage import color
import os
import time
import base64
import imageio
import pillow_avif
import pillow_heif

def open_and_get_image_info(filename):
    _, ext = os.path.splitext(filename)
    start_time = time.time()  
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

    with open(filename, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

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
    IMAGES_FOLDER = "images"
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    
    # Save the original image if it doesn't exist yet
    original_filepath = os.path.join(IMAGES_FOLDER, f"{filename}_original.png")
    if not os.path.exists(original_filepath):
        imageio.imwrite(original_filepath, original_image)

    filepath = os.path.join(IMAGES_FOLDER, f"{filename}.{image_format}")
    with open(filepath, "wb") as file:
        file.write(image_bytes)

def convert_to_avif(image, image_name):
    """Converts an image to AVIF format."""
    pil_image = PILImage.fromarray(np.uint8(image))
    with ioo.BytesIO() as temp_file:
        pil_image.save(temp_file, format="AVIF", quality=80)
        avif_bytes = temp_file.getvalue()
        save_image_to_folder(image, avif_bytes, 'avif',  image_name)
    return avif_bytes

def convert_to_png(image, image_name):
    """Converts the image to PNG format."""
    image_png = PILImage.fromarray(np.uint8(image))
    with ioo.BytesIO() as png_bytes:
        image_png.save(png_bytes, format="png")
        raw_bytes = png_bytes.getvalue()
        save_image_to_folder(image, raw_bytes, 'png',  image_name)
    return raw_bytes

def convert_to_webp(image, image_name):
    """Converts the image to WebP format."""
    image_webp = PILImage.fromarray(np.uint8(image))
    with ioo.BytesIO() as webp_bytes:
        image_webp.save(webp_bytes, format="webp", quality=80)
        raw_bytes = webp_bytes.getvalue()
        save_image_to_folder(image, raw_bytes, 'webp',  image_name)
    return raw_bytes  # Fixed: returns bytes instead of wrapper IO stream

def find_folder_containing_subfolder(target_subfolder):
    current_path = os.getcwd()
    while current_path != '/':
        if target_subfolder in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)
    return None

def convert_to_heif(image, image_name, image_path):
    """Converts an image to HEIF format and handles clean timing profiles."""
    pillow_heif.register_heif_opener()
    IMAGES_FOLDER = "images"
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    heif_path = os.path.join(IMAGES_FOLDER, f"{image_name}.heif")

    # Benchmarking Encode Process
    start_time = time.time()
    img = PILImage.open(image_path)
    img.save(heif_path, format="HEIF", quality=80)
    encoding_time = time.time() - start_time

    # Benchmarking Decode Process
    start_time = time.time()
    heif_file = pillow_heif.open_heif(heif_path, convert_hdr_to_8bit=True)
    np_array = np.asarray(heif_file)
    decoded_img = PILImage.frombytes(heif_file.mode, heif_file.size, np_array, "raw", heif_file.mode, heif_file.stride)
    decoding_time = time.time() - start_time

    with open(heif_path, "rb") as f:
        heif_bytes = f.read()

    return heif_bytes, decoded_img, encoding_time, decoding_time

def calculate_compression_ratio(original_image, compressed_image_bytes):
    original_image_size = original_image.nbytes
    compressed_image_size = len(compressed_image_bytes)
    return original_image_size / compressed_image_size

def convert_to_numpy_array(image):
    return np.array(image)

def evaluate_image_quality(image, dist_metric, data_range=255):
    """Evaluates the quality of two images sharing matching dimensions on a 0-255 domain."""
    mse = mean_squared_error(image, dist_metric)
    psnr = peak_signal_noise_ratio(image, dist_metric, data_range=data_range)
    
    # Check if images are grayscale or color to apply channel rules properly
    if len(image.shape) == 3 and image.shape[-1] in [3, 4]:
        ssim = structural_similarity(image, dist_metric, channel_axis=-1, data_range=data_range)
    else:
        ssim = structural_similarity(image, dist_metric, data_range=data_range)
        
    return mse, psnr, ssim

def evaluate_image_formats(image, image_name, image_path):
    formats = ["JPEG", "WebP", "HEIF", "AVIF"]

    mse_results = {f: [] for f in formats}
    psnr_results = {f: [] for f in formats}
    ssim_results = {f: [] for f in formats}
    encoding_times = {f: [] for f in formats}
    decoding_times = {f: [] for f in formats}
    compression_ratios = {f: [] for f in formats}

    # Track structural reference details
    # Ensure source image remains standard 0-255 ndarray throughout processing loop
    if image.dtype != np.uint8:
        image = np.uint8(image)

    for fmt in formats:
        if fmt == "JPEG":
            start_time = time.time()
            image_jpeg = PILImage.fromarray(image)
            image_jpeg_bytes = ioo.BytesIO()
            image_jpeg.save(image_jpeg_bytes, format="jpeg", quality=80)
            jpeg_data = image_jpeg_bytes.getvalue()
            encoding_time = time.time() - start_time

            start_time = time.time()
            image_format = PILImage.open(ioo.BytesIO(jpeg_data))
            image_format.load()  # Force execution processing loops inside timing brackets
            decoding_time = time.time() - start_time
            
            compression_ratio = calculate_compression_ratio(image, jpeg_data)
            save_image_to_folder(image, jpeg_data, 'jpg', image_name)

        elif fmt == "WebP":
            start_time = time.time()
            webp_data = convert_to_webp(image, image_name)
            encoding_time = time.time() - start_time

            start_time = time.time()
            image_format = PILImage.open(ioo.BytesIO(webp_data))
            image_format.load()
            decoding_time = time.time() - start_time
            
            compression_ratio = calculate_compression_ratio(image, webp_data)

        elif fmt == "AVIF":
            start_time = time.time()
            avif_bytes = convert_to_avif(image, image_name)
            encoding_time = time.time() - start_time

            start_time = time.time()
            image_format = PILImage.open(ioo.BytesIO(avif_bytes))
            image_format.load()
            decoding_time = time.time() - start_time
            
            compression_ratio = calculate_compression_ratio(image, avif_bytes)

        elif fmt == "HEIF":
            heif_bytes, image_format, encoding_time, decoding_time = convert_to_heif(
                image, image_name, image_path)
            compression_ratio = calculate_compression_ratio(image, heif_bytes)

        # Build clean output ndarrays
        image_format_array = convert_to_numpy_array(image_format)
        
        # Ensure channel matching without normalized down-scaling modifications
        if len(image.shape) == 2 and len(image_format_array.shape) == 3:
            image_format_array = color.rgb2gray(image_format_array)
            image_format_array = np.uint8(image_format_array * 255) # Re-normalize up to scale domain
        elif len(image.shape) == 3 and len(image_format_array.shape) == 2:
            image_format_array = np.repeat(image_format_array[:, :, np.newaxis], 3, axis=2)

        # Run direct metric evaluation using identical resolutions
        mse, psnr, ssim = evaluate_image_quality(image, image_format_array, data_range=255)
        
        mse_results[fmt].append(round(mse, 5))
        psnr_results[fmt].append(round(psnr, 2))
        ssim_results[fmt].append(round(ssim, 4))
        encoding_times[fmt].append(round(encoding_time, 4))
        decoding_times[fmt].append(round(decoding_time, 4))
        compression_ratios[fmt].append(round(compression_ratio, 2))
        
    return mse_results, psnr_results, ssim_results, encoding_times, decoding_times, compression_ratios
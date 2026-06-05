import os
import time
import io as ioo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import pillow_avif
import pillow_heif

# Register modern image codecs with Pillow
pillow_heif.register_heif_opener()

def run_benchmark(img_array, image_path):
    formats = ["JPEG", "WebP", "AVIF", "HEIF"]
    
    encoding_times = []
    decoding_times = []
    compression_ratios = []
    
    # Ensure source image remains standard 0-255 uint8 ndarray
    if img_array.dtype != np.uint8:
        img_array = np.uint8(img_array)
        
    original_size = img_array.nbytes
    pil_img = PILImage.fromarray(img_array)

    for fmt in formats:
        # --- Benchmark Encoding ---
        start_encode = time.time()
        buf = ioo.BytesIO()
        
        if fmt == "JPEG":
            pil_img.save(buf, format="JPEG", quality=80)
            compressed_bytes = buf.getvalue()
        elif fmt == "WebP":
            pil_img.save(buf, format="WebP", quality=80)
            compressed_bytes = buf.getvalue()
        elif fmt == "AVIF":
            pil_img.save(buf, format="AVIF", quality=80)
            compressed_bytes = buf.getvalue()
        elif fmt == "HEIF":
            # HEIF requires tracking file IO context closely for realistic timing
            temp_heif = "images/temp_cam.heif"
            os.makedirs("images", exist_ok=True)
            pil_img.save(temp_heif, format="HEIF", quality=80)
            with open(temp_heif, "rb") as f:
                compressed_bytes = f.read()
            if os.path.exists(temp_heif):
                os.remove(temp_heif)
                
        enc_time = time.time() - start_encode
        encoding_times.append(enc_time)

        # --- Benchmark Decoding ---
        start_decode = time.time()
        if fmt == "HEIF":
            heif_file = pillow_heif.open_heif(ioo.BytesIO(compressed_bytes), convert_hdr_to_8bit=True)
            recon_array = np.asarray(heif_file)
        else:
            recon_img = PILImage.open(ioo.BytesIO(compressed_bytes))
            recon_img.load()  # Force internal decompression operations inside bracket
            recon_array = np.array(recon_img)
            
        dec_time = time.time() - start_decode
        decoding_times.append(dec_time)

        # --- Compression Ratio ---
        cr = original_size / len(compressed_bytes)
        compression_ratios.append(cr)
        
    return formats, encoding_times, decoding_times, compression_ratios

def main():
    # Attempt to load standard Cameraman image
    try:
        from skimage import data
        img = data.camera()
    except ImportError:
        img = np.uint8(np.indices((512, 512))[0] % 255)

    formats, enc_times, dec_times, comp_ratios = run_benchmark(img, "images/camera.png")

    # --- Plot Layout Generation ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"] 

    # Subplot 1: Encoding Time
    axes[0].bar(formats, enc_times, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("Encoding Time (Lower = Faster)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Time (seconds)", fontsize=10)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Decoding Time
    axes[1].bar(formats, dec_times, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title("Decoding Time (Lower = Faster)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Time (seconds)", fontsize=10)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Subplot 3: Compression Ratio
    axes[2].bar(formats, comp_ratios, color=colors, edgecolor='black', alpha=0.85)
    axes[2].set_title("Compression Ratio (Higher = Better)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Ratio ($Size_{orig} / Size_{comp}$)", fontsize=10)
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)

    # plt.suptitle("Codec Computational Overhead & Efficiency Analysis (Cameraman)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save chart asset out cleanly
    output_path = "images/codec_benchmarks.png"
    os.makedirs("images", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Benchmark plotting complete! Visualization chart asset saved to: '{output_path}'")
    plt.show()

if __name__ == "__main__":
    main()
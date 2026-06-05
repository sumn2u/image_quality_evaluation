import os
import time
import io as ioo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from skimage.metrics import structural_similarity
import pillow_avif
import pillow_heif

# Register modern image codecs with Pillow
pillow_heif.register_heif_opener()

def calculate_cr(original_array, compressed_bytes):
    """Calculates compression ratio: original size in bytes / compressed size in bytes."""
    original_size = original_array.nbytes
    compressed_size = len(compressed_bytes)
    return original_size / compressed_size

def get_metrics_for_quality(img_array, fmt, quality):
    """Compresses an image at a specific quality and returns its SSIM and CR."""
    pil_img = PILImage.fromarray(img_array)
    buf = ioo.BytesIO()
    
    # Compress/Save to memory buffer
    if fmt == "JPEG":
        pil_img.save(buf, format="JPEG", quality=quality)
    elif fmt == "WebP":
        pil_img.save(buf, format="WebP", quality=quality)
    elif fmt == "AVIF":
        pil_img.save(buf, format="AVIF", quality=quality)
    elif fmt == "HEIF":
        pil_img.save(buf, format="HEIF", quality=quality)
        
    compressed_bytes = buf.getvalue()
    cr = calculate_cr(img_array, compressed_bytes)
    
    # Decompress to compute SSIM
    buf.seek(0)
    recon_img = PILImage.open(buf)
    recon_array = np.array(recon_img)
    
    # Match grayscale dimensions if conversions layered extra channels
    if len(img_array.shape) == 2 and len(recon_array.shape) == 3:
        recon_array = recon_array[:, :, 0] # Extract single channel
        
    ssim = structural_similarity(img_array, recon_array, data_range=255)
    return cr, ssim

def main():
    # Create mock Cameraman image (Grayscale 512x512) if skimage package isn't directly pulled
    try:
        from skimage import data
        img = data.camera()
    except ImportError:
        print("skimage not found, generating a synthetic reference grid for testing...")
        img = np.uint8(np.indices((512, 512))[0] % 255)

    formats = ["JPEG", "WebP", "AVIF", "HEIF"]
    
    # Define a range of quality parameters to sweep through
    quality_range = [10, 20, 40, 60, 75, 85, 95]
    
    # Style configuration for scientific paper plots
    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    
    markers = {"JPEG": "o", "WebP": "s", "AVIF": "^", "HEIF": "D"}
    colors = {"JPEG": "#e74c3c", "WebP": "#3498db", "AVIF": "#2ecc71", "HEIF": "#9b59b6"}

    for fmt in formats:
        cr_values = []
        ssim_values = []
        
        for q in quality_range:
            try:
                cr, ssim = get_metrics_for_quality(img, fmt, q)
                cr_values.append(cr)
                ssim_values.append(ssim)
            except Exception as e:
                print(f"Skipping quality setting {q} for format {fmt} due to library limitation.")
                continue
        
        # Sort values by Compression Ratio so line renders continuously without crossing over
        sorted_indices = np.argsort(cr_values)
        sorted_cr = np.array(cr_values)[sorted_indices]
        sorted_ssim = np.array(ssim_values)[sorted_indices]
        
        # Plot rate-distortion operational curve
        plt.plot(sorted_cr, sorted_ssim, label=fmt, marker=markers[fmt], 
                 color=colors[fmt], linewidth=2, markersize=7)

    # Label details 
    # plt.title("Rate-Distortion Evaluation: SSIM vs. Compression Ratio (Cameraman)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Compression Ratio (Higher = Smaller File Size)", fontsize=11)
    plt.ylabel("Structural Similarity Index (SSIM)", fontsize=11)
    plt.ylim(0.80, 1.01) # Zoom into high fidelity window to capture precise divergence
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    
    # Save chart asset out cleanly
    output_plot = "images/ssim_vs_cr_curve.png"
    os.makedirs("images", exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlotting complete! Rate-Distortion curve saved to: '{output_plot}'")
    plt.show()

if __name__ == "__main__":
    main()
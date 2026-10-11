# src/utils/image_utils.py

import os
import numpy as np
from PIL import Image, ImageDraw

def load_image(image_input, target_size=None):
    """
    Loads an image from a path or file-like object and converts to RGB.
    Optionally resizes to target_size (width, height).
    """
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")
        
    if target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return image

def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, alpha: float = 0.5, color=(34, 139, 34)) -> Image.Image:
    """
    Overlays a binary segmentation mask (1 = canopy, 0 = background) onto the PIL image.
    """
    orig_np = np.array(image.convert("RGB"))
    
    # Resize mask to match image if dimensions differ
    if mask.shape != orig_np.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8)).resize((orig_np.shape[1], orig_np.shape[0]), resample=Image.Resampling.NEAREST)
        mask = np.array(mask_pil)
        
    overlay = orig_np.copy()
    overlay[mask == 1] = color
    
    blended = (alpha * overlay + (1.0 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)

def create_synthetic_satellite_image(output_path: str, size: tuple = (512, 512)) -> str:
    """
    Creates a realistic synthetic urban satellite tile with concrete roads, 
    building rooftops, and distinct green tree canopy clusters for end-to-end testing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = size
    
    # Base urban ground (light asphalt / dirt)
    img = Image.new("RGB", size, color=(160, 160, 155))
    draw = ImageDraw.Draw(img)
    
    # Asphalt roads
    draw.rectangle([0, int(height * 0.45), width, int(height * 0.55)], fill=(70, 70, 75))
    draw.rectangle([int(width * 0.45), 0, int(width * 0.55), height], fill=(70, 70, 75))
    
    # Building rooftops (industrial grey & terra cotta)
    draw.rectangle([20, 20, int(width * 0.38), int(height * 0.38)], fill=(120, 115, 110))
    draw.rectangle([int(width * 0.60), 30, int(width * 0.90), int(height * 0.35)], fill=(185, 100, 80))
    draw.rectangle([int(width * 0.58), int(height * 0.60), int(width * 0.92), int(height * 0.90)], fill=(95, 105, 115))
    
    # Tree canopy patches (shades of rich green)
    # Patch 1: Park in bottom left
    for offset_x in range(30, int(width * 0.38), 25):
        for offset_y in range(int(height * 0.60), int(height * 0.92), 25):
            radius = np.random.randint(14, 22)
            green_val = np.random.randint(100, 150)
            draw.ellipse(
                [offset_x - radius, offset_y - radius, offset_x + radius, offset_y + radius],
                fill=(34, green_val, 40)
            )
            
    # Street trees along roads
    for x in range(20, width - 20, 45):
        if not (int(width * 0.40) < x < int(width * 0.60)):
            draw.ellipse([x - 10, int(height * 0.42) - 10, x + 10, int(height * 0.42) + 10], fill=(40, 140, 45))
            draw.ellipse([x - 10, int(height * 0.58) - 10, x + 10, int(height * 0.58) + 10], fill=(30, 130, 35))

    img.save(output_path)
    return output_path

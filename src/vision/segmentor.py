# src/vision/segmentor.py

import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class CanopySegmenter:
    """
    SegFormer-based semantic segmentation model for aerial/satellite urban tree canopy assessment.
    Runs on CUDA if available, falling back smoothly to CPU.
    """
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CanopySegmenter model '{model_name}' on device: {self.device}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # In ADE20K, class index 4 corresponds to 'tree'
        self.tree_class_id = 4

    def segment(self, image_input):
        """
        Takes an image file path or PIL Image and returns:
          coverage_percentage: float (0.0 - 100.0)
          tree_mask: np.ndarray (H, W) uint8 binary mask (1 = canopy, 0 = non-canopy)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")

        orig_w, orig_h = image.size

        # Preprocess input image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        # Upsample logits back to input image spatial dimensions
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        predictions = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        tree_mask = (predictions == self.tree_class_id).astype(np.uint8)

        total_pixels = tree_mask.size
        tree_pixels = int(np.sum(tree_mask))
        coverage_percentage = round((tree_pixels / max(total_pixels, 1)) * 100.0, 2)

        return coverage_percentage, tree_mask

    def analyze_canopy(self, image_path):
        """Backwards compatibility alias for existing routes."""
        return self.segment(image_path)

if __name__ == "__main__":
    segmenter = CanopySegmenter()
    # Create test tile
    test_img = Image.new("RGB", (512, 512), color=(100, 150, 100))
    cov, mask = segmenter.segment(test_img)
    print(f"Test completed successfully: coverage = {cov}%")

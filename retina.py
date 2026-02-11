"""
Retina preprocessing utilities.

Scales images preserving aspect ratio and pads to fit a fixed-size square retina.
Used by both the Music OCR model (single retina) and the Symbol Similarity model
(dual retina).
"""

from PIL import Image
import numpy as np
import torch


def fit_to_retina(image: Image.Image, retina_size: int = 1024,
                  bg_color=(255, 255, 255)) -> tuple:
    """
    Scale image preserving aspect ratio, center-pad to retina_size x retina_size.

    Args:
        image: PIL Image to fit.
        retina_size: Target square dimension.
        bg_color: Background/padding color (RGB tuple).

    Returns:
        (retina_image, scale, offset_x, offset_y) — the fitted image and
        the transform parameters needed to map coordinates back and forth.
    """
    if image.mode == 'RGBA':
        # Composite onto white background
        bg = Image.new('RGB', image.size, bg_color)
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    w, h = image.size
    scale = min(retina_size / w, retina_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    retina = Image.new('RGB', (retina_size, retina_size), bg_color)
    offset_x = (retina_size - new_w) // 2
    offset_y = (retina_size - new_h) // 2
    retina.paste(resized, (offset_x, offset_y))

    return retina, scale, offset_x, offset_y


def map_bbox_to_retina(bbox: tuple, scale: float, offset_x: int, offset_y: int,
                       retina_size: int = 1024) -> tuple:
    """
    Map a bounding box from original image coordinates to normalized retina
    coordinates in [0, 1].
    """
    x1, y1, x2, y2 = bbox
    rx1 = (x1 * scale + offset_x) / retina_size
    ry1 = (y1 * scale + offset_y) / retina_size
    rx2 = (x2 * scale + offset_x) / retina_size
    ry2 = (y2 * scale + offset_y) / retina_size
    return (rx1, ry1, rx2, ry2)


def map_bbox_from_retina(bbox_norm: tuple, scale: float, offset_x: int,
                         offset_y: int, retina_size: int = 1024) -> tuple:
    """
    Map normalized retina coordinates back to original image coordinates.
    """
    rx1, ry1, rx2, ry2 = bbox_norm
    x1 = (rx1 * retina_size - offset_x) / scale
    y1 = (ry1 * retina_size - offset_y) / scale
    x2 = (rx2 * retina_size - offset_x) / scale
    y2 = (ry2 * retina_size - offset_y) / scale
    return (x1, y1, x2, y2)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to normalized torch tensor [C, H, W] in [0, 1]."""
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert torch tensor [C, H, W] back to PIL Image."""
    arr = tensor.permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

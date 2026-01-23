import numpy as np

def adjust_brightness(img: np.ndarray, value: int) -> np.ndarray:
    """
    Adjust image brightness.
    """
    img = img.astype(int) + value
    return np.clip(img, 0, 255).astype(np.uint8)


def normalize_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Normalize grayscale image to uint8.
    """
    img = img - img.min()
    img = img / img.max()
    return (img * 255).astype(np.uint8)
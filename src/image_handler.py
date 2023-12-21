from dataclasses import dataclass

from PIL.Image import Image


@dataclass
class ImageHandler:
    img: Image
    true_x: float
    true_y: float

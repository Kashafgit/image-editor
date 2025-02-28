import cv2
import numpy as np
from PIL import Image
from rembg import remove
from io import BytesIO

def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def apply_grayscale(image):
    gray = cv2.cvtColor(pil_to_cv(image),cv2.COLOR_BGR2GRAY)
    return cv_to_pil(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))



def apply_blur(image, intensity = 5):
    return cv_to_pil(cv2.GaussianBlur(pil_to_cv(image), (intensity, intensity),0))



def remove_bg(image):
    img_data = remove(image)
    return Image.open(BytesIO(img_data))


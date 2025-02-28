import cv2
import numpy as np
from PIL import Image
from rembg import remove
from io import BytesIO
# convert pil image to opencv formate
def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# convert openccv img to pil formate
def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# apply grayscale filter

def apply_grayscale(image):
    gray = cv2.cvtColor(pil_to_cv(image),cv2.COLOR_BGR2GRAY)
    return cv_to_pil(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

# apply blur filter

def apply_blur(image, intensity = 5):
    return cv_to_pil(cv2.GaussianBlur(pil_to_cv(image), (intensity, intensity),0))

# apply cartoon effect

def apply_cartoon(image):
    img = pil_to_cv(image)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask = edges)
    return cv_to_pil(cartoon)

def remove_bg(image):
    img_data = remove(image)
    return Image.open(BytesIO(img_data))


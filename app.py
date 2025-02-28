import streamlit as st
from PIL import Image
import cv2
import numpy as np
from rembg import remove
from io import BytesIO

# Custom CSS for styling
TAILWIND_STYLE = '''
<style>
.stButton > button {
    background-color: #4f46e5;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 20px;
}
.stFileUploader {
    text-align: center;
}
</style>
'''

st.set_page_config(page_title="AI Image Editor", layout="wide")
st.markdown(TAILWIND_STYLE, unsafe_allow_html=True)

st.title("🤖📸AI-Powered Image Editor🤳")

uploaded_img = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Convert PIL Image to OpenCV Format
def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Convert OpenCV Image to PIL Format
def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Resize Image to fit within a maximum size
def resize_image(image, max_size=500):
    img = np.array(image)
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(img)

# Crop Image to a square aspect ratio
def crop_image(image):
    img = np.array(image)
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return Image.fromarray(cropped_img)

# Apply Grayscale Filter
def apply_grayscale(image):
    gray = cv2.cvtColor(pil_to_cv(image), cv2.COLOR_BGR2GRAY)
    return cv_to_pil(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

# Apply Blur Filter
def apply_blur(image, intensity=5):
    return cv_to_pil(cv2.GaussianBlur(pil_to_cv(image), (intensity, intensity), 0))

# Remove Background
def remove_bg(image):
    img_data = remove(image)
    return Image.open(BytesIO(img_data))

if uploaded_img:
    image = Image.open(uploaded_img)
    st.write("Original Image Dimensions:", image.size)

    # Resize the image to fit within the visible area
    image = resize_image(image, max_size=500)  # Resize to a maximum of 500px
    st.write("Resized Image Dimensions:", image.size)

    # Display the original image with a fixed maximum width
    st.image(image, caption="Original Image", use_container_width=True)

    # Crop Image Option
    if st.checkbox("Crop Image to Square"):
        image = crop_image(image)
        st.write("Cropped Image Dimensions:", image.size)
        st.image(image, caption="Cropped Image", use_container_width=True)

    # Filter Options
    filter_option = st.selectbox("Select a Filter", ["None", "Grayscale", "Blur", "Remove Background"])

    if st.button("Apply Filter"):
        if filter_option == "Grayscale":
            image = apply_grayscale(image)
        elif filter_option == "Blur":
            image = apply_blur(image)
        elif filter_option == "Remove Background":
            image = remove_bg(image)

        # Resize the processed image to ensure it fits within the visible area
        image = resize_image(image, max_size=500)
        st.write("Processed Image Dimensions:", image.size)

        # Show Processed Image with a fixed maximum width
        st.image(image, caption="Edited Image", use_container_width=True)

        # Save Image for Download
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Download Button
        st.download_button("Download Edited Image", data=img_bytes, file_name="edited_img.png", mime="image/png")
import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
from autoencoder import AutoEncoder
from cnn import ColorizationNet
from imagenette_autoencoder import ImageNetAutoEncoder
from kornia.color import hsv_to_rgb, xyz_to_rgb
from switch_case import switch

st.title("Image Colorization using Deep Learning")

option = st.selectbox(
    'Select which model to use :',
    ('Autoencoder (RGB)', 'Autoencoder (XYZ)', 'Autoencoder (Imagenette)', 'CNN (CIFAR-10)')
)

device = "cuda" if torch.cuda.is_available() else "cpu"
mode = switch(option)
if mode == "imagenette":
    model = ImageNetAutoEncoder().to(device)
elif mode != "imagenette" and mode != "cnn":
    model = AutoEncoder().to(device)
else:
    model = ColorizationNet().to(device)
model_path = "./models/" + mode + "_model.pth"

if os.path.exists(model_path) == False:
    st.write("No existing model found. Please train, test, and save a model before running the app.")
    st.stop()
else:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))


st.write("Upload an image to colorize it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    image = Image.open(uploaded_file)
    grayscale_img = image.convert("L")
    if mode != "imagenette":
        image_size = (150, 150)
    else:
        image_size = (224, 224)
    resized_img = image.resize(image_size)
    resized_grayscale_img = grayscale_img.resize(image_size)
    with col1:
        st.image(resized_img, caption="Original Image", use_column_width=False)
    with col2:
        st.image(resized_grayscale_img, caption="Grayscale Image", use_column_width=False)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(image_size)])
    image_tensor = transform(grayscale_img).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        colorized_tensor = model(image_tensor)
    if mode == "hsv":
        colorized_image = hsv_to_rgb(colorized_tensor)
    elif mode == "xyz":
        colorized_image = xyz_to_rgb(colorized_tensor)
    colorized_image = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
    colorized_image = colorized_image.resize(image_size)
    with col3:
        st.image(colorized_image, caption="Prediction", use_column_width=False)
else:
    st.write("Please upload an image file to continue.")


def switch(model_name):
    if model_name == "Autoencoder (RGB)":
        return "rgb"
    elif model_name == "Autoencoder (HSV)":
        return "hsv"
    elif model_name == "Autoencoder (XYZ)":
        return "xyz"
    elif model_name == "Autoencoder (Imagenette)":
        return "imagenette"
    else:
        return "cnn"

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model.architecture import UNetLandmark
from preprocessing import preprocess_image
from postprocessing import heatmap_to_coords

LANDMARK_NAMES = [
    "A-point (Subspinale)",
    "Anterior Nasal Spine",
    "B-point (Supramentale)",
    "Menton",
    "Nasion",
    "Orbitale",
    "Pogonion",
    "Posterior Nasal Spine",
    "Ramus",
    "Sella",
    "Articulare",
    "Condylion",
    "Gnathion",
    "Gonion",
    "Porion",
    "Lower 2nd PM Cusp Tip",
    "Lower Incisor Tip",
    "Lower Molar Cusp Tip",
    "Upper 2nd PM Cusp Tip",
    "Upper Incisor Apex",
    "Upper Incisor Tip",
    "Upper Molar Cusp Tip",
    "Lower Incisor Apex",
    "Labrale inferius",
    "Labrale superius",
    "Soft Tissue Nasion",
    "Soft Tissue Pogonion",
    "Subnasale",
    "Pronasale"
]   


@st.cache_resource
def load_model():
    model = UNetLandmark()

    state_dict = torch.load(
        "model/detection_3_best.pth",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    return model


model = load_model()


st.title("Cephalometric Landmark Detection")

uploaded_file = st.file_uploader(
    "Upload the X-ray Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    processed_image = preprocess_image(image_array)

    input_tensor = processed_image.unsqueeze(0)

    with torch.no_grad():
        heatmaps = model(input_tensor)
        coordinates = heatmap_to_coords(
            heatmaps,
            image_array.shape
        )

    coordinates = coordinates[0].cpu().numpy()

    st.subheader("Detected Landmarks")

    fig, ax = plt.subplots()

    ax.imshow(image)

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=25
    )

    for i, (x, y) in enumerate(coordinates):
        ax.annotate(
            str(i + 1),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_xlim(0, image_array.shape[1])
    ax.set_ylim(image_array.shape[0], 0)
    ax.axis("off")

    st.pyplot(fig)

    st.subheader("Detected Landmarks")

    for i, (name, (x, y)) in enumerate(zip(LANDMARK_NAMES, coordinates), start=1):
        st.write(f"{i} - {name}")   
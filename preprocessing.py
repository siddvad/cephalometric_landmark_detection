import cv2
import numpy as np
import torch
import torch.nn.functional as F


def to_grayscale(img):
    img = img.astype(np.float32)

    if len(img.shape) == 2:
        return img

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    gray = (
        0.2989 * img[:, :, 0]
        + 0.5870 * img[:, :, 1]
        + 0.1140 * img[:, :, 2]
    )

    return gray


def apply_clahe(image):
    image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)


def to_normalize(img_array):
    mean = img_array.mean()
    std = img_array.std() + 1e-8

    return (img_array - mean) / std


def resize(img_array, size=(512, 512)):
    img = torch.from_numpy(img_array).float()

    img = img.unsqueeze(0).unsqueeze(0)

    img = F.interpolate(
        img,
        size=size,
        mode="bilinear",
        align_corners=False
    )

    return img.squeeze(0)


def preprocess_image(img_array):
    gray_image = to_grayscale(img_array)
    clahe_image = apply_clahe(gray_image)
    normalized_image = to_normalize(clahe_image)
    resized_image = resize(normalized_image)

    return resized_image
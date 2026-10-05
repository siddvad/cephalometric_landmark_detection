import torch


def heatmap_to_coords(heatmaps, original_shape):
    batch_size, num_landmarks, heatmap_height, heatmap_width = heatmaps.shape

    original_height, original_width = original_shape[:2]

    flat_indices = heatmaps.view(
        batch_size,
        num_landmarks,
        -1
    ).argmax(dim=2)

    y = flat_indices // heatmap_width
    x = flat_indices % heatmap_width

    x = x.float() * (original_width / heatmap_width)
    y = y.float() * (original_height / heatmap_height)

    coords = torch.stack([x, y], dim=2)

    return coords
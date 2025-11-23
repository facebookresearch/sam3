import cv2
import numpy as np
import torch

# Vibrant color palette (BGR format for OpenCV)
COLOR_PALETTE = [
    # (255, 42, 4),     # 0  - #042AFF   Deep Blue
    # (235, 219, 11),   # 1  - #0BDBEB   Aqua Blue
    # (183, 223, 0),    # 3  - #00DFB7   Teal Green
    # (104, 31, 17),    # 4  - #111F68   Navy Blue
    # (221, 111, 255),  # 5  - #FF6FDD   Pink/Magenta
    # (79, 68, 255),    # 6  - #FF444F   Red-Pink
    (0, 237, 204),    # 7  - #CCED00   Lime Yellow-Green
    # (68, 243, 0),     # 8  - #00F344   Neon Green
    # (255, 0, 189),    # 9  - #BD00FF   Purple
    # (255, 180, 0),    # 10 - #00B4FF   Sky Blue
    # (186, 0, 221),    # 11 - #DD00BA   Magenta Purple
    # (255, 255, 0),    # 12 - #00FFFF   Cyan
    # (0, 192, 38),     # 13 - #26C000   Bright Green
]


def get_color(i):
    """Get color from palette"""
    return COLOR_PALETTE[i % len(COLOR_PALETTE)]


def draw_box_and_masks(img_cv, results, show_boxes=True, show_masks=True,
                       mask_alpha=0.35, line_width=4, show_conf=False, label=None):
    """
    Fast and awesome visualization for SAM3 results

    Args:
        img_cv: Input image (BGR format)
        results: Dictionary with 'boxes', 'masks', 'scores'
        show_boxes: Whether to draw bounding boxes
        show_masks: Whether to draw masks
        mask_alpha: Transparency of mask overlay (0-1)
        show_conf: Whether to show confidence scores
        line_width: Line width for bounding boxes (affects text size and padding)
        label: Bounding box or mask label.
    """
    if not show_boxes and not show_masks:
        return img_cv

    result = img_cv.copy()
    total_objects = len(results["scores"])
    h, w = result.shape[:2]

    for i in range(total_objects):
        color = get_color(i)

        # Draw masks first (so boxes appear on top)
        if show_masks:
            mask = results["masks"][i].squeeze(0).cpu().numpy()
            if mask.shape != (h, w):  # Handle mask dimensions
                if mask.shape == (w, h):
                    mask = mask.T
                else:
                    mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_bool = mask.astype(bool)

            overlay = result.copy()
            overlay[mask_bool] = color  # Apply semi-transparent overlay
            result = cv2.addWeighted(result, 1 - mask_alpha, overlay, mask_alpha, 0)

            # Draw contour for clarity
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, max(1, line_width // 2))

        # Draw bounding boxes
        if show_boxes:
            box = results["boxes"][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            prob = results["scores"][i].item()  # confidence score
            box_width, box_height = x2 - x1, y2 - y1
            box_area = box_width * box_height  # Calculate box dimensions
            cv2.rectangle(result, (x1, y1), (x2, y2), color, line_width)  # bbox plotting
            label = f"{label}:{prob:.2f}" if show_conf else f"{label}"

            # Adaptive font scale based on bbox size and line width
            # Larger boxes and thicker lines = larger text
            base_scale = np.sqrt(box_area) / 400  # Base scale from box size
            font_scale = max(0.9, min(1.2, base_scale * line_width / 4))  # Line width influence
            text_thickness = max(1, line_width // 2)  # Text thickness scales with line width

            (text_w, text_h), baseline = cv2.getTextSize(label,
                                                         cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale,
                                                         text_thickness)

            padding = max(3, line_width)  # Padding scales with line width
            label_x, label_y = x1, y1 - padding - baseline  # Position label at top-left corner of box
            if label_y - text_h < 0:  # If label goes above image, put it inside the box
                label_y = y1 + text_h + padding
            label_x = max(0, min(label_x, w - text_w - 2 * padding))

            # Draw label background
            cv2.rectangle(result,
                          (label_x, label_y - text_h - padding),
                          (label_x + text_w + 2 * padding, label_y + baseline + padding),
                          color, -1)

            # Draw label text
            cv2.putText(result, label, (label_x + padding, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)

    return result
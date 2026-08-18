import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def detect_fish_in_image(image_path, lower_hsv, upper_hsv, lower_skin=None, upper_skin=None):
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if lower_skin is not None and upper_skin is not None:
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_rgb.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 100:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(result, [largest_contour], 0, (0, 255, 0), 2)

            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(result, f"Area: {area}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_rgb, result, mask


if __name__ == "__main__":
    fish_dir = r"D:\vision\mv\HSV\fish"
    output_dir = r"D:\vision\mv\HSV\detected"
    Path(output_dir).mkdir(exist_ok=True)

    lower_hsv = np.array([0, 10, 30])
    upper_hsv = np.array([40, 255, 255])

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    images = sorted(Path(fish_dir).glob("*.jpg"))
    num_images = len(images)

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(images):
        print(f"Processing: {img_path.name}")

        use_skin_detection = (idx == 4 or idx == 5)

        if use_skin_detection:
            img_rgb, result, mask = detect_fish_in_image(img_path, lower_hsv, upper_hsv, lower_skin, upper_skin)
        else:
            img_rgb, result, mask = detect_fish_in_image(img_path, lower_hsv, upper_hsv)

        if img_rgb is not None:
            axes[idx, 0].imshow(img_rgb)
            axes[idx, 0].set_title(f"{img_path.name} - Original")
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(result)
            axes[idx, 1].set_title(f"{img_path.name} - Detection")
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(mask, cmap='gray')
            axes[idx, 2].set_title(f"{img_path.name} - Mask")
            axes[idx, 2].axis('off')

            output_path = Path(output_dir) / f"detected_{img_path.name}"
            cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.use('Agg') 

def yolo_to_pixel(location, img_width, img_height):
    x_center, y_center, width, height = location
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    box_width = int(width * img_width)
    box_height = int(height * img_height)
    return x_min, y_min, box_width, box_height


def visualization(before_inference_path, data):
    # Load image
    image = Image.open(before_inference_path)
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    # 저장할 경로 설정
    output_path = f"result/detection/{os.path.basename(before_inference_path)}"

    # Define unique colors for diseases
    label_colors = sns.color_palette("hsv", n_colors=11)

    # Map disease names to specific colors
    disease_color_map = {
        "Caries": label_colors[0],
        "Gingivitis": label_colors[1],
        "Periodontitis": label_colors[2],
        "Oral Cancer": label_colors[3],
        "Erosion": label_colors[4],
        "Tartar": label_colors[5],
        "Fracture": label_colors[6],
        "Ulcer": label_colors[7],
        "Abscess": label_colors[8],
        "Discoloration": label_colors[9],
        "Other": label_colors[10]
    }

    # Add bounding boxes and labels for tooth diseases
    for tooth, diseases in data['tooth_diseases'].items():
        for disease in diseases:
            x_min, y_min, box_width, box_height = yolo_to_pixel(disease['location'], img_width, img_height)
            confidence = disease['confidence']
            disease_name = disease['disease_name']

            # Select color for disease
            color = tuple([int(c * 255) for c in disease_color_map.get(disease_name, [0, 255, 0])])

            # Draw bounding box
            draw.rectangle(
                [(x_min, y_min), (x_min + box_width, y_min + box_height)],
                outline=color,
                width=3
            )

            # Draw text label
            draw.text(
                (x_min, y_min - 15),
                f"{disease_name}",
                fill=color
            )

    # Add bounding boxes and labels for gum diseases
    for region, diseases in data['gum_diseases'].items():
        for disease in diseases:
            x_min, y_min, box_width, box_height = yolo_to_pixel(disease['location'], img_width, img_height)
            confidence = disease['confidence']
            disease_name = disease['disease_name']

            # Select color for disease
            color = tuple([int(c * 255) for c in disease_color_map.get(disease_name, [0, 255, 0])])

            # Draw bounding box
            draw.rectangle(
                [(x_min, y_min), (x_min + box_width, y_min + box_height)],
                outline=color,
                width=3
            )

            # Draw text label
            draw.text(
                (x_min, y_min - 15),
                f"{disease_name}",
                fill=color
            )

    # Add bounding boxes and labels for etc diseases by region
    for region, diseases in data['etc'].items():
        for disease in diseases:
            x_min, y_min, box_width, box_height = yolo_to_pixel(disease['location'], img_width, img_height)
            disease_name = disease['disease_name']

            # Select color for "etc" diseases
            color = tuple([int(c * 255) for c in disease_color_map.get(disease_name, [0, 255, 0])])

            # Draw bounding box
            draw.rectangle(
                [(x_min, y_min), (x_min + box_width, y_min + box_height)],
                outline=color,
                width=3
            )

            # Draw text label with disease name and region
            draw.text(
                (x_min, y_min - 15),
                f"{region}: {disease_name}",
                fill=color
            )
    if image.mode == "RGBA":
        image = image.convert("RGB")
    # Save the modified image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Final detection results directly on image saved at {output_path}")
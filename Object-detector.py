import gradio as gr
from PIL import Image, ImageFont,ImageDraw

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import pipeline

model_path = ("../Models/models--facebook--detr-resnet-50/snapshots"
              "/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b")

# pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

object_detection = pipeline("object-detection",
                             model=model_path)

def highlight_objects(image, detections):
    # Create a copy so the original image remains unchanged
    image = image.copy()

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    # Draw bounding boxes for each detected object
    for detection in detections:
        # Extract bounding box coordinates
        box = detection["box"]
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]

        # Extract label and confidence score
        label = detection["label"]
        score = detection["score"]

        # Draw the bounding box
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline="red",
            width=4
        )

        # Create label text
        text = f"{label}: {score:.2f}"

        # Calculate text background dimensions
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)

        # Draw background for the label
        draw.rectangle(
            text_bbox,
            fill="red"
        )

        # Draw label text
        draw.text(
            (xmin, ymin),
            text,
            fill="white",
            font=font
        )

    return image

def detect_object(image):
    raw_image = image
    output = object_detection(raw_image)
    processed_image = highlight_objects(raw_image, output)
    return processed_image

# print(output)

demo = gr.Interface(fn=detect_object,
    inputs=[gr.Image(label="Select Image", type ="pil")],
    outputs=[gr.Image(label="Processed Image", type ="pil")],
    title="GENAI PROJECT6: Object_Detection",
    description="THIS APPLICATION WILL BE USED TO DETECT OBJECT IN provided IMAGE")

demo.launch()
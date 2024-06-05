import cv2
import numpy as np

# Load pre-trained object detection model (e.g., YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Function to detect objects in the image
def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append((label, x, y, w, h))
    return detected_objects

# Function to calculate height and wingspan of the bird
def calculate_features(image, bird_box):
    bird_x, bird_y, bird_w, bird_h = bird_box
    # Calculate bird height (height of bounding box)
    bird_height = bird_h
    # Calculate bird wingspan (width of bounding box)
    bird_wingspan = bird_w
    return bird_height, bird_wingspan

# Function to extract color of wings
def extract_color(image, bird_box):
    bird_x, bird_y, bird_w, bird_h = bird_box
    # Extract region of interest (ROI) for wings
    wing_roi = image[bird_y: bird_y + bird_h, bird_x: bird_x + bird_w]
    # Convert ROI to HSV color space
    hsv_wing = cv2.cvtColor(wing_roi, cv2.COLOR_BGR2HSV)
    # Define range for detecting color (e.g., blue)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Create a mask for blue color
    mask = cv2.inRange(hsv_wing, lower_blue, upper_blue)
    # Count pixels with blue color
    blue_pixels = cv2.countNonZero(mask)
    return blue_pixels

# Main function to process image and extract features
def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Detect objects in the image
    detected_objects = detect_objects(image)
    # Assume the first detected object is the bird
    if detected_objects:
        bird_label, bird_x, bird_y, bird_w, bird_h = detected_objects[0]
        bird_box = (bird_x, bird_y, bird_w, bird_h)
        # Calculate features
        bird_height, bird_wingspan = calculate_features(image, bird_box)
        # Extract color
        blue_pixels = extract_color(image, bird_box)
        # Display results
        print("Bird Label:", bird_label)
        print("Bird Height:", bird_height)
        print("Bird Wingspan:", bird_wingspan)
        print("Blue Pixels (Wing Color):", blue_pixels)
    else:
        print("No bird detected in the image.")

# Example usage
if __name__ == "__main__":
    # Path to the bird image in your dataset
    image_path = "./static/uploads/Black_Footed_Albatross_0005_796090.jpg"
    # Process the image
    process_image(image_path)

import cv2
import numpy as np

def extract_bird_information(image_path, actual_size_cm):
    # Load the image
    image = cv2.imread(image_path)
    
    # Preprocess the image if necessary (resize, normalize, etc.)
    # Preprocessing steps...
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the bird
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of the bird
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the bird
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate conversion factor (pixels per centimeter)
    pixels_per_cm = max(h, w) / actual_size_cm
    
    # Convert height and wingspan from pixels to centimeters
    height_cm = h / pixels_per_cm
    wingspan_cm = w / pixels_per_cm
    
    # Extract average color of the bird
    bird_roi = image[y:y+h, x:x+w]
    avg_color = np.mean(bird_roi, axis=(0, 1)).astype(int)
    
    # Extract dominant colors using k-means clustering
    num_colors = 5  # You can adjust this parameter based on your needs
    bird_pixels = bird_roi.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(bird_pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_colors = centers.astype(int)
    
    # Return the extracted information
    return {
        'height_cm': height_cm,
        'wingspan_cm': wingspan_cm,
        'avg_color': avg_color,
        'dominant_colors': dominant_colors
    }

# Example usage
image_path = './static/uploads/Black_Footed_Albatross_0005_796090.jpg'
actual_size_cm = 20  # Assuming the bird is 20 cm tall in the image
bird_information = extract_bird_information(image_path, actual_size_cm)
print(bird_information)

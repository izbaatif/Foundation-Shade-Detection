import pickle
import numpy as np
import pandas as pd
import cv2
import os
import math
from sklearn.cluster import KMeans
from collections import Counter
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Correct import for CORS
from werkzeug.utils import secure_filename
import io
from PIL import Image

# Path to pickle file
pickle_file_path = "random_forest_model.pkl"

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    loaded_rf_model = pickle.load(file)

labels = ["#a47860", "#ac8775", "#b78f7d", "#c19983", "#c7a18e", "#d6ac99", "#dba382", "#f3c1ac", "#f6be97", "#fed9ba"]

class ColorExtractionPipeline:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters

    def get_labels(self, pixels):
        clf = KMeans(n_clusters=self.n_clusters)
        labels = clf.fit_predict(pixels)
        return labels, clf


    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def HEX2RGB(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)])

    def extract_colors(self, skin_pixels):
        reshaped_pixels = skin_pixels.reshape(-1, 3)
        labels, clf = self.get_labels(reshaped_pixels)
        counts = Counter(labels)
        center_colors = clf.cluster_centers_
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        most_dominant_indices = sorted(counts, key=counts.get, reverse=True)[:10]
        dominant_colors = [center_colors[i] for i in most_dominant_indices]
        return hex_colors, dominant_colors, center_colors

    def closest_to_white(self, center_colors):
        return min(center_colors, key=lambda c: np.linalg.norm(c - [255, 255, 255]))

    def average_of_colors(self, colors):
        return np.mean(colors, axis=0)

    def weighted_average(self, color1, color2, weight1, weight2):
        return (color1 * weight1) + (color2 * weight2)

    def __call__(self, skin_pixels):
        return self.extract_colors(skin_pixels)

color_extraction = ColorExtractionPipeline(n_clusters=10)

def allowed_file(filename):
    """
    Check if a filename has a valid extension.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_tone(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    #now check if there are any faces detected
    if len(faces) == 0:
        return False
    else:
        #Select the largest face by getting max
        face_detected = max(faces, key=lambda rect: rect[2] * rect[3])  

        #get the x,y, width and height of the face
        x,y,h,w=face_detected

        #Crop the image to get the face
        face = image[y:y+h, x:x+w]

    # Convert the image to HSV
    hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    
    # Define minimum and maximum values for skin HSV
    lower_hsv = np.array([0, 40, 30], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    
    # Create a mask to only get pixels in the specified range
    skin_mask = cv2.inRange(hsv_face, lower_hsv, upper_hsv)
    
    # Apply the mask to the image to detect skin pixels
    skin_pixels_bgr = cv2.bitwise_and(face, face, mask=skin_mask)
    
    # Convert skin pixels from BGR to RGB
    skin_pixels_rgb = cv2.cvtColor(skin_pixels_bgr, cv2.COLOR_BGR2RGB)

    #Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Define thresholds for white areas
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)

    #Create a mask for white areas
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    #Detect white areas in the image
    white_areas = cv2.bitwise_and(image, image, mask=white_mask)

    #Calculate the average color of the white areas
    non_zero_pixels = white_areas[white_mask == 255]
    if non_zero_pixels.size > 0:
        avg_white = np.mean(non_zero_pixels, axis=0)
    else:
        avg_white = [0, 0, 0]
    
    #extract colors from class
    colors, dominant_colors, center_colors = color_extraction(skin_pixels_rgb)
        
    # Get the closest color to white
    closest_to_white_rgb = color_extraction.closest_to_white(center_colors)
        
    # Calculate the average of middle 2 colors to avoid areas like hair
    average_skin_tone = color_extraction.average_of_colors(dominant_colors[3:7])

    # Calculate the average of the 2nd and 3rd dominant colors and the closest to white color
    average_with_lightest = color_extraction.weighted_average(average_skin_tone, closest_to_white_rgb, 0.9, 0.3)

    #Convert rbg values to numpy arrays
    skin_rgb = np.array(average_with_lightest, dtype=float)
    white_rgb = np.array(avg_white, dtype=float)
    true_white = np.array([255, 255, 255], dtype=float)

    if np.array_equal(white_rgb, [0, 0, 0]):
        # If no white areas were detected, return the original skin tone
        corrected_skin_rgb = np.clip(np.round(skin_rgb), 0, 255).astype(int)

    else:
        # Calculate the difference between true white and detected white
        white_difference = true_white - white_rgb

        # Adjust skin tone by subtracting the weighted difference
        corrected_skin = skin_rgb + (white_difference * 0.5)

        # Clip the values to stay within 0 and 255 and round them
        corrected_skin_rgb = np.clip(np.round(corrected_skin), 0, 255).astype(int)

    tone=corrected_skin_rgb.astype(int)

    # Convert the average skin tone to HEX
    hex_final_skin_tone = color_extraction.RGB2HEX(tone)

    return hex_final_skin_tone

def get_x(skin_hex):
    skin_rgb = color_extraction.HEX2RGB(skin_hex)
    skin_red, skin_green, skin_blue = skin_rgb
    brightness = 0.2989 * skin_red + 0.587 * skin_green + 0.114 * skin_blue
    red_ratio = skin_red / (skin_red + skin_green + skin_blue)
    green_ratio = skin_green / (skin_red + skin_green + skin_blue)
    blue_ratio = skin_blue / (skin_red + skin_green + skin_blue)
    normalised_red = skin_red / 255.0
    normalised_green = skin_green / 255.0
    normalised_blue = skin_blue / 255.0
    normalised_brightness = brightness / 255.0

    return [[normalised_red, normalised_green, normalised_blue, normalised_brightness, red_ratio, green_ratio, blue_ratio]]

foundations = pd.read_csv("../data/Foundation Shades.csv")

def find_closest_match(skin_hex, df):
    closest_match = None
    min_distance = float('inf')
    skin_rgb = color_extraction.HEX2RGB(skin_hex)

    for _, row in foundations.iterrows():
        makeup_rgb = color_extraction.HEX2RGB(row['Hex'])
        distance = np.linalg.norm(skin_rgb - makeup_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_match = row

    return closest_match

app = Flask(__name__)
CORS(app)  # This will allow all origins to access the server

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        # Process the file without saving it
        image = Image.open(io.BytesIO(file.read()))
        tone = get_tone(image)
        if not tone:
            return jsonify({"error": "No face detected"}), 400

        y_pred = loaded_rf_model.predict(get_x(tone))
        predicted_class = np.argmax(y_pred)
        predicted_hex = labels[predicted_class]
        closest_match = find_closest_match(tone, foundations)

        return jsonify({
            "skin_tone": predicted_hex,
            "closest_match": {
                "brand": closest_match["Brand"],
                "product": closest_match["Product"],
                "shade": closest_match["Shade"]
            }
        })

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(host='localhost', port=3000)

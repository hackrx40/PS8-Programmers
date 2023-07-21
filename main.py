



 







from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2
def capture_screenshot(url, output_path):
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    
    # Initialize the Chrome WebDriver with the specified options
    driver = webdriver.Chrome(executable_path='path/to/chromedriver', options=chrome_options)
    
    try:
        # Open the URL in the browser
        driver.get(url)
        
        # Wait for the page to load (you may need to adjust the waiting time based on your website)
        driver.implicitly_wait(10)
        
        # Capture a screenshot of the page
        driver.save_screenshot(output_path)
        print("Screenshot captured successfully.")
    except Exception as e:
        print("Error occurred:", e)
    finally:
        # Close the browser
        driver.quit()

def compare_images(image1_path, image2_path):
    # Read the images from file
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert the images to grayscale (required for some comparison methods)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform image comparison using structural similarity (SSIM) index
    ssim_index, diff_image = cv2.compareSSIM(gray_image1, gray_image2, full=True)

    # Threshold for considering images as similar or different (adjust this based on your needs)
    similarity_threshold = 0.95

    if ssim_index > similarity_threshold:
        print("Website layout is similar.")
    else:
        print("Website layout is different.")

if __name__ == "__main__":
    # URL of the website you want to test
    website_url = "https://www.example.com"

    # Output path to save the screenshots
    screenshot_output = "path/to/screenshot.png"

    # Capture the screenshot of the website
    capture_screenshot(website_url, screenshot_output)

    # Path to the reference screenshot for comparison
    reference_screenshot = "path/to/reference_screenshot.png"

    # Perform visual testing by comparing the screenshots
    compare_images(reference_screenshot, screenshot_output)


import cv2

def extract_layout_from_screenshot(image_path):
    # Read the screenshot
    screenshot = cv2.imread(image_path)

    # Convert the screenshot to grayscale
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Apply edge detection or other preprocessing as needed
    edges = cv2.Canny(gray_screenshot, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process the contours and gather layout information
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print("Element position (x, y):", x, y)
        print("Element size (width, height):", w, h)

if __name__ == "__main__":
    # Path to the screenshot image
    screenshot_path = "path/to/screenshot.png"

    # Extract layout information from the screenshot
    extract_layout_from_screenshot(screenshot_path)


import requests

def get_figma_file(figma_file_id, access_token):
    headers = {
        "X-Figma-Token": access_token
    }
    url = f"https://api.figma.com/v1/files/{figma_file_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        figma_data = response.json()
        return figma_data
    else:
        print("Error:", response.status_code)
        return None

if __name__ == "__main__":
    # Figma file ID and access token
    figma_file_id = "your_figma_file_id"
    access_token = "your_figma_access_token"

    # Get the Figma file data
    figma_data = get_figma_file(figma_file_id, access_token)

    # Process the Figma data to gather layout information
    # (depends on the specific data structure returned by the Figma API)
    # Extract position, size, and other layout details of design elements

import cv2
import numpy as np
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)
vk_size = (5, 5)  # Set the kernel size for blurring
image_blurred = cv2.GaussianBlur(image, k_size, 0)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
alpha = 1.5  # Increase or decrease for different levels of contrast
enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
beta = 50  # Increase or decrease for different levels of brightness
enhanced_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
# Load the VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Print the model summary to see the layers
base_model.summary()
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()


import cv2
import numpy as np

# Load the image in grayscale mode
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Set a threshold value (adjust this value based on your image)
threshold_value = 128

# Threshold the image (pixels with intensity >= threshold_value will be set to 255, others to 0)
_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Display the original and binary images using OpenCV
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()


# Example usage
image_path = 'path_to_your_image.jpg'
extracted_features = extract_features(image_path, base_model)

import tkinter as tk
from tkinter import messagebox

def submit_url():
    url = url_entry.get()
    if url:
        # You can perform any additional validation here if needed
        messagebox.showinfo("Success", f"URL '{url}' added successfully!")
        url_entry.delete(0, tk.END)
    else:
        messagebox.showwarning("Error", "Please enter a valid URL!")

# Create the main window
root = tk.Tk()
root.title("Website URL Input")

# Create the URL input field
url_label = tk.Label(root, text="Enter Website URL:")
url_label.pack(pady=10)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

# Create the Submit button
submit_button = tk.Button(root, text="Submit", command=submit_url)
submit_button.pack(pady=10)

# Start the main event loop
root.mainloop()

import unittest

def add_numbers(a, b):
    return a + b


class TestAddition(unittest.TestCase):

    def test_addition_positive_numbers(self):
        result = add_numbers(2, 3)
        self.assertEqual(result, 5)

    def test_addition_negative_numbers(self):
        result = add_numbers(-2, -3)
        self.assertEqual(result, -5)

if __name__ == "__main__":
    unittest.main()

def add_numbers(a, b):
    return a + b

def test_addition_positive_numbers():
    result = add_numbers(2, 3)
    assert result == 5

def test_addition_negative_numbers():
    result = add_numbers(-2, -3)
    assert result == -5


import cv2

# Load the image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred_image, 50, 150)

# Display the original image and processed edges
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()


from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
sample_data = {
    "items": [
        {"id": 1, "name": "Item 1", "price": 10.99},
        {"id": 2, "name": "Item 2", "price": 15.99},
        {"id": 3, "name": "Item 3", "price": 20.50}
    ]
}

# Endpoint to get all items
@app.route('/items', methods=['GET'])
def get_all_items():
    return jsonify(sample_data)

# Endpoint to get an item by ID
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item_by_id(item_id):
    item = next((item for item in sample_data["items"] if item["id"] == item_id), None)
    if item:
        return jsonify(item)
    else:
        return jsonify({"message": "Item not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

import requests

# Get all items
response = requests.get('http://localhost:5000/items')
print(response.json())

# Get an item by ID
response = requests.get('http://localhost:5000/items/2')
print(response.json())


import requests
from bs4 import BeautifulSoup
from pyquery import PyQuery as pq

def extract_layout(url):
    try:
        # Fetch the website's HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Get all elements with their tags and attributes
        layout = {}
        for element in soup.find_all(True):
            tag = element.name
            attributes = element.attrs
            layout[tag] = layout.get(tag, 0) + 1

        # Parse CSS styles using pyquery
        css_styles = pq(html_content)
        layout_styles = {}
        for style_element in css_styles('style').items():
            styles = style_element.text().split('\n')
            for style in styles:
                if '{' in style and '}' in style:
                    selector, properties = style.split('{')
                    selector = selector.strip()
                    properties = properties.split('}')[0].strip()
                    layout_styles[selector] = properties

        return layout, layout_styles

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)
        return None, None

if __name__ == "__main__":
    website_url = "https://example.com"  # Replace with the URL of the website you want to analyze
    layout, layout_styles = extract_layout(website_url)
    if layout and layout_styles:
        print("Website Layout:")
        print(layout)
        print("\nCSS Styles:")
        for selector, properties in layout_styles.items():
            print(f"{selector}: {properties}")
    else:
        print("Layout extraction failed.")


import requests
from bs4 import BeautifulSoup

def extract_layout(url):
    try:
        # Fetch the website's HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Initialize layout dictionary to store tag counts
        layout = {}

        # Extract layout by counting tags
        for element in soup.find_all(True):
            tag = element.name
            layout[tag] = layout.get(tag, 0) + 1

        return layout

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)
        return None

if __name__ == "__main__":
    website_url = "https://example.com"  # Replace with the URL of the website you want to analyze
    layout = extract_layout(website_url)
    if layout:
        print("Website Layout:")
        for tag, count in layout.items():
            print(f"{tag}: {count}")
    else:
        print("Layout extraction failed.")

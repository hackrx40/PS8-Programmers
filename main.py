



 







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



import requests
from bs4 import BeautifulSoup

def extract_layout(url):
    try:
        # Fetch the webpage content using requests
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")
            return None

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract layout information (e.g., tags and their attributes)
        layout_info = []
        for tag in soup.find_all():
            tag_name = tag.name
            attributes = {attr: tag[attr] for attr in tag.attrs}
            layout_info.append((tag_name, attributes))

        return layout_info

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    url_to_extract = "https://www.bajajfinserv.in/"  # Replace with the URL of the website you want to extract the layout from
    layout_data = extract_layout(url_to_extract)
    if layout_data:
        print(layout_data)


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  # Disable CSRF protection for simplicity (not recommended for production)
def process_data(request):
    if request.method == 'POST':
        data = request.POST.get('data', None)
        if data:
            # Call your software's processing function and get the result
            # Replace the following line with the actual processing logic of your software
            processed_data = your_software_process_function(data)

            # Return the processed data as a JSON response
            return JsonResponse({'result': processed_data})
        else:
            return JsonResponse({'error': 'Invalid data'})
    else:
        return JsonResponse({'error': 'Only POST method is supported'})


from django.urls import path
from . import views

urlpatterns = [
    path('process/', views.process_data, name='process_data'),
]

INSTALLED_APPS = [
    # ... other apps ...
    'my_app',
]



# my_app/utils.py
def process_data(input_data):
    # Your software's processing logic goes here
    # Replace the following line with your actual processing code
    return f"Processed: {input_data}"
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import process_data

@csrf_exempt  # Disable CSRF protection for simplicity (not recommended for production)
def process_data_api(request):
    if request.method == 'POST':
        data = request.POST.get('data', None)
        if data:
            processed_result = process_data(data)
            return JsonResponse({'result': processed_result})
        else:
            return JsonResponse({'error': 'Invalid data'})
    else:
        return JsonResponse({'error': 'Only POST method is supported'})
from django.urls import path
from . import views

urlpatterns = [
    path('api/process/', views.process_data_api, name='process_data_api'),
]
INSTALLED_APPS = [
    # ... other apps ...
    'my_app',
]


import requests
from bs4 import BeautifulSoup

def extract_layout(url):
    try:
        # Fetch the webpage content using requests
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")
            return None

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract layout information (e.g., tags and their attributes)
        layout_info = []
        for tag in soup.find_all():
            tag_name = tag.name
            attributes = {attr: tag[attr] for attr in tag.attrs}
            layout_info.append((tag_name, attributes))

        return layout_info

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    url_to_extract = "https://www.bajajfinserv.in/"  # Replace with the URL of the website you want to extract the layout from
    layout_data = extract_layout(url_to_extract)
    if layout_data:
        print(layout_data)

#
# 
# web site layout comparision

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
    website_url1 = "https://www.bajajfinserv.in/"  # Replace with the URL of the first website
    website_url2 = "https://www.bajajfinserv.in/"  # Replace with the URL of the second website

    layout1 = extract_layout(website_url1)
    layout2 = extract_layout(website_url2)

    if layout1 and layout2:
        print("Layout of Website 1:")
        for tag, count in layout1.items():
            print(f"{tag}: {count}")

        print("\nLayout of Website 2:")
        for tag, count in layout2.items():
            print(f"{tag}: {count}")

        # Compare layout of both websites
        if layout1 == layout2:
            print("\nThe layouts are identical.")
        else:
            print("\nThe layouts are different.")
    else:
        print("Layout extraction failed.")

#
#
# figma layout comparision
import requests
from bs4 import BeautifulSoup

def get_figma_title(figma_url):
    # Extracts the title from the Figma design link
    # You may need to replace this function with Figma API calls if you want to extract other data.
    # For simplicity, we'll just retrieve the page and use BeautifulSoup to find the title tag.
    response = requests.get(figma_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
    return None

def get_website_title(website_url):
    # Extracts the title from the other website link
    response = requests.get(website_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
    return None

def main():
    figma_url = "https://www.figma.com/file/1LbWxAvQ12spYAKQGvDUe4/Personal-Loan-PDP?type=design&node-id=1%3A4695&mode=design&t=MLzndGKsvU2LyF9P-1"
    website_url = "https://www.bajajfinserv.in/"  # Replace this with the website you want to compare

    figma_title = get_figma_title(figma_url)
    website_title = get_website_title(website_url)

    if figma_title and website_title:
        if figma_title == website_title:
            print("The Figma design matches the website layout.")
        else:
            print("The Figma design does not match the website layout.")
    else:
        print("Failed to retrieve data from either the Figma design or the website.")

if __name__ == "__main__":
    main()



#####
#rating
#

from bs4 import BeautifulSoup
import requests

def extract_layout(url):
    try:
        # Fetch the website's HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract layout by collecting unique HTML tags
        layout = set(tag.name for tag in soup.find_all(True))

        return layout

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)
        return None

def calculate_jaccard_similarity(layout1, layout2):
    # Calculate Jaccard similarity index
    intersection = len(layout1.intersection(layout2))
    union = len(layout1.union(layout2))
    jaccard_similarity = intersection / union if union != 0 else 0

    return jaccard_similarity

# Replace with the URLs of the websites you want to compare
website_url1 = "https://www.bajajfinserv.in/personal-loan"
website_url2 = "https://www.bajajfinserv.in/doctor-loan"

# Extract layout from the websites
layout1 = extract_layout(website_url1)
layout2 = extract_layout(website_url2)

if layout1 and layout2:
    print("Layout of Website 1:", layout1)
    print("Layout of Website 2:", layout2)

    # Calculate similarity between the layouts
    similarity_rating = calculate_jaccard_similarity(layout1, layout2)
    print("Rating of Website based on Jaccard Similarity:", similarity_rating*100 ,"%")
else:
    print("Layout extraction failed.")

    #
    # API generation
    # 

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  # Disable CSRF protection for simplicity (not recommended for production)
def process_data(request):
    if request.method == 'POST':
        data = request.POST.get('data', None)
        if data:
            # Call your software's processing function and get the result
            # Replace the following line with the actual processing logic of your software
            processed_data = your_software_process_function(data)
            your_software_process_function : any
            # Return the processed data as a JSON response
            return JsonResponse({'result': processed_data})
        else:
            return JsonResponse({'error': 'Invalid data'})
    else:
        return JsonResponse({'error': 'Only POST method is supported'})

from django.urls import path
from . import views

urlpatterns = [
    path('process/', views.process_data, name='process_data'),
]

INSTALLED_APPS = [
    # ... other apps ...
    'my_app',
]    
    

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

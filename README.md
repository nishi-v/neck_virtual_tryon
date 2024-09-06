# Earring Virtual Try-On Application

This Streamlit application allows users to try on virtual earrings using their webcam. Users can select an earring, capture an image of their ear, and see how the selected earring looks on them.

## Features:
- Earring Selection: Choose from a list of earrings to try on.
- Image Capture: Capture an image of your ear using your webcam.
- Earring Overlay: Overlay the selected earring on the detected ear in the captured image.

## Requirements:
- Python 3.7 or later
- Required Python packages:
  - numpy
  - opencv-python-headless
  - streamlit
  - requests
  - Pillow
  - python-dotenv

You can install the required packages using pip:
pip install numpy opencv-python-headless streamlit requests Pillow python-dotenv

## Setup:
1. Clone the Repository:
   git clone https://github.com/nishi-v/ear_virtual_tryon
   cd ear_virtual_tryon

2. Create a .env File:
   In the project directory, create a .env file with the following content:
   API_URL=your_api_url
   Replace your_api_url with the URL of the API used for detecting ear coordinates.

3. Run the Application:
   Start the application with Streamlit:
   streamlit run ear_vto.py
   This will open the Streamlit app in your default web browser.

## How It Works:
1. Earring Selection:
   - Users are presented with images of available earrings.
   - They can click the "Try On" button to select an earring.

2. Image Capture:
   - After selecting an earring, users are prompted to capture an image of their ear using the webcam.

3. Earring Overlay:
   - The captured image is sent to an API that detects the ear's coordinates.
   - The selected earring is resized and overlaid on the detected ear(s).
   - The final image with the earring overlay is displayed.

## Code Explanation:
- Environment Variables: Loaded using dotenv to fetch the API URL.
- Earring Selection: Displayed using Streamlit's image and button widgets.
- Image Capture and API Request: Captures an image, sends it to the API, and processes the response to detect ear coordinates.
- Overlay Earring: Resizes and overlays the selected earring on the detected ear(s) using OpenCV and PIL.

## Troubleshooting:
- Ensure that the .env file is correctly configured with the API URL.
- Make sure all required Python packages are installed.

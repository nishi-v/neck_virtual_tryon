# Necklace Virtual Try-On

## Description

This application allows users to virtually try on necklaces by overlaying selected necklace images onto a captured photo of their neck. The app uses image processing techniques to adjust and align the necklace image based on detected neck coordinates.

## Features

- Select from a list of necklace images.
- Capture an image of your neck using the camera.
- Overlay the selected necklace on the captured image.
- Automatically adjust and align the necklace based on detected neck coordinates.

## Installation

### Prerequisites

- Python 3.10
- Anaconda (for managing environments)
- Required Python packages: `numpy`, `opencv-python`, `requests`, `Pillow`, `streamlit`, `python-dotenv`

### Setup

1. Clone the repository:
   
   git clone https://github.com/nishi-v/neck_virtual_tryon.git
   
3. Navigate to the project directory:

   cd neck_virtual_tryon
   
5. Create and activate the Conda environment:

   conda env create -f environment.yml
   
   conda activate your-environment-name
   
7. Install any additional dependencies:

   pip install -r requirements.txt

9. Create a `.env` file in the root directory with the following content:

   API_URL=your_api_url

## Usage

1. Run the Streamlit app:

   streamlit run neck_vto.py


3. Open the application in your browser.
4. Select a necklace from the available options.
5. Capture an image of your neck using the camera.
6. The app will process the image and overlay the selected necklace on the captured photo.
7. View the virtual try-on result.

## Troubleshooting:
- Ensure that the .env file is correctly configured with the API URL.
- Make sure all required Python packages are installed.

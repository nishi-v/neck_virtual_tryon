import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw
import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# Get the current working directory
dir = Path(os.getcwd())

# Load environment variables from .env file
ENV_PATH = dir / '.env'
load_dotenv(ENV_PATH)

# Get API URL from environment variables
API_URL = os.environ["API_URL"]
BEARER_TOKEN = os.environ["BEARER_TOKEN"]

st.title('Necklace Virtual Try-On')

# Initialize session state if not already done
if "necklace_selected" not in st.session_state:
    st.session_state.necklace_selected = False
    st.session_state.type_selected = False
    st.session_state.type_detected = []
    st.session_state.neck_to_coords = {}  # Store neck data for each selection
    st.session_state.selected_necklace = None
    st.session_state.necklace_img = None
    st.session_state.type = None  # Initialize type in session state

# Define necklaces
necklaces = {
    "Necklace 1": "necklace/NFIS004S1F.png",
    "Necklace 2": "necklace/NFPW002G1F.png",
    "Necklace 3": "necklace/NFPW003S1F.png",
    "Necklace 4": "necklace/NFTG001S1F.png",
    "Necklace 5": "necklace/NMP006S1F.png",
    "Necklace 6": "necklace/NMPW005G1F.png",
    "Necklace 7": "necklace/download.png"
}

# Function to get the leftmost and rightmost points of the necklace
def get_necklace_endpoints(necklace_img_np: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    grayscale_img = cv2.cvtColor(necklace_img_np[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(grayscale_img, 1, 255, cv2.THRESH_BINARY)
    non_zero_points = cv2.findNonZero(binary_img)

    if non_zero_points is None:
        st.error("No non-zero points found in necklace image.")
        #return #None, None 

    # Find the leftmost and rightmost points of the necklace
    left_point = tuple(non_zero_points[non_zero_points[:, :, 0].argmin()][0])
    right_point = tuple(non_zero_points[non_zero_points[:, :, 0].argmax()][0])

    # Compute the inward adjustment factor (10%)
    adjustment_factor = 0.1

    # Move the left and right points inward by 10%
    def move_point_toward_center(start_point, end_point, factor):
        return (
            int(start_point[0] + factor * (end_point[0] - start_point[0])),
            int(start_point[1] + factor * (end_point[1] - start_point[1]))
        )

    # Move the left point toward the right point by 10%
    left_adjusted = move_point_toward_center(left_point, right_point, adjustment_factor)
    
    # Move the right point toward the left point by 10%
    right_adjusted = move_point_toward_center(right_point, left_point, adjustment_factor)

    return left_adjusted, right_adjusted

def overlay_necklace(face_img: np.ndarray, necklace_img: np.ndarray):
    if st.session_state.type is None:
        st.error("No neck type selected.")
        return None

    left_end_necklace, right_end_necklace = get_necklace_endpoints(necklace_img)

    neck_coordinates = st.session_state.neck_to_coords
    if st.session_state.type not in neck_coordinates:
        st.error("Neck type not found.")
        return None

    type_data = neck_coordinates[st.session_state.type]
    left_neck = tuple(type_data["left_point"])
    right_neck = tuple(type_data["right_point"])
    rotation_angle = type_data["rotation_angle"]

    img_height, img_width = face_img.shape[:2]
    left_neck_pixel = (int(left_neck[0] * img_width), int(left_neck[1] * img_height))
    right_neck_pixel = (int(right_neck[0] * img_width), int(right_neck[1] * img_height))

    # Calculate the neck center and average y-coordinate
    neck_center_pixel = (
        int((left_neck_pixel[0] + right_neck_pixel[0]) / 2),
        int((left_neck_pixel[1] + right_neck_pixel[1]) / 2)
    )
    average_y_neck = (left_neck_pixel[1] + right_neck_pixel[1]) // 2

    # Calculate the distance and angle between neck points
    neck_distance = np.linalg.norm(np.array(left_neck_pixel) - np.array(right_neck_pixel))
    necklace_distance = np.linalg.norm(np.array(left_end_necklace) - np.array(right_end_necklace))
    
    scaling_factor = neck_distance / necklace_distance if necklace_distance > 0 else 1
    
    # Resize the necklace image based on the scaling factor
    new_width = int(necklace_img.shape[1] * scaling_factor)
    new_height = int(necklace_img.shape[0] * scaling_factor)
    resized_necklace = cv2.resize(necklace_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate angle between neck points
    neck_angle = np.degrees(np.arctan2(right_neck_pixel[1] - left_neck_pixel[1], 
                                       right_neck_pixel[0] - left_neck_pixel[0]))
    
    # Flip the sign of the angle to correct the direction
    neck_angle = - neck_angle  # Flip the angle for correct direction
    st.write(f"Given neck angle: {rotation_angle}")
    st.write(f"Rotation angle calculated: {neck_angle}")
    st.write("Using neck angle calculated using angle between neck points.")

    # Rotate the resized necklace around its center by the corrected neck angle
    rotation_matrix = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), neck_angle, 1)
    rotated_necklace = cv2.warpAffine(resized_necklace, rotation_matrix, (new_width, new_height), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Calculate offsets to align the necklace with neck points
    resized_left_end_necklace, resized_right_end_necklace = get_necklace_endpoints(rotated_necklace)

    # Set y_offset to align the top of the necklace with the average y of the neck points
    y_offset = average_y_neck #- (new_height // 10)  # Adjust so the top aligns with the neck
    x_offset = neck_center_pixel[0] - (new_width // 2)  # Center the necklace horizontally

    # Overlay the rotated necklace onto the face image
    overlay = Image.fromarray(face_img).convert("RGBA")
    necklace_overlay = Image.fromarray(rotated_necklace, "RGBA")
    overlay.paste(necklace_overlay, (x_offset, y_offset), necklace_overlay)

    # Convert the overlay back to NumPy for OpenCV drawing
    overlay_img_np = np.array(overlay)

    # Convert the overlay image to BGR for OpenCV operations
    overlay_img_bgr = cv2.cvtColor(overlay_img_np, cv2.COLOR_RGBA2BGR)
    
    # Define the color for the points (red) and contours (green)
    red_color = (255, 0, 0)  # BGR format for red
    green_color = (0, 255, 0)  # BGR format for green

    # Draw red circles at the neck points
    point_radius = int(img_height/200)
    for point in [left_neck_pixel, right_neck_pixel]:
        cv2.circle(overlay_img_bgr, point, point_radius, red_color, -1)

    # Convert the rotated necklace to grayscale for contour detection
    necklace_gray = cv2.cvtColor(np.array(necklace_overlay), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(necklace_gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded necklace image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours (green) around the necklace
    if contours:
        cv2.drawContours(overlay_img_bgr, contours, -1, green_color, 2, offset=(x_offset, y_offset))

    # Convert the image back to RGBA to maintain transparency
    overlay_img_rgba = cv2.cvtColor(overlay_img_bgr, cv2.COLOR_BGR2RGBA)

    # Convert back to PIL for final output
    result_img = Image.fromarray(overlay_img_rgba)
    return result_img

if not st.session_state.necklace_selected:
    for name, image_path in necklaces.items():
        try:
            necklace_img = Image.open(image_path).convert("RGBA")
        except Exception as e:
            st.error(f"Error loading image {image_path}: {e}")
            continue

        st.image(necklace_img, caption=name, width=200)
        
        if st.button(f"Try On {name}"):
            st.session_state.necklace_selected = True
            st.session_state.selected_necklace = name
            st.session_state.necklace_img = necklace_img
            break

else:
    st.image(st.session_state.necklace_img, caption="Selected Necklace", width=200) # type: ignore

    # Provide options to either upload or capture an image
    option = st.radio("Choose Image Source", ("Capture Image", "Upload Image"))

    if option == "Capture Image":
        # Streamlit widget to capture an image using the webcam
        camera_image = st.camera_input("Capture an image of the neck")
        if camera_image is not None:
            with open(dir / "temp_image_cam.jpg", "wb") as f:
                f.write(camera_image.getbuffer())
            img_path = str(dir / 'temp_image_cam.jpg')

    elif option == "Upload Image":
        # Streamlit widget to upload an image
        uploaded_image = st.file_uploader("Upload an image of the neck", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            with open(dir / "temp_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())
            img_path = str(dir / 'temp_image.jpg')

    # Proceed if either a camera or uploaded image is available
    if (option == "Capture Image" and camera_image) or (option == "Upload Image" and uploaded_image):
        start = time.time()

        payload = {'isEar': 'false', 'isNeck': 'true'}
        files = [('image', (img_path, open(img_path, 'rb'), 'image/jpeg'))]
        headers = {
            'Authorization': f"Bearer {BEARER_TOKEN}"
        }
        response = requests.post(API_URL, headers=headers, data=payload, files=files, verify=False)
        end = time.time() - start
        st.write(f"Time taken: {end}")
        results = response.text
        try:
            data = json.loads(results)

            neck_coordinates = data["results"]["neck_coordinates"]
            st.session_state.type_detected = [
                neck_type for neck_type, coords in neck_coordinates.items() 
                if coords["left_point"] and coords["right_point"] and coords["rotation_angle"]
            ]
            st.session_state.neck_to_coords = neck_coordinates
        except requests.RequestException as e:
            st.error(f"Error with API request: {e}")
            st.write("Here's the response text:")
            st.write(results)
            st.stop()
        except json.JSONDecodeError:
            st.write("Error decoding JSON. Here's the response text:")
            st.write(results)  # Display raw response text
            st.error("Failed to decode JSON from the API response.")
            st.stop()
        except KeyError:
            st.write("Error: Expected data format is not present in the response.")
            st.write("Here's the response text:")
            st.write(results)  # Display raw response text
            st.error("Failed to find expected 'neck' data in the API response.")
            st.stop()

        st.image(img_path, caption="Captured Image", width=400)

        if st.session_state.type_detected:
            st.write("Neck types detected:")
            selected_type = st.selectbox("Select Neck Type", st.session_state.type_detected)
            st.session_state.type = selected_type

            if st.session_state.selected_necklace:
                face_img = np.array(Image.open(img_path).convert("RGB"))
                necklace_img = np.array(st.session_state.necklace_img)
                result_img = overlay_necklace(face_img, necklace_img)

                if result_img is not None:
                    st.image(result_img, caption="Virtual Try-On Result", width=300)
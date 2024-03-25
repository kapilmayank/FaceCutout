import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_face_cutout(image):
    # Convert PIL Image to OpenCV format (BGR)
    image_np = np.array(image)

    # Convert BGR to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Assume only one face in this example
        x, y, w, h = faces[0]

        # Create a mask for the face region
        mask = np.zeros_like(image_np)
        mask[y:y+h, x:x+w] = image_np[y:y+h, x:x+w]

        return Image.fromarray(mask)

    else:
        st.error("No face detected.")
        return None

def main():
    st.title("Face Cutout")

    input_option = st.radio("Select Input Source:", ("Upload Photo", "Camera"))

    if input_option == "Upload Photo":
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            st.write("Original Image:")
            st.image(pil_image, use_column_width=True)

            face_cutout = create_face_cutout(pil_image)

            if face_cutout is not None:
                st.write("Face Cutout:")
                st.image(face_cutout, use_column_width=True)

    elif input_option == "Camera":
        st.subheader("Camera Input:")

        # Open the camera
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()

            face_cutout = create_face_cutout(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            if face_cutout is not None:
                st.write("Face Cutout:")
                st.image(face_cutout, use_column_width=True)

        else:
            st.error("Unable to open camera.")

        # Release the camera
        cap.release()

if __name__ == "__main__":
    main()

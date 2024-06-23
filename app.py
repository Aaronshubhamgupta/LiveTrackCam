import cv2
import streamlit as st
import numpy as np
from tracker import EuclideanDistTracker  # Import your tracker class here

def main():
    st.title("Object Tracking App")

    # Create tracker object
    tracker = EuclideanDistTracker()

    # Initialize camera variable
    camera_started = False

    # Add a button to start/stop camera
    start_stop_button = st.button("Start/Stop Camera", key="start_stop_button")

    if start_stop_button:
        camera_started = True

        # Video capture from webcam (change the index if you have multiple cameras)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Failed to open webcam.")
            return

        # Object detection from Stable camera
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

        # Create an empty placeholder for the image
        frame_placeholder = st.empty()

        while camera_started:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture video.")
                break

            height, width, _ = frame.shape

            # Extract Region of interest
            roi = frame[340:720, 500:800]

            # 1. Object Detection
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for cnt in contours:
                # Calculate area and remove small elements
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append([x, y, w, h])

            # 2. Object Tracking
            boxes_ids = tracker.update(detections)
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Combine ROI back into the frame
            frame[340:720, 500:800] = roi

            # Convert frame to RGB (required by Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the displayed frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True, caption="Object Tracking")

        cap.release()

if __name__ == "__main__":
    main()

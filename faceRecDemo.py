import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import time
import argparse

# Constants
FACE_DETECTION_MODEL = "haarcascade_frontalface_default.xml"
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
EMOTION_MODEL = "emotion_model.hdf5"

# Age and gender labels
AGE_LABELS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDER_LABELS = ['Male', 'Female']
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Face Detection with Age, Gender, and Emotion Recognition')
    parser.add_argument('--max_faces', type=int, default=3, help='Maximum number of faces to detect and analyze')
    parser.add_argument('--font_scale', type=float, default=1.3, help='Scale factor for font size')
    return parser.parse_args()


def load_models():
    print("Loading models... This might take a moment.")

    # Load face detection model
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + FACE_DETECTION_MODEL)

    # Load age detection model
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

    # Load gender detection model
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    # Load emotion recognition model
    try:
        # Set memory growth to avoid consuming all GPU memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU enabled: {gpus}")
        
        emotion_model = load_model(EMOTION_MODEL)
        print("Emotion model loaded successfully")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        raise e

    print("All models loaded successfully!")
    return face_detector, age_net, gender_net, emotion_model


def preprocess_face_for_dnn(face):
    # Create a blob and preprocess for the model
    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    return blob


def detect_face_attributes(frame, face_detector, age_net, gender_net, emotion_model, max_faces=5, font_scale=1.0):
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray_frame, 1.1, 5)
    
    # Convert to list of face objects with coordinates and area
    face_list = []
    for (x, y, w, h) in faces:
        face_list.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h})
    
    # Sort faces by area (largest first)
    face_list = sorted(face_list, key=lambda x: x['area'], reverse=True)
    
    # Apply non-maximum suppression to avoid overlapping faces
    selected_faces = []
    for face in face_list:
        overlap = False
        for selected in selected_faces:
            # Calculate intersection
            x_left = max(face['x'], selected['x'])
            y_top = max(face['y'], selected['y'])
            x_right = min(face['x'] + face['w'], selected['x'] + selected['w'])
            y_bottom = min(face['y'] + face['h'], selected['y'] + selected['h'])
            
            if x_right > x_left and y_bottom > y_top:
                # Calculate overlap area
                intersection = (x_right - x_left) * (y_bottom - y_top)
                overlap_ratio = intersection / min(face['area'], selected['area'])
                
                # If overlap is significant, don't select this face
                if overlap_ratio > 0.3:  # Threshold for overlap
                    overlap = True
                    break
        
        if not overlap and len(selected_faces) < max_faces:
            selected_faces.append(face)
    
    # Convert back to list of tuples for processing
    faces = [(face['x'], face['y'], face['w'], face['h']) for face in selected_faces]

    for (x, y, w, h) in faces:
        # Draw rectangle around the face - thicker line for better visibility
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            continue

        try:
            # Gender detection
            gender_blob = preprocess_face_for_dnn(face_roi)
            gender_net.setInput(gender_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LABELS[gender_preds[0].argmax()]
            gender_confidence = gender_preds[0].max() * 100

            # Age detection
            age_blob = preprocess_face_for_dnn(face_roi)
            age_net.setInput(age_blob)
            age_preds = age_net.forward()
            age = AGE_LABELS[age_preds[0].argmax()]
            age_confidence = age_preds[0].max() * 100

            # Emotion detection
            emotion_roi = cv2.resize(face_roi, (64, 64))  # Match your model's expected input size
            emotion_roi = cv2.cvtColor(emotion_roi, cv2.COLOR_BGR2GRAY)
            emotion_roi = emotion_roi.astype("float") / 255.0
            emotion_roi = img_to_array(emotion_roi)
            emotion_roi = np.expand_dims(emotion_roi, axis=0)

            # Use TensorFlow's predict method with proper error handling
            emotion_preds = emotion_model.predict(emotion_roi, verbose=0)[0]
            emotion = EMOTION_LABELS[emotion_preds.argmax()]
            emotion_confidence = emotion_preds.max() * 100

            # Create results text - split into multiple lines for readability
            gender_text = f"Gender: {gender} ({gender_confidence:.1f}%)"
            age_text = f"Age: {age} ({age_confidence:.1f}%)"
            emotion_text = f"Emotion: {emotion} ({emotion_confidence:.1f}%)"
            
            # Enhanced text display with background for better readability
            # Draw a semi-transparent black background for text
            text_bg_height = int(85 * font_scale)
            alpha = 0.7
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y-text_bg_height), (x+w, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
            # Position text on the background
            font_thickness = max(1, int(2 * font_scale))
            cv2.putText(frame, gender_text, (x+5, y-text_bg_height+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (255, 255, 255), font_thickness)
            cv2.putText(frame, age_text, (x+5, y-text_bg_height+45), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (255, 255, 255), font_thickness)
            cv2.putText(frame, emotion_text, (x+5, y-text_bg_height+70), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (255, 255, 255), font_thickness)

        except Exception as e:
            print(f"Error processing face: {e}")
            # Just display a basic label if processing fails
            cv2.putText(frame, "Face detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (0, 255, 0), font_thickness)

    # Display count of detected faces
    face_count_text = f"Faces detected: {len(faces)}/{max_faces}"
    cv2.putText(frame, face_count_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (0, 255, 255), max(1, int(2 * font_scale)))

    return frame


def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        max_faces = args.max_faces
        font_scale = args.font_scale
        
        print(f"Setting up with max_faces={max_faces}, font_scale={font_scale}")
        
        print("Initializing...")
        face_detector, age_net, gender_net, emotion_model = load_models()

        print("Starting webcam...")
        # Start webcam
        cap = cv2.VideoCapture(0)

        # Check if webcam opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam. Check if it's properly connected.")
            return

        # Get webcam properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Webcam initialized: {frame_width}x{frame_height} @ {fps}fps")
        print("Press 'q' to quit the application")
        print("Press '+' to increase font size")
        print("Press '-' to decrease font size")

        # FPS calculation variables
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_display = 0

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to grab frame from webcam")
                break

            # Calculate FPS
            fps_frame_count += 1
            fps_current_time = time.time()
            time_diff = fps_current_time - fps_start_time

            if time_diff >= 1.0:
                fps_display = fps_frame_count / time_diff
                fps_frame_count = 0
                fps_start_time = fps_current_time

            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (0, 0, 255), max(1, int(2 * font_scale)))

            # Process frame for face detection and attribute prediction
            processed_frame = detect_face_attributes(
                frame, face_detector, age_net, gender_net, emotion_model, max_faces, font_scale)

            # Display the resulting frame
            cv2.imshow('Face Detection with Age, Gender and Emotion', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Break the loop when 'q' is pressed
            if key == ord('q'):
                break
            # Increase font size when '+' is pressed
            elif key == ord('+'):
                font_scale += 0.1
                print(f"Font scale increased to {font_scale:.1f}")
            # Decrease font size when '-' is pressed
            elif key == ord('-'):
                font_scale = max(0.5, font_scale - 0.1)
                print(f"Font scale decreased to {font_scale:.1f}")

        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
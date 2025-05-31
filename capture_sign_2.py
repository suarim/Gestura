import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time  
import requests 
import os 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def create_frame_landmarks_df(results, frame, xyz):
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def do_capture_loop(xyz):
    all_landmarks = []
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return all_landmarks

        # FPS calculation variables
        prev_frame_time = 0
        curr_frame_time = 0
        fps = 0

        # Timer variables
        start_time = time.time()

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            frame_num = 0  # Use frame_num instead of frame to avoid conflict
            while cap.isOpened():
                frame_num += 1
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Calculate FPS
                curr_frame_time = cv2.getTickCount()
                if prev_frame_time > 0:
                    fps = cv2.getTickFrequency() / (curr_frame_time - prev_frame_time)
                prev_frame_time = curr_frame_time

                # Calculate elapsed time
                elapsed_time = int(time.time() - start_time)
                minutes = elapsed_time // 60
                seconds = elapsed_time % 60
                timer_text = f"Time: {minutes:02d}:{seconds:02d}"

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                landmarks = create_frame_landmarks_df(results, frame_num, xyz)
                all_landmarks.append(landmarks)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Put FPS text on image
                cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Put Timer text on image
                cv2.putText(image, timer_text, (image.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        
        print(f"Total frames captured in video stream: {frame_num}") # Use frame_num
        cap.release()
    except Exception as e:
        print(f"Exception occurred: {e}")
        return all_landmarks
    return all_landmarks

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    print(f"Processed {n_frames} frames for inference")
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def create_frame_windows(landmarks_df, window_size=20, overlap=3):
    """
    Split landmarks data into overlapping windows.
    
    Args:
        landmarks_df: DataFrame with frame landmarks
        window_size: Number of frames per window
        overlap: Number of frames overlapping between windows
        
    Returns:
        List of numpy arrays, each representing a window of frames
    """
    # Get all frame numbers
    frames = landmarks_df['frame'].unique()
    frames.sort()
    
    windows = []
    stride = window_size - overlap
    
    for start_idx in range(0, len(frames) - window_size + 1, stride):
        window_frames = frames[start_idx:start_idx + window_size]
        window_data = landmarks_df[landmarks_df['frame'].isin(window_frames)]
        
        # Reshape the window data
        n_frames = len(window_frames)
        data_columns = ['x', 'y', 'z']
        window_array = []
        
        for frame in window_frames:
            frame_data = window_data[window_data['frame'] == frame][data_columns].values
            window_array.append(frame_data)
            
        # Make sure each window has the right shape
        if len(window_array) == window_size:
            window_array = np.array(window_array).reshape(window_size, ROWS_PER_FRAME, len(data_columns))
            windows.append(window_array.astype(np.float32))
    
    return windows

if __name__ == '__main__':
    xyz = pd.read_parquet("./static/1460359.parquet")
    landmarks = do_capture_loop(xyz)
    
    # Calculate the number of frames captured
    if landmarks:
        frame_count = max([df['frame'].max() for df in landmarks]) if landmarks else 0
        print(f"Captured {frame_count} frames")
    else:
        # Handle the case where landmarks might be empty or capture failed
        print("No landmarks captured or capture failed.")
        frame_count = 0
    
    # Combine all landmarks into a DataFrame
    if landmarks: # Only proceed if landmarks were captured
        combined_landmarks = pd.concat(landmarks).reset_index(drop=True)
        combined_landmarks.to_parquet('output.parquet')
        ROWS_PER_FRAME = 543 # Define this earlier or pass as argument if needed

        interpreter = tf.lite.Interpreter("new_model.tflite")
        interpreter.allocate_tensors()
        found_signatures = list(interpreter.get_signature_list().keys())
        prediction_fn = interpreter.get_signature_runner("serving_default")

        train = pd.read_csv("./static/train.csv")
        train['sign_ord'] = train['sign'].astype('category').cat.codes

        SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
        ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
        
        # Create windows of frames
        print("\nProcessing frames in windows of 5 frames with 3 frame overlap")
        windows = create_frame_windows(combined_landmarks, window_size=5, overlap=3)
        print(f"Created {len(windows)} windows")
        
        # Process each window
        window_predictions = []
        for i, window_data in enumerate(windows):
            # Get prediction for window
            prediction = prediction_fn(inputs=window_data)
            sign_idx = prediction['outputs'].argmax()
            # Check if sign_idx is valid before accessing ORD2SIGN
            if sign_idx not in ORD2SIGN:
                print(f"Window {i+1}: Invalid sign index {sign_idx}. Skipping.")
                continue
                
            sign_name = ORD2SIGN[sign_idx]
            confidence = prediction['outputs'].max()
            
            print(f"Window {i+1}: Processing sign '{sign_name}' with confidence {confidence:.4f}")

            # Condition 1 & 2: Check confidence threshold and NaN
            if confidence < 0.35 or np.isnan(confidence):
                print(f"Window {i+1}: Confidence {confidence:.4f} below threshold or NaN. Skipping.")
                continue

            # Condition 3: Check for consecutive duplicates
            if window_predictions and window_predictions[-1][1] == sign_name:
                 print(f"Window {i+1}: Sign '{sign_name}' is the same as the previous. Skipping.")
                 continue
            
            # If all checks pass, append the prediction
            window_predictions.append((i, sign_name, confidence))
            print(f"Window {i+1}: Added sign '{sign_name}' with confidence {confidence:.4f}")
        
        # Print overall most common prediction
        if window_predictions:
            print(window_predictions)
            sign_counts = {}
            for _, sign, _ in window_predictions:
                sign_counts[sign] = sign_counts.get(sign, 0) + 1
            
            most_common_sign = max(sign_counts.items(), key=lambda x: x[1])[0]
            print(f"\nMost common prediction: '{most_common_sign}' ({sign_counts[most_common_sign]}/{len(window_predictions)} windows)")

            # --- Ollama Sentence Generation --- 
            final_signs = [sign for _, sign, _ in window_predictions]
            print(f"\nFinal filtered signs: {final_signs}")

            if final_signs:
                OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
                OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b") # Or choose another suitable model
                
                sign_string = ", ".join(final_signs)
                # New prompt focused on simplicity and sign language context
                prompt = f"Translate the following sequence of sign language words into a simple English phrase. Use only the given words and the absolute minimum necessary connecting words (like 'on', 'in', 'at'). Signs: {sign_string}"
                
                print(f"\nSending prompt to Ollama ({OLLAMA_MODEL}): {prompt}")
                
                ollama_payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
                
                try:
                    response = requests.post(OLLAMA_URL, json=ollama_payload, timeout=60) # Added timeout
                    response.raise_for_status() # Raise an exception for bad status codes
                    
                    ollama_data = response.json()
                    generated_sentence = ollama_data.get('response', '').strip()
                    
                    print(f"\nOllama Generated Sentence: {generated_sentence}")
                    
                except requests.exceptions.RequestException as e:
                    print(f"\nError connecting to Ollama: {e}")
                    print(f"Ensure Ollama is running at {OLLAMA_URL}")
                except Exception as e:
                    print(f"\nError during Ollama request: {e}")
            else:
                 print("\nNo signs to send to Ollama.")
            # --- End Ollama Sentence Generation ---

        else:
            print("No predictions made")
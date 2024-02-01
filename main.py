import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

p_left_eye = [385, 380, 387, 373, 362, 263]
p_right_eye = [160, 144, 158, 153, 33, 133]

p_eyes = p_left_eye+p_right_eye

p_mouth = [82, 87, 13, 14, 312, 317, 78, 308]

def calculate_ear(face, p_right_eye, p_left_eye):
  '''Calculates the Eye Aspect Ratio (EAR) of a face'''

  try:
    # Extracting coordinates of the face landmarks
    face = np.array([[coord.x, coord.y] for coord in face])

    # Extracting coordinates of the left eye landmarks
    face_left = face[p_left_eye, :]

    # Extracting coordinates of the right eye landmarks
    face_right = face[p_right_eye, :]

    # Calculating Eye Aspect Ratio (EAR) for the left eye
    ear_left = (np.linalg.norm(face_left[0] - face_left[1]) + np.linalg.norm(face_left[2] - face_left[3])) / (2 * (np.linalg.norm(face_left[4] - face_left[5])))

    # Calculating Eye Aspect Ratio (EAR) for the right eye
    ear_right = (np.linalg.norm(face_right[0] - face_right[1]) + np.linalg.norm(face_right[2] - face_right[3])) / (2 * (np.linalg.norm(face_right[4] - face_right[5])))
    
  except:
    # Set EAR values to 0.0 in case of an exception
    ear_left = 0.0
    ear_right = 0.0
    
  # Calculate the average EAR for both eyes
  median_ear = (ear_left + ear_right) / 2
  
  return median_ear

def calculate_mar(face, p_mouth):
  '''Calculates the mouth aspect ratio (MAR) of a face'''
  try:
    # Extracting coordinates of the face landmarks
    face = np.array([[coord.x, coord.y] for coord in face])

    # Extracting coordinates of the mouth landmarks
    face_mouth = face[p_mouth, :]

    # Calculating Mouth Aspect Ratio (MAR)
    mar = (np.linalg.norm(face_mouth[0] - face_mouth[1]) + 
            np.linalg.norm(face_mouth[2] - face_mouth[3]) + 
            np.linalg.norm(face_mouth[4] - face_mouth[5])) / (2 * (np.linalg.norm(face_mouth[6] - face_mouth[7])))
  except:
    # Set MAR to 0.0 in case of an exception
    mar = 0.0
    
  return mar

# Set thresholds for Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
ear_threshold = 0.3
mar_threshold = 0.1

# Initialize variables for tracking eye status and blink count
sleeping = 0
blink_count = 0

# Initialize time-related variables
elapsed_time = 0
temporary_count = 0
count_list = []

# Get the initial time for blink tracking
t_blinks = time.time()

# Open the camera (assuming camera index 1, you may need to adjust)
cap = cv2.VideoCapture(1)

# Set up FaceMesh with confidence thresholds
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
  while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()

    # If the frame is empty, skip it
    if not success:
      print('Ignoring empty camera frame.')
      continue

    # Get the dimensions of the frame
    width, height, _ = frame.shape

    # Convert the frame to RGB for FaceMesh processing
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    facemesh_output = facemesh.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
      # Iterate through detected face landmarks
      for face_landmarks in facemesh_output.multi_face_landmarks:
        # Draw FaceMesh landmarks on the frame
        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
          connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1))

        # Extract face landmarks
        face = face_landmarks.landmark

        # Draw circles on eye and mouth landmarks
        for id_coord, coord_xyz in enumerate(face):
          if id_coord in p_eyes:
            coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, height, width)
            cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
          if id_coord in p_mouth:
            coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, height, width)
            cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

        # Calculate Eye Aspect Ratio (EAR) and draw information on the frame
        ear = calculate_ear(face, p_right_eye, p_left_eye)
        cv2.rectangle(frame, (0, 1), (290, 140), (58, 58, 55), -1)
        cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
          cv2.FONT_HERSHEY_DUPLEX,
          0.9, (255, 255, 255), 2)

        # Calculate Mouth Aspect Ratio (MAR) and draw information on the frame
        mar = calculate_mar(face, p_mouth)
        cv2.putText(frame, f"MAR: {round(mar, 2)} {'Open' if mar >= mar_threshold else 'Closed'}", (1, 50),
          cv2.FONT_HERSHEY_DUPLEX,
          0.9, (255, 255, 255), 2)
                
        # Track blink events based on EAR and MAR
        if ear < ear_threshold and mar < mar_threshold:
          t_initial = time.time() if sleeping == 0 else t_initial
          blink_count = blink_count + 1 if sleeping == 0 else blink_count
          sleeping = 1
        if (sleeping == 1 and ear >= ear_threshold) or (ear <= ear_threshold and mar >= mar_threshold):
            sleeping = 0
        t_final = time.time()
        elapsed_time = t_final - t_blinks

        # Update blink count per second and blink count per minute
        if elapsed_time >= (temporary_count + 1):
          temporary_count = elapsed_time
          blinks_per_sec = blink_count - count_list[-1] if count_list else blink_count
          count_list.append(blinks_per_sec)
          count_list = count_list if (len(count_list) <= 60) else count_list[-60:]

        blinks_per_min = 15 if elapsed_time <= 60 else sum(count_list)

        # Display blink-related information on the frame
        cv2.putText(frame, f"Blinks: {blink_count}", (1, 120),
          cv2.FONT_HERSHEY_DUPLEX,
          0.9, (109, 233, 219), 2)
        
        time_spent = (t_final - t_initial) if sleeping == 1 else 0.0
        cv2.putText(frame, f"Time: {round(time_spent, 3)}", (1, 80),
          cv2.FONT_HERSHEY_DUPLEX,
          0.9, (255, 255, 255), 2)
                
        # Provide a warning if signs of sleepiness are detected
        if blinks_per_min < 10 or time_spent >= 1.5:
          cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
          cv2.putText(frame, f"You might be sleepy,", (60, 420),
          cv2.FONT_HERSHEY_DUPLEX, 0.85, (58, 58, 55), 1)
          cv2.putText(frame, f"consider taking a break.", (180, 450),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85, (58, 58, 55), 1)
        
      except:
        pass

      # Display the frame with annotations
      cv2.imshow('Camera', frame)

      # Break the loop if 'c' key is pressed
      if cv2.waitKey(10) & 0xFF == ord('c'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

for face_landmarks in facemesh_output.multi_face_landmarks:
  # Extract the coordinates of all facial landmarks for the current face
  face = face_landmarks
    
  # Enumerate through each facial landmark and print its identifier (id_coord)
  # This provides the index of the landmark within the face.landmark list
  for id_coord, coord_xyz in enumerate(face.landmark):
    print(id_coord)

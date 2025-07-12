# blink-sleep-detection

## Overview

This project utilizes the MediaPipe library to implement real-time eye and mouth movement analysis through a webcam feed. The system detects facial landmarks, calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), and provides feedback on blink frequency and duration. It includes features to raise awareness about potential sleepiness based on the user's eye movements.

## Features

- **Real-time FaceMesh Processing:** Utilizes the MediaPipe FaceMesh model to detect and analyze facial landmarks.
  
- **Eye and Mouth Landmark Visualization:** Draws circles on specific eye and mouth landmarks for better visualization.
  
- **EAR and MAR Calculation:** Calculates the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) using specific sets of landmarks.
  
- **Blink Detection:** Tracks blink events based on EAR and MAR values, measuring the time elapsed since the first blink.
  
- **Blink-Related Information Display:** Displays current blink count, time spent blinking, and provides warnings if signs of sleepiness are detected.
  
- **User Interface:** The video frame, annotated with facial landmarks and relevant information, is continuously displayed.

## Setup

1. **Clone the Repository:**
   ```
   git clone https://github.com/luizvilasboas/blink-sleep-detection.git
   cd blink-sleep-detection
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```
   python main.py
   ```

4. **Usage:**
   - Press 'c' to exit the application.

## Dependencies

- [MediaPipe](https://github.com/google/mediapipe)
- [OpenCV](https://github.com/opencv/opencv)

## Contributing

If you wish to contribute to this project, feel free to open a merge request. We welcome all forms of contribution!

## License

This project is licensed under the [MIT License](https://github.com/luizvilasboas/blink-sleep-detection/blob/main/LICENSE). Refer to the LICENSE file for more details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for providing the FaceMesh model.
- [OpenCV](https://opencv.org/) for image processing and video capture.


# Blockage Detection




## Overview 

This project is designed to detect blockages in real-time using a webcam feed. The application leverages YOLOv8 for person detection and OpenCV for video processing. When a blockage is detected and persists for more than 5 seconds, the video feed freezes, displaying the detected blockage frame.
## Code Overview

#### main.py 
- Coordinates the video feed, processes frames, and handles user input.
- Integrates with Gangway_opencv for video processing and Gangway_yolo for person detection.

### Gangway_opencv
- Handles video processing, blockage detection, and frame freezing logic.
- Utilizes OpenCV for image processing.

### Gangway_yolo
- Implements person detection using the YOLOv8 model.
- Processes each frame to detect and label persons.

## Requirements

Python libraries listed in requirements.txt

`opencv-python-headless==4.5.5.64`

`numpy==1.23.5`

`ultralytics==8.0.88`

## Installation

### Prerequisites 
Ensure you have the following software and libraries installed:

- Python 3.7+
- opencv-Python
- numpy
- ultralytics
- YOLOv8 GitHub clone

### Setup

1. **Clone the Repository**

`git clone https://github.com/Vanadium03/blockage-detection.git`

`cd blockage-detection`

2. **Install dependencies**

Create a virtual environment and install the required packages:

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`



## Usage

### Running the Application

1. **Starting the Application**
 Open your terminal or comand prompt and type:

 `python main.py`

2. **Interacting with the Application**
- Blockage Detection: When a blockage is detected, the frame will freeze, and you can press Enter to resume the video feed.
- Quit: Press 'q' to exit the application at any time.

## Features

- **Real-time Webcam Feed:** Continuously captures video from the webcam.
- **Person Detection:** Uses YOLOv8 to detect persons in the frame.
- **Blockage Detection:** Identifies objects within person bounding boxes as blockages.
- **Frame Freezing:** Freezes the frame when a blockage is detected for more than 5 seconds.


## Contributing

Feel free to fork the repository and contribute! Pull requests are welcome.





## License

This project is licensed under the MIT License. See the LICENSE.md for more details.




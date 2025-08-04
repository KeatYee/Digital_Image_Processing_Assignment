import cv2
import numpy as np

# Load main video, talking video (PIP), and endscreen video
vid = cv2.VideoCapture('office.mp4')
talking_vid = cv2.VideoCapture('talking.mp4')
end_vid = cv2.VideoCapture('endscreen.mp4')

# Check if main video opened successfully
if not vid.isOpened():
    print("Error: cannot open the video")
    exit()

# Get main video properties
total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS) 
fps = fps * 0.75

# Create a video writer to save the processed video
out = cv2.VideoWriter('PART_A_Office_Video.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      fps,
                      (frame_width, frame_height))

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('face_detector.xml')
if face_cascade.empty():
    print("Error: Failed to load face detector.")
    exit()
    
# Load watermark images
wm1 = cv2.imread('watermark1.png', cv2.IMREAD_UNCHANGED)
wm2 = cv2.imread('watermark2.png', cv2.IMREAD_UNCHANGED)

# Function to overlay watermark
def overlay_watermark(frame, watermark):
    return cv2.addWeighted(frame, 1.0, watermark, 0.4, 0)

# Process each frame of the main video
for frame_count in range(total_no_frames):
    success, frame = vid.read()
    if not success:
        break
        
    # Convert frame to grayscale for brightness and face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # If average brightness is low, increase brightness (nighttime enhancement)
    if avg_brightness < 100:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=50)
        
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Detect faces and blur them
        face_region = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face
        
    ret_talking, talking_frame = talking_vid.read()
    
    if ret_talking:
        # Resize talking frame to 1/4 size of main frame (PIP)
        pip_width = frame_width // 4
        pip_height = frame_height // 4
        talking_resized = cv2.resize(talking_frame, (pip_width, pip_height))

        # Overlay talking video at top-left corner with margin
        margin_x = 50
        margin_y = 50
        x2 = frame_width - margin_x
        x1 = x2 - pip_width
        y1 = margin_y
        y2 = y1 + pip_height

        # Draws a black rectangle on the frame
        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0,0,0), thickness=6)
        
        frame[y1:y2, x1:x2] = talking_resized

    # Alternate watermarks every 3 seconds
    watermark = wm1 if (frame_count // 90) % 2 == 0 else wm2
    frame = overlay_watermark(frame, watermark)

    # Save processed frame to output video
    out.write(frame)

# After processing all main video frames, append the endscreen
if not end_vid.isOpened():
    print("Error: cannot open endscreen video")
else:
    while True:
        success, end_frame = end_vid.read()
        if not success:
            break

        out.write(end_frame)

    end_vid.release()

# Release all resources
vid.release()
talking_vid.release()
out.release()
print("video office processed")


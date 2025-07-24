import cv2
import numpy as np

vid = cv2.VideoCapture("resources/street.mp4")
talking_vid = cv2.VideoCapture("resources/talking.mp4")
end_vid = cv2.VideoCapture("resources/endscreen.mp4")

if not vid.isOpened():
    print("Error: cannot open the video")
    exit()

total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('resources/PART_A_processed_video.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      fps,
                      (frame_width, frame_height))

face_cascade = cv2.CascadeClassifier("resources/face_detector.xml")

wm1 = cv2.imread('resources/watermark1.png', cv2.IMREAD_UNCHANGED)
wm2 = cv2.imread('resources/watermark2.png', cv2.IMREAD_UNCHANGED)

def overlay_watermark(frame, watermark):
    return cv2.addWeighted(frame, 1.0, watermark, 0.4, 0)

for frame_count in range(total_no_frames):
    success, frame = vid.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness < 100:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=50) 

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        face_region = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face
        
    ret_talking, talking_frame = talking_vid.read()
    
    if ret_talking:
        pip_width = frame_width // 4
        pip_height = frame_height // 4
        talking_resized = cv2.resize(talking_frame, (pip_width, pip_height))
    
        margin_x = 50
        margin_y = 50
        x1, y1 = margin_x, margin_y
        x2, y2 = x1 + pip_width, y1 + pip_height
        
        frame[y1:y2, x1:x2] = talking_resized

    watermark = wm1 if (frame_count // 90) % 2 == 0 else wm2
    frame = overlay_watermark(frame, watermark)

    out.write(frame)

if not end_vid.isOpened():
    print("Error: cannot open endscreen video")
else:
    while True:
        success, end_frame = end_vid.read()
        if not success:
            break

        out.write(end_frame)

    end_vid.release()
    
vid.release()
talking_vid.release()
out.release()
print("video processed")


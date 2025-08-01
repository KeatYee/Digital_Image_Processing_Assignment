import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('resources/face_detector.xml')
#1.define constants and paths 
#Define a list containing all main video paths
MAIN_VIDEO_PATHS = [ 'resources/alley.mp4', 'resources/office.mp4', 'resources/singapore.mp4', 'resources/traffic.mp4']
TALKING_VIDEO_PATH = 'resources/talking.mp4'
ENDSCREEN_VIDEO_PATH = 'resources/endscreen.mp4'
WATERMARK1_PATH = 'resources/watermark1.png'
WATERMARK2_PATH = 'resources/watermark2.png'
#grayscale brightness threshold, if below this value will consider as night
BRIGHTNESS_THRESHOLD = 70
 #Gamma correction value
BRIGHTNESS_GAMMA = 0.5 
# the width of talking video
TALKING_VIDEO_OVERLAY_WIDTH_RATIO = 0.2 
# margin value, distance from upper edge and left edge
PIP_MARGIN = 100 
#the width of black border of talking video
PIP_BORDER_SIZE = 3 
#the interval of watermarks switching
WATERMARK_SWITCH_INTERVAL_SECONDS = 5 

#2.load watermarks
watermark1 = cv2.imread(WATERMARK1_PATH, cv2.IMREAD_UNCHANGED)
watermark2 = cv2.imread(WATERMARK2_PATH, cv2.IMREAD_UNCHANGED)
if watermark1 is None or watermark2 is None:
    print("Error: Could not load watermarks. Ensure they are in the correct path.")
    exit()

#3.define the video processing function
def process_video(main_video_path, output_video_path):
    print(f"\n--- Starting to process video: {main_video_path} ---")
    #initialize video readers
    cap_main = cv2.VideoCapture(main_video_path)
    cap_talking = cv2.VideoCapture(TALKING_VIDEO_PATH)
    cap_endscreen = cv2.VideoCapture(ENDSCREEN_VIDEO_PATH)
    if not cap_main.isOpened(): print(f"Error: Could not open main video at {main_video_path}"); return
    if not cap_talking.isOpened(): print(f"Error: Could not open talking video at {TALKING_VIDEO_PATH}"); return
    if not cap_endscreen.isOpened(): print(f"Error: Could not open endscreen video at {ENDSCREEN_VIDEO_PATH}"); return
    #get properties of main videos
    main_width = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
    main_height = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))
    main_fps = int(cap_main.get(cv2.CAP_PROP_FPS))
    #get talking.mp4's dimensions to calculate ratio to overlay
    talking_original_width = int(cap_talking.get(cv2.CAP_PROP_FRAME_WIDTH))
    talking_original_height = int(cap_talking.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #calculate the dimensions of the overlaid talking video
    overlay_talk_width = int(main_width * TALKING_VIDEO_OVERLAY_WIDTH_RATIO)
    overlay_talk_height = int(overlay_talk_width * talking_original_height / talking_original_width)
    #create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out = cv2.VideoWriter(output_video_path, fourcc, main_fps, (main_width, main_height))   
    #resize watermarks according to main video dimensions
    resized_watermark1 = cv2.resize(watermark1, (main_width, main_height))
    resized_watermark2 = cv2.resize(watermark2, (main_width, main_height))

    print(f"Main video resolution: {main_width}x{main_height} at {main_fps} FPS")

    #process main video frame by frame 
    frame_count = 0
    while True:
        #read main video frame
        ret_main, frame_main = cap_main.read()
        if not ret_main:
            #if finished
            break 

        #(1) night detection and brightness enhancement
        gray_frame = cv2.cvtColor(frame_main, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)
        if avg_brightness < BRIGHTNESS_THRESHOLD:
            frame_main = np.array(255 * (frame_main / 255.0)**BRIGHTNESS_GAMMA, dtype=np.uint8)

        #(2) blur faces 
        #difine the coordinates
        x_pip_offset = PIP_MARGIN
        y_pip_offset = PIP_MARGIN   
        #calculate the final dimensions including the border
        pip_total_width = overlay_talk_width + 2 * PIP_BORDER_SIZE
        pip_total_height = overlay_talk_height + 2 * PIP_BORDER_SIZE  
        x_pip_end = min(x_pip_offset + pip_total_width, main_width)
        y_pip_end = min(y_pip_offset + pip_total_height, main_height)
        #detect the faces
        gray_for_face = cv2.cvtColor(frame_main, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_for_face, 1.3, 5)

        for (x, y, w, h) in faces:
            #extract the region of faces
            face_roi = frame_main[y:y+h, x:x+w]
            #based on face size, caluculate the Guassian blur kernel size
            blur_ksize = (max(3, w//3*2+1), max(3, h//3*2+1))
            #apply Guassian blur
            blurred_face = cv2.GaussianBlur(face_roi, blur_ksize, 0)
            frame_main[y:y+h, x:x+w] = blurred_face

        #(3) overlay the talking video
        #read frame
        ret_talk, frame_talk = cap_talking.read()
        if ret_talk:
            #resize the frame 
            resized_talk = cv2.resize(frame_talk, (overlay_talk_width, overlay_talk_height))
            #add black border to the fixed frame
            bordered_talk = cv2.copyMakeBorder(resized_talk, PIP_BORDER_SIZE, PIP_BORDER_SIZE, PIP_BORDER_SIZE, PIP_BORDER_SIZE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            #calculate actual space availbale for overlay
            actual_overlay_width = x_pip_end - x_pip_offset
            actual_overlay_height = y_pip_end - y_pip_offset
            #crop the bordered frame and overlay
            if actual_overlay_height > 0 and actual_overlay_width > 0:
                final_pip_content = bordered_talk[0:actual_overlay_height, 0:actual_overlay_width]
                frame_main[y_pip_offset:y_pip_end, x_pip_offset:x_pip_end] = final_pip_content
        else:
            #if talking video ends, restart
            cap_talking.set(cv2.CAP_PROP_POS_FRAMES, 0)

        #(4) add watermarks
        #calculate current time in seconds
        current_time_seconds = frame_count / main_fps
        #switch two watermarks in a defiend interval
        if (int(current_time_seconds // WATERMARK_SWITCH_INTERVAL_SECONDS)) % 2 == 0:
            current_watermark = resized_watermark1
        else:
            current_watermark = resized_watermark2
        #check if the watermarks images has alpha channel or not
        if current_watermark.shape[2] == 4:
            #seperate the alpha channel
            wm_bgr = current_watermark[:, :, :3]
            wm_alpha = current_watermark[:, :, 3] / 255.0
            #convert into 3 channel alpha mask
            wm_alpha = np.stack([wm_alpha]*3, axis=2)
            #blend watermark with main frame
            frame_main = (frame_main * (1 - wm_alpha) + wm_bgr * wm_alpha).astype(np.uint8)
        else:
            #blend by fixed apacity
            frame_main = cv2.addWeighted(frame_main, 1.0, current_watermark, 0.4, 0)
        #write frame into output video and change to next
        out.write(frame_main)
        frame_count += 1

    #append endscreen video
    print("Appending end screen video...")
    while True:
        ret_endscreen, frame_endscreen = cap_endscreen.read()
        if not ret_endscreen:
            break
        #resize the endscreen video to match main video and write it into output
        resized_endscreen_frame = cv2.resize(frame_endscreen, (main_width, main_height))
        out.write(resized_endscreen_frame)

    #release resources
    cap_main.release()
    cap_talking.release()
    cap_endscreen.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_video_path}")
    
#main program,address all videos
if __name__ == "__main__":
    for main_video_path in MAIN_VIDEO_PATHS:
        #extract filename
        dot_index = main_video_path.rfind('.')
        if dot_index != -1:
            filename = main_video_path[:dot_index]
        else:
            filename = main_video_path
        #generate output filename
        output_video_path = f"{filename}_processed.avi"
        #process current video
        process_video(main_video_path, output_video_path)
    cv2.destroyAllWindows()
    print("\nAll videos processed.")

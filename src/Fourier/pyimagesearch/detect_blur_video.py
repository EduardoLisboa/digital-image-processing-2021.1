# import the necessary packages
from imutils.video import VideoStream
from pyimagesearch.blur_detector import detect_blur_fft
import imutils
import time
import cv2

# Initialize the video stream and allow the camera sensor to warm up
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Convert the frame to grayscale and detect blur in it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean, blurry = detect_blur_fft(gray, size=60, thresh=10, vis=False)

    # Draw on the frame, indicating whether or not it is blurry
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = f'Blurry ({mean:.4f})' if blurry else f'Not Blurry ({mean:.4f})'
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

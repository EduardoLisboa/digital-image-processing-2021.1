# import the necessary packages
from pyimagesearch.blur_detector import detect_blur_fft
import numpy as np
import imutils
import cv2

thresh = 10
vis = False
test = False

orig = cv2.imread('images/Mulan1.jpg')
orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# apply our blur detector using the FFT
mean, blurry = detect_blur_fft(gray, size=60, thresh=thresh, vis=vis)

# draw on the image, indicating whether or not it is blurry
image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = f'Blurry ({mean:.4f})' if blurry else f'Not Blurry ({mean:.4f})'

cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
print(f'[INFO] {text}')
# show the output image
cv2.imshow('Output', image)
cv2.waitKey(0)

# Check to see if are going to test our FFT blurriness detector using various sizes of a Gaussian kernel
if test:
	# Loop over various blur radii
	for radius in range(1, 30, 2):
		# Clone the original grayscale image
		image = gray.copy()
		# Check to see if the kernel radius is greater than zero
		if radius > 0:
			# Blur the input image by the supplied radius using a Gaussian kernel
			image = cv2.GaussianBlur(image, (radius, radius), 0)

			# Apply our blur detector using the FFT
			mean, blurry = detect_blur_fft(gray, size=60, thresh=thresh, vis=vis)

			# Draw on the image, indicating whether or not it is blurry
			# draw on the image, indicating whether or not it is blurry
			image = np.dstack([gray] * 3)
			color = (0, 0, 255) if blurry else (0, 255, 0)
			text = f'Blurry ({mean:.4f})' if blurry else f'Not Blurry ({mean:.4f})'

			cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
			print(f'[INFO] Kernel: {radius}, Result: {text}')
		# show the image
		cv2.imshow("Test Image", image)
		cv2.waitKey(0)

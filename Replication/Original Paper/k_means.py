import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class KMEANS:
    def __init__(self, *filename, **image):
        image_path = image["image"]
        bgr_image = cv.imread(image_path)
        self.remove_background(bgr_image)

    def remove_background(self, leaf_image):

        # Gaussian blur image to remove noise
        blured = cv.GaussianBlur(leaf_image, (1, 1), 0)

        # Convert blured Image from BGR to HSV
        hsv_leaf = cv.cvtColor(blured, cv.COLOR_BGR2HSV)

        SV_channel = hsv_leaf.copy()

        SV_channel[:, :, 0] = np.zeros(
            (SV_channel.shape[0], SV_channel.shape[1]))  # Set the 'H' channel to Zero

        SV_channel[:, :, 2] = np.zeros(
            (SV_channel.shape[0], SV_channel.shape[1]))
        # Create a binary mask from the SV Channel

        mask = cv.inRange(SV_channel, (0, 0, 0), (0, 85, 0))
        # Invert mask, White areas represent green components and black the background
        mask = cv.bitwise_not(mask)

        # perform bitwise_and between mask and hsv image
        background_extracted = cv.bitwise_and(hsv_leaf, hsv_leaf, mask=mask)

        self.segment_diseased(background_extracted)

    def segment_diseased(self, bg_extracted_hsv):
        bg_extracted_rgb = cv.cvtColor(bg_extracted_hsv, cv.COLOR_HSV2RGB)
        ycrcb = cv.cvtColor(bg_extracted_rgb, cv.COLOR_RGB2YCrCb)

        r = bg_extracted_rgb[:, :, 0]
        g = bg_extracted_rgb[:, :, 1]
        b = bg_extracted_rgb[:, :, 2]

        #ycbcr_image = cv.cvtColor(bg_extracted_rgb, cv.COLOR_RGB2YCrCb)
        #ycbcr_image[:, :, 0] = np.zeros_like(ycbcr_image[:, :, 0])

        #cr = 128 + (112 * r - 93.786 * g - 18.214 * b) / 256

        cr = ycrcb[:, :, 1]
        z = cr.reshape((cr.shape[0] * cr.shape[1]))
        print(cr.shape)
        print(z.shape)
        # convert to np.float32
        z = np.float32(z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1)
        K = 2
        ret, label, center = cv.kmeans(
            z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((cr.shape))

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(res2, cmap="gray")
        axs[1].imshow(cr, cmap="gray")
        plt.show()

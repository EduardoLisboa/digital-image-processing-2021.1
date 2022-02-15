import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


class KMEANS:
    image_name: str
    original_rgb_image: np.ndarray
    leaf_area_value: int
    disease_area_value: int

    leaf_x: int
    leaf_y: int

    leaf_contour: np.ndarray
    leaf_filled_contour: np.ndarray
    disease_contour: np.ndarray
    disease_filled_contour: np.ndarray

    save_contour_plot: bool
    save_comparison_plot: bool

    def __init__(self, image, img_name, save_comparison_plot=False, save_contour_plot=False):
        """
        Constructs all the necessary attributes for the KMEANS object.

        :param image: (str) Full path of an image file
        :param img_name: (str) Name of the image file
        """
        KMEANS.image_name = img_name.split('.')[0]
        image_path = image
        bgr_image = cv2.imread(image_path)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        KMEANS.original_rgb_image = rgb_image

        KMEANS.save_comparison_plot = save_comparison_plot
        KMEANS.save_contour_plot = save_contour_plot

        self.remove_background(bgr_image)

    @staticmethod
    def remove_background(leaf_image):
        """
        Removes the background of an image based on its HSV values.

        :param leaf_image: An image file to have its background extracted
        :return: None
        """
        # Gaussian blur image to remove noise
        blured = cv2.GaussianBlur(leaf_image, (1, 1), 0)

        # rgb_image = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2RGB)

        # Convert blured Image from BGR to HSV
        hsv_leaf = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

        SV_channel = hsv_leaf.copy()

        SV_channel[:, :, 0] = np.zeros((SV_channel.shape[0], SV_channel.shape[1]))  # Set the 'H' channel to zero
        SV_channel[:, :, 2] = np.zeros((SV_channel.shape[0], SV_channel.shape[1]))  # Set the 'V' channel to zero

        # Create a binary mask from the SV Channel
        mask = cv2.inRange(SV_channel, (0, 0, 0), (0, 65, 0))

        # Invert mask, White areas represent green components and black the background
        mask = cv2.bitwise_not(mask)

        _, threshold_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        normalized_mask = np.zeros((threshold_mask.shape[0], threshold_mask.shape[1]))
        normalized_mask = cv2.normalize(threshold_mask, normalized_mask, 0, 1, cv2.NORM_MINMAX)
        KMEANS.leaf_area_value = np.sum(normalized_mask)

        cnts = cv2.findContours(threshold_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        KMEANS.leaf_contour = KMEANS.original_rgb_image.copy()
        KMEANS.leaf_filled_contour = KMEANS.original_rgb_image.copy()

        cv2.drawContours(KMEANS.leaf_contour, [c], -1, (255, 0, 0), thickness=5)
        cv2.drawContours(KMEANS.leaf_filled_contour, [c], -1, (255, 0, 0), thickness=cv2.FILLED)

        boxes = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        boxes = np.asarray(boxes)
        left, top = np.min(boxes, axis=0)[:2]
        right, bottom = np.max(boxes, axis=0)[2:]

        KMEANS.leaf_y = bottom - top
        KMEANS.leaf_x = right - left

        # perform bitwise_and between mask and hsv image
        background_extracted = cv2.bitwise_and(hsv_leaf, hsv_leaf, mask=mask)

        KMEANS.segment_diseased(background_extracted)

    @staticmethod
    def segment_diseased(bg_extracted_hsv):
        """
        Segments an image to show the segmented part as white and the rest as black

        :param bg_extracted_hsv: An image file to be segmented
        :return: None
        """
        bg_extracted_rgb = cv2.cvtColor(bg_extracted_hsv, cv2.COLOR_HSV2RGB)
        ycrcb = cv2.cvtColor(bg_extracted_rgb, cv2.COLOR_RGB2YCrCb)

        cr = ycrcb[:, :, 1]
        z = cr.reshape((cr.shape[0] * cr.shape[1]))

        # convert to np.float32
        z = np.float32(z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1)
        K = 2
        ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()].reshape(cr.shape)

        _, threshold_res = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY)
        normalized_res = np.zeros((threshold_res.shape[0], threshold_res.shape[1]))
        normalized_res = cv2.normalize(res, normalized_res, 0, 1, cv2.NORM_MINMAX)
        KMEANS.disease_area_value = np.sum(normalized_res)

        cnts = cv2.findContours(threshold_res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key=cv2.contourArea)

        KMEANS.disease_contour = KMEANS.original_rgb_image.copy()
        KMEANS.disease_filled_contour = KMEANS.original_rgb_image.copy()

        cv2.drawContours(KMEANS.disease_contour, cnts, -1, (255, 0, 0), thickness=5)
        cv2.drawContours(KMEANS.disease_filled_contour, cnts, -1, (255, 0, 0), thickness=cv2.FILLED)

        KMEANS.plot_images(cr, res)

    @staticmethod
    def plot_images(processed, disease_area):
        """
        Plots three images to the screen and saves them in a PNG file in a folder called 'Plots', which must exist prior
        to the code execution. The original image plotted comes from the contructor, that stores it in a class variable
        called 'original_rgb_image'.

        :param processed: An image file to be ploted
        :param disease_area: An image file to be ploted
        :return: None
        """

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # fig, axs = plt.subplots(1, 3)
        axs[0].set_title('Original Image', fontsize=25)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].imshow(KMEANS.original_rgb_image, cmap="gray")

        axs[1].set_title('Processed Image', fontsize=25)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].imshow(processed, cmap="gray")

        axs[2].set_title('Diseased Area', fontsize=25)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].imshow(disease_area, cmap="gray")

        if KMEANS.save_comparison_plot:
            plt.savefig(f'Plots/{KMEANS.image_name} - Comparison.png')
            print('\nComparison plot saved in "Plots" folder.')
        else:
            print('\nComparison plot not saved, as requested.')

        plt.show()

        KMEANS.leaf_information()

    @staticmethod
    def leaf_information():
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        fig.suptitle('Contour Areas', fontsize=25)

        axs[0][0].set_title('Outlined', fontsize=20)
        axs[0][0].set_xticks([])
        axs[0][0].set_yticks([])
        axs[0][0].imshow(KMEANS.leaf_contour)

        axs[0][1].set_title('Filled', fontsize=20)
        axs[0][1].set_xticks([])
        axs[0][1].set_yticks([])
        axs[0][1].imshow(KMEANS.leaf_filled_contour)

        axs[1][0].set_title('Outlined', fontsize=20)
        axs[1][0].set_xticks([])
        axs[1][0].set_yticks([])
        axs[1][0].imshow(KMEANS.disease_contour)

        axs[1][1].set_title('Filled', fontsize=20)
        axs[1][1].set_xticks([])
        axs[1][1].set_yticks([])
        axs[1][1].imshow(KMEANS.disease_filled_contour)

        if KMEANS.save_contour_plot:
            plt.savefig(f'Contours Plots/{KMEANS.image_name} - Contours.png')
            print('\nContours plot saved in "Contours Plots" folder.')
        else:
            print('\nContour plot not saved, as requested.')

        plt.show()

        print(f'\nIMAGE {KMEANS.image_name}')
        print(f'Image shape:            {KMEANS.original_rgb_image.shape[0]} x {KMEANS.original_rgb_image.shape[1]} px')
        print(f'Approximate leaf shape: {KMEANS.leaf_y} x {KMEANS.leaf_x} px')
        print(f'Total image area:       {KMEANS.original_rgb_image.shape[0] * KMEANS.original_rgb_image.shape[1]} px^2')
        print(f'Total leaf area:        {KMEANS.leaf_area_value} px^2')
        print(f'Diseased area:          {KMEANS.disease_area_value} px^2')
        percentage = (KMEANS.disease_area_value * 100) / KMEANS.leaf_area_value
        print(f'Diseased percentage:    {percentage:.2f}%')

        print('\nSaving processing information in "Information.txt" file.')
        with open('Information.txt', 'a') as file:
            file.write(f'IMAGE {KMEANS.image_name}\n')
            file.write(f'Image shape:            {KMEANS.original_rgb_image.shape[0]} x {KMEANS.original_rgb_image.shape[1]} px\n')
            file.write(f'Approximate leaf shape: {KMEANS.leaf_y} x {KMEANS.leaf_x} px\n')
            file.write(f'Total leaf area:        {KMEANS.leaf_area_value} px^2\n')
            file.write(f'Diseased area:          {KMEANS.disease_area_value} px^2\n')
            file.write(f'Diseased percentage:    {percentage:.2f}%\n\n')

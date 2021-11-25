import matplotlib.pyplot as plt
import numpy as np


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # Get dimensions of the image and get center coordinates
    h, w = image.shape
    cx, cy = int(w / 2.0), int(h / 2.0)

    # Compute the FFT to find the frequency transform, then shift the zero frequency component (i.e., DC component
    # located at the top-left corner) to the center where it will be easier to analyze
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fft_shift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # Zero-out the center of the FFT shift (i.e., remove low frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply the inverse FFT

    fft_shift[cy - size:cy + size, cx - size:cx + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    # Compute the magnitude spectrum of the reconstructed image, then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # The image will be considered "blurry" if the mean value of the magnitudes is less than the threshold value
    return mean, mean <= thresh

from ycbcr import YCBCR
from k_means import KMEANS
import os


class Main:
    # Path to saved model
    def getImagePath(self, image_path, filename):
        # get path of image file
        return os.path.join(image_path, filename)

    def getModelPath(self):
        # get path of model
        return ("hi")

    def classifyImage(self, imagepath, filename):
        # Start Image Segmentation
        # YCBCR(filename, image=self.getImagePath(
        #    imagepath, filename))
        KMEANS(filename, image=self.getImagePath(
            imagepath, filename))


if __name__ == "__main__":
    image_path = '/home/frexg/Downloads/lara2018-master/segmentation/dataset/images/test'
    c = Main()
    if os.path.exists(image_path):
        for dirpath, dirname, filenames in os.walk(image_path):
            for ImageFile in filenames:
                c.classifyImage(image_path, ImageFile)
    else:
        print("Incorrent path")

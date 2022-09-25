import os
import sys  # for commandline arguments
import cv2
import random  # random color generator
import logging
import coloredlogs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Setting up font for matplotlib
mpl.rc("font", family=["Josefin Sans", "Trebuchet MS", "Inconsolata"], weight="medium")

# Setting up logger for the project
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def randcolor():
    '''
        Generate a random color, as list
        '''
    return [random.randint(0, 256) for i in range(3)]


def getImg(imgName):
    '''
        Load image by name (relative or absolute path)
        Convert BGR to RGB/Grayscale
        Get edge using Canny
        '''
    image = cv2.imread(imgName)
    igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    irgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    iedge = cv2.Canny(igray, 100, 200)  # using Canny to get edge
    return image, irgb, igray, iedge


def getFeatures(edge):
    '''
        Detect contours in an image
        Get minimum area box / best fit ellilpse of contours
        '''
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = [None]*len(contours)  # [None, None, None, ...]
    
    # All detected contour will have a min area box
    # But only those with more than 5 points will have a ellipse
    for index, contour in enumerate(contours):
        if contour.shape[0] > 5:
            ellipses[index] = cv2.fitEllipse(contour)
    return contours,ellipses


def render(shape, contours, ellipses):
    '''
        Render the contours, rectangles and ellipses to an image
        for later display
        '''
    # Specifying the data type so that matplotlib won't treat this as a floating point image
    # which will be truncated to [0, 1] (we're uint8 in [0, 255])
    draw = np.zeros(shape, dtype='uint8')
    log.info(f"Drawing on shape: {shape} with type {draw.dtype}")
    for index, contour in enumerate(contours):
        color = randcolor()
        log.debug(f"Getting random color: {color}")
        cv2.drawContours(draw, contours, index, color, 1, cv2.LINE_AA)
        
        # All detected contour will have a min area box
        # But only those with more than 5 points will have a ellipse
        if contour.shape[0] > 5:
            cv2.ellipse(draw, ellipses[index], color, 3, cv2.LINE_AA)

return draw


def main():
    if len(sys.argv)>1:
        imgName = sys.argv[1]
    else:
        imgName = "./resources/23.jpg"
    # Or he/she might mistype the file name
    if not os.path.exists(imgName):
        log.error(f"Cannot find the file specified: {imgName}")
        return 1
    
    img, rgb, gray, edge = getImg(imgName)
    contours, ellipses = getFeatures(edge)
    drawing = render(img.shape, contours, ellipses)

# Display all four images in the same figure window
plt.figure("Fitting")
    plt.suptitle("Fitting ellipses and finding min-area rectangles", fontweight="bold")
    plt.subplot(221)
    plt.title("original", fontweight="bold")
    plt.imshow(rgb)
    plt.subplot(222)
    plt.title("grayscale", fontweight="bold")
    plt.imshow(gray, cmap="gray")
    plt.subplot(223)
    plt.title("canny Edge", fontweight="bold")
    plt.imshow(edge, cmap="gray")
    plt.subplot(224)
    plt.title("contour & ellipses ", fontweight="bold")
    plt.imshow(drawing)
    plt.show()
    
    # Display the detected contours rectangles and ellipses separatedly for examination
    plt.figure("contour & ellipses ")
    plt.suptitle("contour & ellipses", fontweight="bold")
    plt.imshow(drawing)
    plt.show()


# with the #!python at the beginning of this file
# you can ommit the python when executing this program
if __name__ == "__main__":
    main()

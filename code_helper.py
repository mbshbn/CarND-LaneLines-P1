# -*- coding: utf-8 -*-
"""
Created on Fri May  1 08:55:04 2020

@author: mbshb

cv2.inRange() for color selection
cv2.fillPoly() for regions selection
cv2.line() to draw lines on an image given endpoints
cv2.addWeighted() to coadd / overlay two images
cv2.cvtColor() to grayscale or change color
cv2.imwrite() to output images to file
cv2.bitwise_and() to apply a mask to an image
"""

import sys
sys.modules[__name__].__dict__.clear()

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    xp =[]
    yp =[]
    xn =[]
    yn =[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [0,255,0], thickness)
            if ((y2-y1)/(x2-x1)) > 0:
                xp += [x1, x2]
                yp += [y1, y2]
            else:
                xn += [x1, x2]
                yn += [y1, y2]

    pxp, pyp, cxp, cyp = extrapolate(xp,yp)
    pxn, pyn, cxn, cyn = extrapolate(xn,yn)

    #plt.plot((pxp, cxp), (pyp, cyp), 'r')
    #plt.plot((pxn, cxn), (pyn, cyn), 'g')
    #plt.show()
    cv2.line(img, (pxp, pyp), (cxp, cyp), color, thickness)
    cv2.line(img, (pxn, pyn), (cxn, cyn), color, thickness)
    #print(temp_l,temp_p)

def extrapolate(xp,yp):
    zp = np.polyfit(xp, yp, 1)
    fp = np.poly1d(zp)
    #for i in range(min(xp), max(xp)):
    #    plt.plot(i, fp(i), 'go')
    #plt.show()
    x_newp = np.linspace(min(xp), max(xp), 10).astype(int)
    y_newp = fp(x_newp).astype(int)
    points_newp = list(zip(x_newp, y_newp))
    pxp, pyp = points_newp[0]
    cxp, cyp = points_newp[-1]
    return pxp, pyp, cxp, cyp

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def Lane_Finding_Pipeline_image(image):
    # TODO: Build your pipeline that will draw lane lines on the test_images
    gray = grayscale(image)
    #plt.imshow(gray, cmap='gray')
    #plt.show()

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    #plt.imshow(blur_gray, cmap='gray')
    #plt.show()

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = image.shape
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[(0,ysize),(2.4*xsize/5, 1.19*ysize/2), (2.6*xsize/5, 1.19*ysize/2), (xsize,ysize)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Display the image and show region and color selections
    # plt.imshow(image)
    #x = [0,2.4*xsize/5,2.6*xsize/5,xsize ]
    #y = [ysize, 1.16*ysize/2, 1.16*ysize/2,ysize]
    x_region = [vertices[0,0,0],vertices[0,1,0],vertices[0,2,0],vertices[0,3,0] ]
    y_region = [vertices[0,0,1],vertices[0,1,1],vertices[0,2,1],vertices[0,3,1] ]
    # plt.plot(x_region, y_region, 'b--', lw=2)
    # plt.show()

    rho = 1 # 2 #distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1 #15#    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #40 #minimum number of pixels making up a line
    max_line_gap = 20    # 20 #maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    lines_edges = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
    # plt.imshow(lines_edges)
    # plt.plot(x_region, y_region, 'b--', lw=2)
    # plt.show()
    return lines_edges

import os
os.listdir("test_images/")

#reading in an image
image_path = "test_images/solidWhiteCurve.jpg"
image_path = 'test_images/solidWhiteRight.jpg'
image_path = 'test_images/solidYellowCurve.jpg'
image_path = 'test_images/solidYellowCurve2.jpg'
image_path = 'test_images/solidYellowLeft.jpg'
image_path = 'test_images/whiteCarLaneSwitch.jpg'
image = mpimg.imread(image_path)

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
# plt.show()
lines_edges = Lane_Finding_Pipeline_image(image)
# then save them to the test_images_output directory.
mpimg.imsave("test_images_output/output.png", lines_edges)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # TODO: Build your pipeline that will draw lane lines on the test_images
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = image.shape
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[(0,ysize),(2.4*xsize/5, 1.19*ysize/2), (2.6*xsize/5, 1.19*ysize/2), (xsize,ysize)]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    rho = 1 # 2 #distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1 #15#    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #40 #minimum number of pixels making up a line
    max_line_gap = 20    # 20 #maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Create a "color" binary image to combine with line image
    result = weighted_img(line_image, image, α=0.8, β=1., γ=0.)

    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
#clip1 = VideoFileClip("test_videos/challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)

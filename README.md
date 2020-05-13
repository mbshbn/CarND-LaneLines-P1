# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* The output should be two solid lines, one for right side, and the other for left side


[//]: # (Image References)

[image1]: ./test_images_output/output.png "Grayscale"
[image2]: ./examples/shortcoming.jpg "shortcoming"
[image3]: ./examples/shortcoming2.jpg "shortcoming"

---

### Pipeline (single images)

My pipeline consisted of the follwoing steps. 

1. Conversion to grayscale using `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
1. Gaussian smoothing uisng `cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)`
1. Edge Detection using Canny algorithm, using `cv2.Canny(img, low_threshold, high_threshold)`
1. Defining the region of interest. This is the area in front of the fixed camera that the lanes appear. I defined an array of four points as the verticies of the polygon, and fill the pixels inside the polygon with a color `cv2.fillPoly`. Then the function returns the image only where the mask pixels are nonzero using `cv2.bitwise_and`.
1. Finding line segments in the binary image using the probabilistic Hough transform using `cv2.HoughLinesP`. The inputs to this function are the distance and anular resolution in pixels of the Hough grid. Also, the minimum number of votes (intersections in Hough grid cell), the minimum number of pixels making up a line, and the maximum gap in pixels between connectable line segments. 
      1. mapping out the full extent of the lane to visualize the result, I defined a function called `draw_lines`. To draw a single line on the left and right lanes, it extrapolates the lines uisng `np.polyfit` and `np.poly1d`. The left and right lines are distinguished using their slope. Usually, the slope is about 0.6 or -0.6. Having this number, to avoid small white and yellow marks on the ground affecting the lines, those who have a slope very different than these usual slopes are ignored. Although this is applied to filter the lines before extrapolating, sometimes the extrapolated line may have a slope very different than the usual slope. To avoid reporting wrong lines, the lines after extrapolation are filtered, and those that do not have a usual slope are ignored.  
1. Combining the oiginal image and the output of the prevoius step using `cv2.addWeighted`.


Here is the final result:

<p align="center">  <img width="460" height="300" src="./test_images_output/output.png "Grayscale""></p>

### 2. Potential shortcomings with the current pipeline


One potential shortcoming would happen when the line's curvature is large. This is overcame in another project called [Advanced Lane Line detection](https://github.com/mbshbn/CarND-Advanced-Lane-Lines) in my github repo.

Another shortcoming could be missled with different signs on the street that are ignored. For example the following lines:

a             |  b
:-------------------------:|:-------------------------:
![alt text][image2]  |  ![alt text][image3] 


### 3. Possible improvements to the pipeline

A possible improvement would be to considering higher-order polynomial lines, not only straight lines.

Another potential improvement could be to considering different local signs on the ground to avoid being missled with them.

# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* The output should be two solid lines, one for right side, and the other for left side


[//]: # (Image References)

[image1]: ./test_images_output/output.png "Grayscale"
[image2]: ./examples/shortcoming.jpg "shortcoming"
[image3]: ./examples/shortcoming2.jpg "shortcoming"

---

### Reflection

### 1. Describe your pipeline. 

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied a Gaussian smoothing. Next, the Canny function is applied. Then, a polygon area with four vertices is selected, and then the Hough lines are drawn. In the end, the image is combined with the output with Hough lines.

To draw a single line on the left and right lanes, I modified the draw_lines() function by extrapolating lines. The left and right lines are distinguished using their slope. usually, the slope is about 0.6 or -0.6. Having this number, to avoid small white and yellow marks on the ground affecting the lines, those who have slope very different than these normal slopes are ignored. Although this is applied to filter lines before extrapolating, sometimes the extrapolated line may have a slope very different than the usual range. to avoid reporting wrong lines, the lines after extrapolation are filtered, and those that do not have a usual slope are ignored.  

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the line's curvature is large.

Another shortcoming could be misleading with different signs on the street that are ignored. For example the following lines:

![alt text][image2]
![alt text][image3]

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to considering higher-order polynomial lines, not only straight lines.

Another potential improvement could be to considering different local signs on the ground to avoid misleading with them.

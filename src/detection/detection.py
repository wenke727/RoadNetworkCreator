#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
from cv2 import cv2

##
# @Author David Awad
# Detection.py, traces and identifies lane
# markings in an image or .mp4 video
# usage: detection.py [-h] [-f FILE] [-v VIDEO]

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)


def region_of_interest(img, vertices):
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


def draw_lines(image, img, lines, color=[255, 0, 0], thickness=8):   #image 原始输入图片
    # reshape lines to a 2d matrix
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # create array of slopes
    slopes = (lines[:,3] - lines[:,1]) / (lines[:,2] - lines[:,0])
    index  = ~np.isnan(slopes) & ~np.isinf(slopes)
    slopes = slopes[index]
    lines = lines[index, :]

    # Curve fitting approach
    left_lines = lines[np.where((slopes<-1.0/np.sqrt(3)) & (slopes>-np.sqrt(3)))]
    if left_lines.size is not 0:

        x_axes = np.append(left_lines[:, 0], left_lines[:, 2])
        y_axes = np.append(left_lines[:, 1], left_lines[:, 3])
        #print('left_lines x_axes : ', x_axes)
        left_curve = np.poly1d(np.polyfit(y_axes, x_axes, 1))   # 多项式
        min_left_x, min_left_y = x_axes.min(), y_axes.min()

    right_lines = lines[np.where((slopes>1.0/np.sqrt(3)) & (slopes<np.sqrt(3)))]
    if right_lines.size is not 0:
        #print('right_lines shape: ', right_lines.dtype, right_lines.ndim)
        x_axes = np.append(right_lines[:, 0], right_lines[:, 2])
        y_axes = np.append(right_lines[:, 1], right_lines[:, 3])
        right_curve = np.poly1d(np.polyfit(y_axes, x_axes, 1))
        min_right_x, min_right_y = x_axes.min(), y_axes.min()

    # 'skyline'
    try:
        min_y = min(min_left_y, min_right_y)        #(x)
    except:
        min_y = img.shape[1] // 2

    #颜色格式转换，为了进行颜色判断
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)                      #找出白线用的图片
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)   #找出黄线

    #计算变量
    left_hsv_pixel_count=0
    left_gray_pixel_count=0
    right_hsv_pixel_count = 0
    right_gray_pixel_count = 0

    # use new curve function f(y) to calculate x values
    if left_lines.size is not 0:
        # l1点坐标 (x, y)中，x代表图片的列，y代表图片的行
        l1 = (int(left_curve(min_y)), min_y)   #(y1, x1)  min_y是对应行 第一项是求出的列
        l2 = (int(left_curve(img.shape[0])), img.shape[0])  #(y2, x2)
        print('Left points l1 and l2,', l1, l2)
        left_pixel_count = (img.shape[0]-min_y)  #计算这条线有多少个点
        print('left_pixel_count: ', left_pixel_count)
        #middle_left = (int((min_y+img.shape[0])/2), int((int(left_curve(min_y))+int(left_curve(img.shape[0])))/2)) #((x1+x2)/2,(y1+y2)/2)
        for x in range(min_y, img.shape[0]):
            left_hsv_pixel = img_hsv[(x, int(left_curve(x)))]   #n*3
            left_gray_pixel = gray[(x, int(left_curve(x)))]
            if (left_hsv_pixel[0] >= 10 and left_hsv_pixel[0] <= 29) and \
                    (left_hsv_pixel[1] >= 43 and left_hsv_pixel[1] <= 255) and \
                        (left_hsv_pixel[2] >= 46 and left_hsv_pixel[2] <= 255):
                #print('left_hsv_pixel:  ', left_hsv_pixel[0], left_hsv_pixel[1], left_hsv_pixel[2])
                left_hsv_pixel_count += 1
                #cv2.line(img, l1, l2, [255, 255, 0], thickness)
            if left_gray_pixel >= 200 and left_gray_pixel <= 255:
                #print('left_gray_pixel: ', left_gray_pixel)
                #cv2.line(img, l1, l2, [255, 255, 255], thickness)
                left_gray_pixel_count += 1
        print('left_hsv_pixel_count: ', left_hsv_pixel_count)
        print('left_gray_pixel_count: ', left_gray_pixel_count)
        # left_hsv_pixel_count > 20表示虚黄线可能黄色像素点比较少，白色像素点因为其他因素影响高于黄色，导致黄虚线误判成白线，这里写成只要存在黄色像素就是黄色
        # left_gray_pixel_count+left_hsv_pixel_count表示虚黄线判断：黄线个数很少，白线个数存在误判
        if left_hsv_pixel_count > left_gray_pixel_count or left_hsv_pixel_count > 20:
            cv2.line(img, l1, l2, [255, 255, 0], thickness)     #代表黄色
        else:
            cv2.line(img, l1, l2, [255, 255, 255], thickness)   #代表白色


    if right_lines.size is not 0:
        r1 = (int(right_curve(min_y)), min_y)
        r2 = (int(right_curve(img.shape[0])), img.shape[0])
        print('Right points r1 and r2,', r1, r2)
        #middle_right = (int((min_y+img.shape[0])/2), int((int(right_curve(min_y))+int(right_curve(img.shape[0])))/2))
        #print('middle_right: ', middle_right)
        right_pixel_count = (img.shape[0] - min_y)  # 计算右边这条线有多少个点
        print('right_pixel_count: ', right_pixel_count)

        for x in range(min_y, img.shape[0]):
            right_hsv_pixel = img_hsv[(x, int(right_curve(x)))]
            right_gray_pixel = gray[(x, int(right_curve(x)))]
            if (right_hsv_pixel[0] >= 10 and right_hsv_pixel[0] <= 29) and \
                    (right_hsv_pixel[1] >= 43 and right_hsv_pixel[1] <= 255) and \
                        (right_hsv_pixel[2] >= 46 and right_hsv_pixel[2] <= 255):
                right_hsv_pixel_count += 1
            if right_gray_pixel >= 200 and right_gray_pixel <= 255:
                right_gray_pixel_count += 1
        print('right_hsv_pixel_count: ', right_hsv_pixel_count)
        print('right_gray_pixel_count: ', right_gray_pixel_count)
        if right_hsv_pixel_count > right_gray_pixel_count or right_hsv_pixel_count > 20:
            cv2.line(img, r1, r2, [255, 255, 0], thickness)  # 代表黄色
        else:
            cv2.line(img, r1, r2, [255, 255, 255], thickness)  # 代表白色

    # Draw All Lines to see
    # if True:
    #     tmp = right_lines
    #     for i in range(tmp.shape[0]):
    #         x0, y0, x1, y1 = tmp[i, 0], tmp[i, 1], tmp[i, 2], tmp[i, 3]
    #         cv2.line(img, (x0, y0), (x1, y1), [255, 0, 0], thickness=5)
    #     plt.imshow(img)
    #     plt.show()

    
    # remove junk from lists
    # lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
    # lines.shape = (lines.shape[0]//2,2)
    # slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    # convert lines into list of points

    # Draw All Lines to see
    # if True:
    #     tmp = np.squeeze(lines)
    #     for i in range(tmp.shape[0]):
    #         x0, y0, x1, y1 = tmp[i, 0], tmp[i, 1], tmp[i, 2], tmp[i, 3]
    #         cv2.line(img, (x0, y0), (x1, y1), [255, 0, 0], thickness=5)
    #     plt.imshow(img)
    #     plt.show()


    # Right lane
    # move all points with negative slopes into right "lane"
    # right_slopes = slopes[slopes<0]
    # right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1]/2), lines)))
    # max_right_x, max_right_y = right_lines.max(axis=0)
    # min_right_x, min_right_y = right_lines.min(axis=0)

    # Left lane
    # all positive  slopes go into left "lane"
    # left_slopes = slopes[slopes > 0]
    # left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1]/2), lines)))
    # max_left_x, max_left_y = left_lines.max(axis=0)
    # min_left_x, min_left_y = left_lines.min(axis=0)

    

    
    
def draw_lines2(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept 
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    alpha =0.2 
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function
    
    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1
        
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1
    
   
    
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")
    
    if first_frame == 1:
        next_frame = current_frame        
        first_frame = 0        
    else :
        prev_frame = cache
        next_frame = (1-alpha)*prev_frame+alpha*current_frame
             
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)
    
    cache = next_frame
    

 
def hough_lines(image_org, img, rho, theta, threshold, min_line_len, max_line_gap):    #image_org 原始输入图片，作为后续采用颜色用；img是masked_edges ROI的mask
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(image_org, line_img, lines)
    return line_img

 
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)



# Takes in a single frame or an image and returns a marked image
def mark_lanes(image):
    if image is None: raise ValueError("no image given to mark_lanes")
    # grayscale the image to make finding gradients clearer
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    #hsv
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([10, 43, 46], dtype = "uint8")   #20,100,100    #26
    upper_yellow = np.array([29, 255, 255], dtype="uint8")    #20.255.255   #34
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)
    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)
    

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 30
    high_threshold = 150
   # edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)   #old blur_gray  gray image 
    edges_img = cv2.Canny(np.uint8(gauss_gray), low_threshold, high_threshold)   #hsv image 
    #print(" edges_img height:",edges_img.shape[0])
    #print(" edges_img width:",edges_img.shape[1])

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),
                          (620, 410),    #[620,410]
                          (imshape[1], imshape[0]) ]],
                          dtype=np.int32)

    #vertices = np.array([[(0+200, imshape[0]),(620, 410),(643, 410),(imshape[1]-300, imshape[0]),(imshape[1]-500, imshape[0]),
    #                       (543, 510),(720, 510),(0+400, imshape[0])]],dtype=np.int32)

    # 3
    #left_bottom = [0, edges_img.shape[0]]
    #right_bottom = [edges_img.shape[1], edges_img.shape[0]]
    #apex = [edges_img.shape[1]/2, 410]
    #vertices = np.array([ left_bottom, right_bottom, apex ], np.int32)

    masked_edges = region_of_interest(edges_img, vertices)


    # Define the Hough transform parameters
    rho             = 2           # distance resolution in pixels of the Hough grid
    theta           = np.pi/180   # angular resolution in radians of the Hough grid
    threshold       = 30       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 150       # minimum number of pixels making up a line
    max_line_gap    = 500       # maximum gap in pixels between connectable line segments

    line_image = hough_lines(image, masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the image
    # initial_img * alpha + img * β + λ
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return  mask_yellow, mask_white, mask_yw, mask_yw_image, blur_gray, edges_img, masked_edges, line_image, lines_edges
    
    

def read_image_for_marking(img_filepath):
    # read in the image
    image = mpimg.imread(img_filepath)
    print('Reading image :', img_filepath, '\nDimensions:', image.shape)

      
    mask_yellow, mask_white, mask_yw, mask_yw_image, blur_gray, edges_image, masked_edges, line_image, marked_lanes = mark_lanes(image)

    # show the image to plotter and then save it to a file
    plt.imshow(marked_lanes)
    plt.savefig(img_filepath[:-4] + '_output.png')

    #plt.imshow(mask_yw)
    #plt.savefig(img_filepath[:-4] + '_mask_yw_output.png')

    #plt.imshow(mask_yw_image)
    #plt.savefig(img_filepath[:-4] + '_mask_yw_image_output.png')

    #plt.imshow(mask_yellow)
    #plt.savefig(img_filepath[:-4] + '_mask_yellow_output.png')

    #plt.imshow(mask_white)
    #plt.savefig(img_filepath[:-4] + '_mask_white_output.png')


    # plt.imshow(line_image)
    # plt.savefig(img_filepath[:-4] + '_line_image_output.png')
      

   # plt.imshow(edges_image)
   # plt.savefig(img_filepath[:-4] + '_edges_image_output.png')
     
   # plt.imshow(blur_gray)
   # plt.savefig(img_filepath[:-4] + '_blur_gray_output.png')

   # plt.imshow(masked_edges)
   # plt.savefig(img_filepath[:-4] + '_masked_edges_output.png')

    plt.imshow(line_image)
    plt.savefig(img_filepath[:-4] + '_line_image_output.png')


if __name__ == "__main__":
    # set up parser
     parser = argparse.ArgumentParser()
     parser.add_argument("-f", "--file", help="filepath for image to mark", default='test_images/solidWhiteRight.jpg')
     parser.add_argument("-v", "--video", help="filepath for video to mark")
     args = parser.parse_args()

     if args.video:
         clip = VideoFileClip(args.video)
         clip = clip.fl_image(mark_lanes)
         clip.write_videofile('output_' + args.video, audio=False)

     else:
        # # if nothing passed running algorithm on image
         read_image_for_marking(args.file)
   # for i in range(1148, 1348, 1):
   #     im_name = 'image/' + str(i) +'.jpg'
   #     read_image_for_marking(im_name)

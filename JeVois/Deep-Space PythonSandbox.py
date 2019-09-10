import libjevois as jevois
import cv2
import numpy as np
import math

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is a basic FRC vision process.
#
# By default, it first gets an image, blurs it, extracts the green channel, thresholds that, and uses the threshold to place
# a mask over the initial image. It then runs an HSV filter on the masked image, erodes and dilates the result, and finally
# finds and filters the contours.
# You can find the constants for all of these using GRIP. Tune the program constants and generate Python code from GRIP. Then,
# paste those constants into the Constructor below. Custom code can also be inserted after all the GRIP process code.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Anand Rajamani
# 
# @videomapping YUYV 320 240 59.9 YUYV 320 240 59.9 JeVois PythonSandbox
# @email anand.rajamani@scu.edu
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules

class PythonSandbox:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)

        # SPECIAL REPLACED BLUR CONSTANT
        self.__blur_type = 0

    # ###################################################################################################  
        # ALL CONSTANTS GO UNDER HERE (make sure to remove the self.__blur_type line)
    
        self.__normalize_type = cv2.NORM_MINMAX
        self.__normalize_alpha = 0.0
        self.__normalize_beta = 255.0

        self.normalize_output = None

        self.__hsv_threshold_input = self.normalize_output
        self.__hsv_threshold_hue = [0.0, 106.20320855614972]
        self.__hsv_threshold_saturation = [0.0, 255.0]
        self.__hsv_threshold_value = [245.0, 255.0]

        self.hsv_threshold_output = None

        self.__cv_erode_src = self.hsv_threshold_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 1.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None

        self.__mask_input = self.cv_erode_output
        self.__mask_mask = self.hsv_threshold_output

        self.mask_output = None

        self.__blur_input = self.mask_output
        self.__blur_radius = 4.319224885262623

        self.blur_output = None

        self.__find_blobs_input = self.blur_output
        self.__find_blobs_min_area = 500.0
        self.__find_blobs_circularity = [0.0, 1.0]
        self.__find_blobs_dark_blobs = False

        self.find_blobs_output = None

        self.__cv_cvtcolor_src = self.blur_output
        self.__cv_cvtcolor_code = cv2.COLOR_YUV2GRAY_420

        self.cv_cvtcolor_output = None

        self.__find_contours_input = self.cv_cvtcolor_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 500.0
        self.__filter_contours_min_perimeter = 0.0
        self.__filter_contours_min_width = 0.0
        self.__filter_contours_max_width = 1000.0
        self.__filter_contours_min_height = 0.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [0, 100]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 0.0
        self.__filter_contours_min_ratio = 0.0
        self.__filter_contours_max_ratio = 1000.0

        self.filter_contours_output = None



        # END CONSTANTS  
    # ###################################################################################################

    ## Process function with USB output

    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        source0 = inimg = inframe.getCvBGR()
        outimg = inimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

#################################################################################################

        # BEGIN GRIP CODE

#################################################################################################
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step Normalize0
        self.normalize_input = source0
        (self.normalize_output) = self.__normalize(self.normalize_input, self.__normalize_type, self.__normalize_alpha, self.__normalize_beta)


        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.normalize_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_Erode0
        self.__cv_erode_input = self.hsv_threshold_output
        (self.cv_erode) = self.__cv_erode(self.__cv_erode_input, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)
               
         # Step Mask0:
        #self.__mask_input = self.cv_erode_output
        self.__mask_input = self.normalize_output
        self.__mask_mask = self.hsv_threshold_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)
        
        # Step Blur0:
        self.__blur_input = self.hsv_threshold_output
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step Find_Blobs0:
        self.__find_blobs_input = self.blur_output
        (self.find_blobs_output) = self.__find_blobs(self.__find_blobs_input, self.__find_blobs_min_area, self.__find_blobs_circularity, self.__find_blobs_dark_blobs)

        # Step CV_cvtColor0:
        self.__cv_cvtcolor_src = self.blur_output
        (self.cv_cvtcolor_output) = self.__cv_cvtcolor(self.__cv_cvtcolor_src, self.__cv_cvtcolor_code)
        
        # Step Find_Contours0:
        self.__find_contours_input = self.cv_cvtcolor_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)
        
#################################################################################################

        # END GRIP CODE

##################################################################################################
        
        # DEFAULT CUSTOM CODE

        def getArea(con): # Gets the area of the contour
            return cv2.contourArea(con)

        def getYcoord(con): # Gets the Y coordinate of the contour
            M = cv2.moments(con)
            cy = int(M['m01']/M['m00'])
            return cy

        def getXcoord(con): # Gets the X coordinate of the contour
            M = cv2.moments(con)
            cy = int(M['m10']/M['m00'])
            return cy

        def sortByArea(conts) : # Returns an array sorted by area from smallest to largest
            contourNum = len(conts) # Gets number of contours
            sortedBy = sorted(conts, key=getArea) # sortedBy now has all the contours sorted by area
            return sortedBy
        
##################################################################################################
        
        # PUT YOUR CUSTOM CODE HERE
        
##################################################################################################
        
        # Draws all contours on original image in red
        cv2.drawContours(outimg, self.filter_contours_output, -1, (0, 0, 255), 1)
        
        # Gets number of contours
        contourNum = len(self.filter_contours_output)

        # Sorts contours by the smallest area first
        newContours = sortByArea(self.filter_contours_output)       

        # Send the contour data over Serial
        #dfch means distance from crosshair
        dfchCON0 = 0
        dfchCON1 = 0
        CONSdfch = 0
        AbsCONSdfch = 0
        for i in range (contourNum):
            cnt = newContours[i]
            #self.markerX = round((getXcoord(cnt)*1000/320)-500, 2))
            #self.markerY = round(375-getYcoord(cnt)*750/240, 2))
            
            #cv2.drawMarker(outimg, (self.markerX, self.markerY) , (255, 0,0), 1, 1, 3)
            x,y,w,h = cv2.boundingRect(cnt) # Get the stats of the contour including width and height
                # which contour, 0 is first
            crosshairX = 165
            conNum = "CON" + str(i)
            xOfT = round(getXcoord(cnt))
            dfch = ((getXcoord(cnt)-crosshairX))                        

            if(conNum == "CON0"):
                toSend = "CON0" + " x = " + str(xOfT) + " dfch = " + str(dfch)
                dfchCON0 = dfchCON0 + dfch
                CONSdfch = CONSdfch + dfch
                AbsCONSdfch = AbsCONSdfch + math.fabs(dfch)
            elif(conNum == "CON1"):
                toSend = "CON1" + " x = " + str(xOfT) + " dfch = " + str(dfch)
                dfchCON1 = dfchCON1 + dfch
                CONSdfch = CONSdfch + dfch
                AbsCONSdfch = AbsCONSdfch + math.fabs(dfch)
        
                     
            jevois.sendSerial(toSend)
        
        toSend = "CONSdfch = " + str(CONSdfch)
        jevois.sendSerial(toSend)
        
        #Drawing the cross-hair
        crosshairX = 165
        cv2.drawMarker(outimg, (crosshairX, 125), (0, 0, 255), 1, 1, 3)

        #Running code only when there are 2 or more contoursks
        if(len(self.filter_contours_output) >= 2):
            #Changing the cross-hair's color
            zero = dfchCON0 + dfchCON1
            zero = math.fabs(zero)
            greenMarker = False
            if (zero <= 10):
                greenMarker = True
            else: 
                greenMarker = False

            if(greenMarker):
                cv2.drawMarker(outimg, (165, 125), (0, 255, 0), 1, 1, 3)
            else:
                cv2.drawMarker(outimg, (165, 125), (0, 0, 255), 1, 1, 3)

            #Getting the distance
            halfOfDistanceBetweenTargs = AbsCONSdfch/2
            toSend = "half of distance = " + str(halfOfDistanceBetweenTargs)
            jevois.sendSerial(toSend)
            tanOfX = halfOfDistanceBetweenTargs*(math.tan(math.radians(32.5))/155)
            angleOfCameraAndTarget = math.atan((halfOfDistanceBetweenTargs*math.tan(math.radians(32.5)))/155)
            toSend = "angle of the camera and target = " + str(angleOfCameraAndTarget)
            jevois.sendSerial(toSend)
            distance = (4/tanOfX) + 12
            toSend = "distance = " + str(distance)
            jevois.sendSerial(toSend)
        else:
            toSend = "No Contours Can Be Found"
            jevois.sendSerial(toSend)

        

            
        #cv2.drawContours(contourNum)
        

        # Write a title:
        cv2.putText(outimg, "Jalen's Code", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        #height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
        height, width, channels = outimg.shape
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Convert our BGR output image to video output format and send to host over USB. If your output image is not
        # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        outframe.sendCvBGR(outimg)
        #outframe.sendCvGRAY(outimg)
        
##################################################################################################
        
        # END CUSTOM CODE
        
###################################################################################################

    # FUNCTIONS GO HERE (Anything that starts with "@staticmethod")
        
    @staticmethod
    def __normalize(input, type, a, b):
        """Normalizes or remaps the values of pixels in an image.
        Args:
            input: A numpy.ndarray.
            type: Opencv enum.
            a: The minimum value.
            b: The maximum value.
        Returns:
            A numpy.ndarray of the same type as the input.
        """
        return cv2.normalize(input, None, a, b, type)

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)

    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __find_blobs(input, min_area, circularity, dark_blobs):
        """Detects groups of pixels in an image.
        Args:
            input: A numpy.ndarray.
            min_area: The minimum blob size to be found.
            circularity: The min and max circularity as a list of two numbers.
            dark_blobs: A boolean. If true looks for black. Otherwise it looks for white.
        Returns:
            A list of KeyPoint.
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = 1
        params.blobColor = (0 if dark_blobs else 255)
        params.minThreshold = 10
        params.maxThreshold = 220
        params.filterByArea = True
        params.minArea = min_area
        params.filterByCircularity = True
        params.minCircularity = circularity[0]
        params.maxCircularity = circularity[1]
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(input)

    @staticmethod
    def __cv_cvtcolor(src, code):
        """Converts an image from one color space to another.
        Args:
           src: A numpy.ndarray.
           code: The conversion code. (opencv eum)
        Result:
           A numpy.ndarray in the new color space.
        """
        return cv2.cvtColor(src, code)

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output


#BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')


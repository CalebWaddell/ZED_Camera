///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, Caleb Waddell.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


/***************************************************************************************************
 ** This application retrieves visual and depth images from the ZED camera				          **
 ** This data is thresholded to detect object in three distinct fields                            **
 ** A line detection algorithm has been implemented on the visual image						      **
 ***************************************************************************************************/

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Core.hpp>
#include <sl/defines.hpp>

#include <math.h>

using namespace sl;
using namespace cv;

cv::Mat slMat2cvMat(sl::Mat& input);
void detectObjects(cv::Mat* thresh_img, cv::Mat* orig_img, int type);
void isolateLine(cv::Mat* thresh_img, cv::Mat* orig_img, int type);
void findLinePos(cv::Mat* thresh_img, cv::Mat* orig_img, cv::Point correction);

enum {
	TYPE_CLOSE = 1,
	TYPE_MED = 2,
	TYPE_FAR = 3,
};

int globalHeight = 720;
int globalWidth = 1280;

int d1 = 80;
int d2 = 450;
int d3 = 200;
int d4 = 720/3;
int d5 = 450;

int MIN_LINE_AREA_THRESH = 1000;

int CLOSE_THRESH = 1000;
int CLOSE_MED_THRESH = 2217;
int MED_FAR_THRESH = 3710;
int FAR_THRESH = 8000;
int MIN_AREA_THRESH = 2000;

int MIN_BLUE_H = 80;
int MAX_BLUE_H = 120;
int MIN_BLUE_S = 100;
int MAX_BLUE_S = 255;
int MIN_BLUE_V = 100;
int MAX_BLUE_V = 255;

int hough_thresh = 130;
int hough_min_line_length = 0;
int hough_max_line_gap = 0;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720;
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
    init_params.coordinate_units = sl::UNIT_METER;
    init_params.depth_minimum_distance = 1.0;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS){
        return 1;
    }

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_STANDARD; // Use STANDARD sensing mode

    Resolution image_size = zed.getResolution();
	sl::Mat image_zed(image_size, sl::MAT_TYPE_8U_C4); // Create a sl::Mat to handle Left image
	cv::Mat image_ocv = slMat2cvMat(image_zed);
	sl::Mat depth_image_zed(image_size, MAT_TYPE_8U_C4);
	cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);

	//SETUP POINTS FOR SEGMENTATION
	cv::Point point12 = cv::Point(((image_ocv.size().width)/2)-d1,0);
	cv::Point point23 = cv::Point(((image_ocv.size().width)/2)+d1,0);

	cv::Point point14 = cv::Point(0,((image_ocv.size().height)/3));
	cv::Point point47 = cv::Point(0,2*((image_ocv.size().height)/3));

	cv::Point point36 = cv::Point((image_ocv.size().width),((image_ocv.size().height)/3));
	cv::Point point69 = cv::Point((image_ocv.size().width),2*((image_ocv.size().height)/3));

	cv::Point point78 = cv::Point(((image_ocv.size().width)/2)-d2,(image_ocv.size().height));
	cv::Point point89 = cv::Point(((image_ocv.size().width)/2)+d2,(image_ocv.size().height));
	//*********************

	cv::Mat image_ocv_hsv,image_ocv_hsv_thresh,image_ocv_hsv_canny,image_ocv_hsv_hough,line_base_roi,filtered_line,line_roi;
	filtered_line = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	line_base_roi = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	line_roi = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	image_ocv_hsv_thresh = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	image_ocv_hsv_canny = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	image_ocv_hsv_hough = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	std::vector<cv::Point> fillContSingle;
	Point pointROI1 = Point(((image_ocv.size().width)/2) - d3,d4);
	Point pointROI2 = Point(((image_ocv.size().width)/2) + d3,d4);
	Point pointROI3 = Point(((image_ocv.size().width)/2) - d5,(image_ocv.size().height));
	Point pointROI4 = Point(((image_ocv.size().width)/2) + d5,(image_ocv.size().height));
	fillContSingle.push_back(pointROI1);
	fillContSingle.push_back(pointROI2);
	fillContSingle.push_back(pointROI3);
	fillContSingle.push_back(pointROI4);
	cv::convexHull(fillContSingle,fillContSingle);
	std::vector<std::vector<cv::Point>> fillContAll;
	fillContAll.push_back(fillContSingle);
	cv::fillPoly(line_roi,fillContAll,cv::Scalar(255));

	sl::Mat depthImageRaw_zed(image_size, sl::MAT_TYPE_32F_C1); // Create a sl::Mat to handle depth image (32 bit floats)
	cv::Mat depthImageRaw_ocv = slMat2cvMat(depthImageRaw_zed);

	cv::Mat depthImageThreshCLOSE, depthImageThreshMEDIUM,depthImageThreshFAR;
	depthImageThreshCLOSE = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	depthImageThreshMEDIUM = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	depthImageThreshFAR = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);

	cv::Mat depthImageThreshCLOSE2,depthImageThreshMEDIUM2,depthImageThreshFAR2,lineMask;
	depthImageThreshCLOSE2 = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	depthImageThreshMEDIUM2 = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);
	depthImageThreshFAR2 = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);

	cv::Mat depthImageThreshCLOSE_CROPPED,depthImageThreshMEDIUM_CROPPED,depthImageThreshFAR_CROPPED;
	Rect roiMEDIUM(0,0,point69.x,point69.y);
	Rect roiFAR(0,0,point36.x,point36.y);

    // Create OpenCV images to display (lower resolution to fit the screen)
    cv::Size displaySize(480, 269);
    cv::Mat image_ocv_display(displaySize, CV_8UC4);
    cv::Mat depth_image_ocv_display(displaySize, CV_8UC4);

    // Give a name to OpenCV Windows
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Line Thresh", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("CLOSE", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("MEDIUM", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("FAR", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);

//    createTrackbar("MIN_BLUE_H","Line Thresh",&MIN_BLUE_H,255);
//    createTrackbar("MAX_BLUE_H","Line Thresh",&MAX_BLUE_H,255);
//    createTrackbar("MIN_BLUE_S","Line Thresh",&MIN_BLUE_S,255);
//    createTrackbar("MAX_BLUE_S","Line Thresh",&MAX_BLUE_S,255);
//    createTrackbar("MIN_BLUE_V","Line Thresh",&MIN_BLUE_V,255);
//    createTrackbar("MAX_BLUE_V","Line Thresh",&MAX_BLUE_V,255);

    createTrackbar("THRESH","Image",&hough_thresh,255);
	createTrackbar("MIN LINE LENGTH","Image",&hough_min_line_length,255);
	createTrackbar("MAX LINE GAP","Image",&hough_max_line_gap,255);

//    createTrackbar("CLOSE","Image",&CLOSE_THRESH,10000);
//    createTrackbar("CLOSE-MED","Image",&CLOSE_MED_THRESH,10000);
//    createTrackbar("MED-FAR","Image",&MED_FAR_THRESH,10000);
//    createTrackbar("FAR","Image",&FAR_THRESH,10000);
//    createTrackbar("MIN Area","Image",&MIN_AREA_THRESH,10000);

    // Jetson only. Execute the calling thread on 2nd core
    Camera::sticktoCPUCore(2);

    // Loop until 'q' is pressed
    char key = ' ';

    int64_t lastTick = getTickCount();
    int64_t currentTick = getTickCount();
    double tickFreq = getTickFrequency();
    double fps = 0;

    while (key != 'q') {

    	lastTick = getTickCount();

        if (zed.grab(runtime_parameters) == SUCCESS) {

            zed.retrieveImage(image_zed, VIEW_LEFT); // Retrieve the left image
            zed.retrieveImage(depth_image_zed, VIEW_DEPTH); // Retrieve the left image
            zed.retrieveMeasure(depthImageRaw_zed, MEASURE_DEPTH); // Retrieve the depth measure (32bits)

            //filtered_line = cv::Mat::zeros(image_size.height,image_size.width,CV_8UC1);

			cv::cvtColor(image_ocv,image_ocv_hsv,CV_BGR2HSV);
			inRange(image_ocv_hsv,Scalar(MIN_BLUE_H,MIN_BLUE_S,MIN_BLUE_V),Scalar(MAX_BLUE_H,MAX_BLUE_S,MAX_BLUE_V),image_ocv_hsv_thresh);	//0-2m
			//isolateLine(&image_ocv_hsv_thresh,&filtered_line,0);

			cv::bitwise_and(image_ocv_hsv_thresh,line_roi,image_ocv_hsv_thresh);

			Rect roi_level_1(200,(image_size.height-25),(image_size.width-200-200),25);
			line_base_roi = image_ocv_hsv_thresh(roi_level_1);
			findLinePos(&line_base_roi,&image_ocv,Point(200,(image_size.height-25)));

			Rect roi_level_2(200,(image_size.height-125),(image_size.width-200-200),25);
			line_base_roi = image_ocv_hsv_thresh(roi_level_2);
			findLinePos(&line_base_roi,&image_ocv,Point(200,(image_size.height-125)));

			Rect roi_level_3(200,(image_size.height-225),(image_size.width-200-200),25);
			line_base_roi = image_ocv_hsv_thresh(roi_level_3);
			findLinePos(&line_base_roi,&image_ocv,Point(200,(image_size.height-225)));

			//cv::Canny(image_ocv_hsv_thresh,image_ocv_hsv_canny,100,200);
//			vector<Vec4i> linesP;
//			cv::HoughLinesP(image_ocv_hsv_canny,linesP,1,CV_PI/180,hough_thresh,hough_min_line_length,hough_max_line_gap);
//			for(int i = 0; i < linesP.size(); i++){
//				Vec4i l = linesP[i];
//				line(image_ocv,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,0,255),3);
//			}
//			vector<Vec2f> lines;
//			cv::HoughLines(image_ocv_hsv_canny,lines,1,CV_PI/180,hough_thresh);
//			for(int i = 0; i < lines.size(); i++){
//				float rho = lines[i][0], theta = lines[i][1];
//				Point pt1, pt2;
//				double a = cos(theta), b = sin(theta);
//				double x0 = a*rho, y0 = b*rho;
//				pt1.x = cvRound(x0 + 1000*(-b));
//				pt1.y = cvRound(y0 + 1000*(a));
//				pt2.x = cvRound(x0 - 1000*(-b));
//				pt2.y = cvRound(y0 - 1000*(a));
//				line(image_ocv,pt1,pt2,Scalar(0,0,255),3);
//			}


//            inRange(depthImageRaw_ocv,Scalar((float)CLOSE_THRESH/1000),Scalar((float)CLOSE_MED_THRESH/1000),depthImageThreshCLOSE);	//0-2m
//            inRange(depthImageRaw_ocv,Scalar((float)CLOSE_MED_THRESH/1000),Scalar((float)MED_FAR_THRESH/1000),depthImageThreshMEDIUM);	//0-2m
//            inRange(depthImageRaw_ocv,Scalar((float)MED_FAR_THRESH/1000),Scalar((float)FAR_THRESH/1000),depthImageThreshFAR);	//0-2m
//
//            cv::line(depthImageThreshCLOSE,point12,point78,0,3);
//			cv::line(depthImageThreshCLOSE,point23,point89,0,3);
//			cv::line(depthImageThreshCLOSE,point14,point36,0,3);
//			cv::line(depthImageThreshCLOSE,point47,point69,0,3);
//			cv::line(depthImageThreshMEDIUM,point12,point78,Scalar(0),3);
//			cv::line(depthImageThreshMEDIUM,point23,point89,Scalar(0),3);
//			cv::line(depthImageThreshMEDIUM,point14,point36,Scalar(0),3);
//			cv::line(depthImageThreshMEDIUM,point47,point69,Scalar(0),3);
//			cv::line(depthImageThreshFAR,point12,point78,Scalar(0),3);
//			cv::line(depthImageThreshFAR,point23,point89,Scalar(0),3);
//			cv::line(depthImageThreshFAR,point14,point36,Scalar(0),3);
//			cv::line(depthImageThreshFAR,point47,point69,Scalar(0),3);
//
//            cv::resize(depthImageThreshCLOSE, depthImageThreshCLOSE2, displaySize);
//			cv::resize(depthImageThreshMEDIUM, depthImageThreshMEDIUM2, displaySize);
//			cv::resize(depthImageThreshFAR, depthImageThreshFAR2, displaySize);
//
//			depthImageThreshMEDIUM_CROPPED = depthImageThreshMEDIUM(roiMEDIUM);
//			depthImageThreshFAR_CROPPED = depthImageThreshFAR(roiFAR);
//
//            detectObjects(&depthImageThreshCLOSE,&image_ocv,TYPE_CLOSE);
//			detectObjects(&depthImageThreshMEDIUM_CROPPED,&image_ocv,TYPE_MED);
//			detectObjects(&depthImageThreshFAR_CROPPED,&image_ocv,TYPE_FAR);
//
			cv::line(image_ocv,point12,point78,Scalar(255,0,0),3);
			cv::line(image_ocv,point23,point89,Scalar(255,0,0),3);
			cv::line(image_ocv,point14,point36,Scalar(255,0,0),3);
			cv::line(image_ocv,point47,point69,Scalar(255,0,0),3);

            cv::resize(image_ocv, image_ocv_display, displaySize*2);
            imshow("Image", image_ocv_display);
//            imshow("Line Thresh", image_ocv_hsv_thresh);
//            imshow("Line", line_base_roi);
//            cv::resize(image_ocv_hsv_canny, image_ocv_hsv_canny, displaySize);
//            imshow("Line Canny", image_ocv_hsv_canny);

//
//            imshow("CLOSE", depthImageThreshCLOSE2);
//            imshow("MEDIUM", depthImageThreshMEDIUM2);
//            imshow("FAR", depthImageThreshFAR2);
//
//            cv::resize(depth_image_ocv, depth_image_ocv_display, displaySize);
//            imshow("Depth", depth_image_ocv_display);

            key = cv::waitKey(1);

            currentTick = getTickCount();
            fps =1/((currentTick-lastTick)/tickFreq);
            printf("FPS: %f\n",fps);
        }
    }

    zed.close();
    return 0;
}

cv::Mat slMat2cvMat(sl::Mat& input)
{

	//convert MAT_TYPE to CV_TYPE
	int cv_type = -1;
	switch (input.getDataType())
	{
	case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
	case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
	case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
	case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
	case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
	case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
	case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
	case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// cv::Mat data requires a uchar* pointer. Therefore, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	//cv::Mat and sl::Mat will share the same memory pointer
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

void detectObjects(cv::Mat* thresh_img, cv::Mat* orig_img, int type) {
	Scalar color;
	if(type == TYPE_CLOSE){
		color = Scalar(0,0,255);
	}
	else if(type == TYPE_MED){
		color = Scalar(0,106,255);
	}
	else if(type == TYPE_FAR){
		color = Scalar(0,255,0);
	}
	else{
		return;
	}
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cv::findContours(*thresh_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mu[i] = moments(contours[i], false);
	}
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	int x_pos = 0;
	int y_pos = 0;
	//int sectionSize = ((thresh_img->size().height)/3);
	int area = 0;
	//int perimeter = 0;
	//float circularity = 1;
	//std::cout << "size: " << contours.size() << std::endl;
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		//perimeter = arcLength(contours[i], true);
		//circularity = (((float)perimeter*(float)perimeter) / (4 * CV_PI*(float)area));

		if ((area > MIN_AREA_THRESH)){//&&(area < thresh_high)) {

			x_pos = mc[i].x;
			y_pos = mc[i].y;
			if(type == TYPE_CLOSE){
				color = Scalar(0,0,255);	//red
				drawContours(*orig_img, contours, i, color, 5, 8, hierarchy, 0, Point());
				cv::circle(*orig_img, mc[i], 50, color, 3, 8, 0);
				std::cout << "***Critical Object Detected*** [x,y] = " << mc[i] << std::endl;
			}
			else if(type == TYPE_MED){
				color = Scalar(0,106,255);
				if(x_pos > ((((d2-d1)/(float)globalHeight)*y_pos) + ((float)globalWidth/2) + d1)){	//we are in the right hand column
					color = Scalar(66,238,244);	//yellow
					drawContours(*orig_img, contours, i, color, 5, 8, hierarchy, 0, Point());
					cv::circle(*orig_img, mc[i], 50, color, 3, 8, 0);
					std::cout << "***Low Threat Object Detected*** [x,y] = " << mc[i] << std::endl;
				}
				else if(x_pos < (((-(d2-d1)/(float)globalHeight)*y_pos) + ((float)globalWidth/2) - d1)){
					color = Scalar(66,238,244);	//yellow
					drawContours(*orig_img, contours, i, color, 5, 8, hierarchy, 0, Point());
					cv::circle(*orig_img, mc[i], 50, color, 3, 8, 0);
					std::cout << "***Low Threat Object Detected*** [x,y] = " << mc[i] << std::endl;
				}
				else{
					color = Scalar(0,106,255);	//orange
					drawContours(*orig_img, contours, i, color, 5, 8, hierarchy, 0, Point());
					cv::circle(*orig_img, mc[i], 50, color, 3, 8, 0);
					std::cout << "***High Threat Object Detected*** [x,y] = " << mc[i] << std::endl;
				}
			}
			else if(type == TYPE_FAR){
				color = Scalar(0,255,0);
				drawContours(*orig_img, contours, i, color, 5, 8, hierarchy, 0, Point());
				cv::circle(*orig_img, mc[i], 50, color, 3, 8, 0);
				std::cout << "***Potential Object Detected*** [x,y] = " << mc[i] << std::endl;
			}
		}
	}
}

void isolateLine(cv::Mat* thresh_img, cv::Mat* orig_img, int type) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cv::findContours(*thresh_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
//	vector<Moments> mu(contours.size());
//	for (int i = 0; i < contours.size(); i++) {
//		mu[i] = moments(contours[i], false);
//	}
//	vector<Point2f> mc(contours.size());
//	for (int i = 0; i < contours.size(); i++) {
//		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
//	}
//
//	int x_pos = 0;
//	int y_pos = 0;
	//int sectionSize = ((thresh_img->size().height)/3);
	int area = 0;
	//int perimeter = 0;
	//float circularity = 1;
	//std::cout << "size: " << contours.size() << std::endl;
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		//perimeter = arcLength(contours[i], true);
		//circularity = (((float)perimeter*(float)perimeter) / (4 * CV_PI*(float)area));

		if (area > MIN_LINE_AREA_THRESH){
			Rect bounding_rect = boundingRect(contours[i]);
			if(((bounding_rect.y + bounding_rect.height) > (globalHeight - 20))&& (bounding_rect.y < (globalHeight - 200))){//&&(area < thresh_high)) {
				drawContours(*orig_img,contours,i,Scalar(255),CV_FILLED);
				//cv::rectangle(*orig_img,bounding_rect,Scalar(0,0,255),3);
			}
		}
	}
}

void findLinePos(cv::Mat* thresh_img, cv::Mat* orig_img, cv::Point correction) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cv::findContours(*thresh_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mu[i] = moments(contours[i], false);
	}
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}
//
//	int x_pos = 0;
//	int y_pos = 0;
	//int sectionSize = ((thresh_img->size().height)/3);
	int area = 0;
	//int perimeter = 0;
	//float circularity = 1;
	//std::cout << "size: " << contours.size() << std::endl;
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		//perimeter = arcLength(contours[i], true);
		//circularity = (((float)perimeter*(float)perimeter) / (4 * CV_PI*(float)area));

		if (area > 20){
			Rect bounding_rect = boundingRect(contours[i]);
			bounding_rect = bounding_rect + correction;
			cv::rectangle(*orig_img,bounding_rect,Scalar(0,0,255),3);
			//circle(*orig_img,Point((correction.x+mc[i].x),(correction.y+mc[i].y)),40,Scalar(0,0,255),3);
		}
	}
}

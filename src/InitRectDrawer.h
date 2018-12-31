#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <sys/io.h>
#include <cstdlib>

#define FAIL  -1
#define SUCCESS     1
#define FALSE  0
#define TRUE  1

#define OUT_OF_FRAME    2

#define ENTER  13
#define ESC   27

using namespace std;

/* ==========================================================================

Class : TargetRectDrawer

This class makes users draw initial rectangles on the window.
It can draw many rectangles and these rectangles are used to initialize
the SingleTracker class.

========================================================================== */
class TargetRectDrawer
{
private:
	cv::Mat img_orig;  // Original Image
	cv::Mat img_prev;  // Previous image(before draw new rectangle)
	cv::Mat img_draw;  // Draw rectangle here
	std::vector<std::pair<cv::Rect, cv::Scalar>> rect_vec;  // Drawing result

	int start_x;   // the first x (x when left mouse button is clicked)
	int start_y;   // the first y (y when left mouse button is clicked)

	bool is_drawing_rect; // is drawing rectangle?

public:
	TargetRectDrawer() : start_x(0), start_y(0), is_drawing_rect(false) {}

	/* Set Function */
	void setImgOrig(cv::Mat& _img) { this->img_orig = _img.clone(); } // Deep copy
	void setImgPrev(cv::Mat& _img) { this->img_prev = _img.clone(); } // Deep copy
	void setImgDraw(cv::Mat& _img) { this->img_draw = _img.clone(); } // Deep copy
	void setStartX(int _start_x) { this->start_x = _start_x; }
	void setStartY(int _start_y) { this->start_y = _start_y; }
	void setIsDrawingRect(bool _is_drawing) { this->is_drawing_rect = _is_drawing; }

	/* Get Function */
	cv::Mat& getImgOrig() { return this->img_orig; }
	cv::Mat& getImgPrev() { return this->img_prev; }
	cv::Mat& getImgDraw() { return this->img_draw; }
	bool getIsDrawingRect() { return this->is_drawing_rect; }
	std::vector<std::pair<cv::Rect, cv::Scalar>>& getRectVec() { return this->rect_vec; }

	/* Core Function */
	int initTargetRectDrawer(cv::Mat& _img);
	static void wrapperCallBackFunc(int event, int x, int y, int flags, void* userdata);
	void CallBackFunc(int event, int x, int y, int flags);
	std::vector<std::pair<cv::Rect, cv::Scalar>>& drawInitRect(cv::Mat& _mat_img);
};
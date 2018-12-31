#include "InitRectDrawer.h"

/* -----------------------------------------------------------------
Function : initTargetRectDrawer

Initialize key variables in TargetRectDrawer class.
----------------------------------------------------------------- */
int TargetRectDrawer::initTargetRectDrawer(cv::Mat& _img)
{
	if (_img.empty())
	{
		std::cout << "====================== Error Occured! =======================" << std::endl;
		std::cout << "Function : int TargetRedcDrawer::initTargetRectDrawer" << std::endl;
		std::cout << "Parameter cv::Mat& _img is empty image!" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	this->img_orig = _img.clone(); // deep copy
	this->img_prev = _img.clone(); // deep copy
	this->img_draw = _img.clone(); // deep copy

	return SUCCESS;
}

/* -----------------------------------------------------------------
Function : callBackFunc

Callback fuction for mouse event. Implement proper action for
each expected mouse event drawing rectangle like left button down,
move, left button up.
----------------------------------------------------------------- */
void TargetRectDrawer::CallBackFunc(int event, int x, int y, int flags)
{
	// When left mouse button is clicked (Normally, next event is moving mouse, EVENT_MOUSEMOVE)
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		// Set is_drawing_rect true
		this->setIsDrawingRect(true);

		cv::Rect new_rect(x, y, 0, 0);
		cv::Scalar new_color(rand() % 256, rand() % 256, rand() % 256);
		std::pair<cv::Rect, cv::Scalar> new_pair = std::make_pair(new_rect, new_color);

		// push new result to the vector
		this->getRectVec().push_back(new_pair);

		// Set start_x and start_y
		this->setStartX(x);
		this->setStartY(y);
	}
	// when mouse is moving (Normally, after left mouse button click, EVENT_LBUTTONDOWN)
	else if (event == cv::EVENT_MOUSEMOVE)
	{
		// when left mouse button is clicked at the same time
		if (getIsDrawingRect())
		{
			int new_width = x - this->start_x;
			int new_height = y - this->start_y;

			if ((new_width * new_height) > 0)
			{
				this->getRectVec().back().first.width = new_width;
				this->getRectVec().back().first.height = new_height;
			}
			else if ((new_width * new_height) < 0)
			{
				this->getRectVec().back().first.x = x;
				this->getRectVec().back().first.width = -new_width;
				this->getRectVec().back().first.height = new_height;
			}

			this->img_draw = this->img_prev.clone();

			cv::Rect new_rect = this->getRectVec().back().first;
			cv::Scalar new_color = this->getRectVec().back().second;

			cv::rectangle(this->img_draw, new_rect, new_color, 2);
			cv::imshow("Tracking System", this->img_draw);
		}
	}
	// when left mouse button is up (Normally, after moving mouse, EVENT_MOUSEMOVE)
	else if (event == cv::EVENT_LBUTTONUP)
	{
		setIsDrawingRect(false);

		cv::Rect& new_rect = this->getRectVec().back().first;  // Want to change some values in new_rect.
		cv::Scalar new_color = this->getRectVec().back().second;

		if (new_rect.width < 0)
		{
			new_rect.x += new_rect.width;
			new_rect.width = -new_rect.width;
		}

		if (new_rect.height < 0)
		{
			new_rect.y += new_rect.height;
			new_rect.height = -new_rect.height;
		}

		cv::rectangle(this->img_draw, new_rect, new_color, 2);

		cv::imshow("Tracking System", this->img_draw);

		// Drawing is done. Make current image as previous image
		this->img_prev = img_draw.clone();
	}
}

/* -----------------------------------------------------------------

Function : WrapperCallBackFunc

Wrapper function for TargetRectDrawer::callBackFunc.
To make mouse call back function(TargetRectDrawer::callBackFunc)
inside the class, static wrapper function(TargetRectDrawer::wrapperCallBackFunc)
is needed.

----------------------------------------------------------------- */
void TargetRectDrawer::wrapperCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	TargetRectDrawer * drawer = (TargetRectDrawer *)userdata;

	// The real call back function
	drawer->CallBackFunc(event, x, y, flags);
}

/* -----------------------------------------------------------------

Function : drawInitRect

Draw initial target rectangle on the window.

----------------------------------------------------------------- */
std::vector<std::pair<cv::Rect, cv::Scalar>>& TargetRectDrawer::drawInitRect(cv::Mat& _mat_img)
{
	int keyboard = 0;  // keyboard input

	TargetRectDrawer * drawer = new TargetRectDrawer(); // This will be passed to callback function, WrapperCallBackFunc														
	drawer->initTargetRectDrawer(_mat_img);				// Initialize

	do
	{
		// Show image
		imshow("Tracking System", drawer->img_draw);

		// Register callback function and get mouse event
		cv::setMouseCallback("Tracking System", TargetRectDrawer::wrapperCallBackFunc, drawer);

		keyboard = cv::waitKey(0);

		// if press ESC key, delete last rectangle
		if (keyboard == ESC)
		{
			// Remove the last rectangle
			drawer->rect_vec.pop_back();

			// Change img_prev and img_draw with the original(with no rectangle) one
			drawer->img_prev = drawer->img_orig.clone();
			drawer->img_draw = drawer->img_orig.clone();

			// Draw rectangle again except the last rectangle
			cout << "Draw rect again" << endl;
			std::for_each(drawer->rect_vec.begin(), drawer->rect_vec.end(), [&](std::pair<cv::Rect, cv::Scalar> _pair) {
				cout << _pair.first << endl;
				cv::rectangle(drawer->img_prev, _pair.first, _pair.second, 2);
				cv::rectangle(drawer->img_draw, _pair.first, _pair.second, 2);
			});
		}
		// if press ENTER key, finish drawing.
	} while (keyboard != ENTER);

	// return drawing result
	return drawer->getRectVec();
}
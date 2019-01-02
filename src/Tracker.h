#pragma once

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define FAIL		-1
#define SUCCESS		1
#define FALSE		0
#define TRUE		1

#define OUT_OF_FRAME	2

#define ENTER	13
#define ESC		27

/* ==========================================================================

Class : SingleTracker

This class is aim to track 'One' target for running time.
'One' SingleTracker object is assigned to 'One' person(or any other object).
In other words, if you are trying to track 'Three' people,
then need to have 'Three' SingleTracker object.

========================================================================== */
class SingleTracker
{
private:
	int		target_id;			// Unique Number for target
	double		confidence;			// Confidence of tracker
	cv::Rect	rect;				// Initial Rectangle for target
	cv::Point	center;				// Current center point of target
	bool		is_tracking_started;// Is tracking started or not? (Is initializing done or not?)
	cv::Scalar	color;				// Box color

public:
	dlib::correlation_tracker tracker;  // Correlation tracker

	/* Member Initializer & Constructor*/
	SingleTracker(int _target_id, cv::Rect _init_rect, cv::Scalar _color)
		: target_id(_target_id), confidence(0), is_tracking_started(false)
	{
		// Exception
		if (_init_rect.area() == 0)
		{
			std::cout << "======================= Error Occured! ======================" << std::endl;
			std::cout << "Function : Constructor of SingleTracker" << std::endl;
			std::cout << "Parameter cv::Rect _init_rect's area is 0" << std::endl;
			std::cout << "=============================================================" << std::endl;
		}
		else
		{
			// Initialize rect and center using _init_rect
			this->setRect(_init_rect);
			this->setCenter(_init_rect);
			this->setColor(_color);
		}
	}

	/* Get Function */
	int			getTargetID() { return this->target_id; }
	cv::Rect	getRect() { return this->rect; }
	cv::Point	getCenter() { return this->center; }
	double		getConfidence() { return this->confidence; }
	bool		getIsTrackingStarted() { return this->is_tracking_started; }
	cv::Scalar	getColor() { return this->color; }

	/* Set Function */
	void setTargetId(int _target_id) { this->target_id = _target_id; }
	void setRect(cv::Rect _rect) { this->rect = _rect; }
	void setRect(dlib::drectangle _drect) { this->rect = cv::Rect(_drect.tl_corner().x(), _drect.tl_corner().y(), _drect.width(), _drect.height()); }
	void setCenter(cv::Point _center) { this->center = _center; }
	void setCenter(cv::Rect _rect) { this->center = cv::Point(_rect.x + (_rect.width) / 2, _rect.y + (_rect.height) / 2); }
	void setCenter(dlib::drectangle _drect) { this->center = cv::Point(_drect.tl_corner().x() + (_drect.width() / 2), _drect.tl_corner().y() + (_drect.height() / 2)); }
	void setConfidence(double _confidence) { this->confidence = _confidence; }
	void setIsTrackingStarted(bool _b) { this->is_tracking_started = _b; }
	void setColor(cv::Scalar _color) { this->color = _color; }

	/* Core Function */
	// Initialize
	int startSingleTracking(cv::Mat _mat_img);

	// Do tracking
	int doSingleTracking(cv::Mat _mat_img);

	// Check the target is inside of the frame
	int isTargetInsideFrame(int _frame_width, int _frame_height);
};

/* ==========================================================================

Class : TrackerManager

TrackerManager is aim to manage vector<std::shared_ptr<SingleTracker>>
for multi-object tracking.
(To make it easy, it's almost same with vector<SigleTracker *>)
So, this class provides insert, find, delete function.

========================================================================== */
class TrackerManager
{
private:
	std::vector<std::shared_ptr<SingleTracker>> tracker_vec; // Vector filled with SingleTracker shared pointer. It is the most important container in this program.

public:
	/* Get Function */
	std::vector<std::shared_ptr<SingleTracker>>& getTrackerVec() { return this->tracker_vec; } // Return reference! not value!

	/* Core Function */
	// Insert new SingleTracker shared pointer into the TrackerManager::tracker_vec
	int insertTracker(cv::Rect _init_rect, cv::Scalar _color, int _target_id);
	int insertTracker(std::shared_ptr<SingleTracker> new_single_tracker);

	// Find SingleTracker in the TrackerManager::tracker_vec using SingleTracker::target_id
	int findTracker(int _target_id);

	// Deleter SingleTracker which has ID : _target_id from TrackerManager::tracker_vec
	int deleteTracker(int _target_id);
};

/* ===================================================================================================

Class : TrackingSystem

TrackingSystem is the highest-ranking manager in this program.
It uses FrameReader class to get the frame images, TrackerManager class for the
smooth tracking. And SingleTracker object will be included in TrackerManager::tracker_vec.
In each of SingleTracker, SingleTracker::startSingleTracking and SingleTracker::doSingleTracking
functios are taking care of tracking each target.
TrackingSystem is using these classes properly and hadling all expected exceptions.

====================================================================================================== */
class TrackingSystem
{
private:
	int				frame_width;	// Frame image width
	int				frame_height;	// Frame image height
	cv::Mat			current_frame;	// Current frame
	std::vector<std::pair<cv::Rect, cv::Scalar>> init_target;

	TrackerManager		manager;	// TrackerManager

public:
	/* Constructor */
	TrackingSystem(){};

	/* Get Function */
	int    getFrameWidth() { return this->frame_width; }
	int    getFrameHeight() { return this->frame_height; }
	cv::Mat   getCurrentFrame() { return this->current_frame; }
	TrackerManager getTrackerManager() { return this->manager; }

	/* Set Function */
	//void   setFramePath(std::string _frame_path) { this->frame_path.assign(_frame_path); }
	void   setFrameWidth(int _frame_width) { this->frame_width = _frame_width; }
	void   setFrameHeight(int _frame_height) { this->frame_height = _frame_height; }
	void   setCurrentFrame(cv::Mat _current_frame) { this->current_frame = _current_frame; }
	void   setInitTarget(std::vector<std::pair<cv::Rect, cv::Scalar>> _init_target) { this->init_target = _init_target; }

	/* Core Function */
	// Initialize TrackingSystem.
	int initTrackingSystem();

	// Start tracking
	int startTracking(cv::Mat& _mat_img);

	// Draw tracking result
	int drawTrackingResult(cv::Mat& _mat_img);

	// Terminate program
	void terminateSystem();
};

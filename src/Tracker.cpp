#include "Tracker.h"

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv.h>

/* ==========================================================================

Class : Util

Many useful but not fundamental functions are implemented in this class.
All functions are static functions so don't need to make class Util object
to use these functions.

========================================================================== */
class Util
{
public:

	/* --------------------------------------------
	Function : cvtRectToRect
	Convert cv::Rect to dlib::drectangle
	----------------------------------------------- */
	static dlib::drectangle cvtRectToDrect(cv::Rect _rect)
	{
		return dlib::drectangle(_rect.tl().x, _rect.tl().y, _rect.br().x - 1, _rect.br().y - 1);
	}


	/* -------------------------------------------------
	Function : cvtMatToArray2d
	convert cv::Mat to dlib::array2d<unsigned char>
	------------------------------------------------- */
	static dlib::array2d<unsigned char> cvtMatToArray2d(cv::Mat _mat) // cv::Mat, not cv::Mat&. Make sure use copy of image, not the original one when converting to grayscale
	{

		//Don't need to use color image in HOG-feature-based tracker
		//Convert color image to grayscale
		if (_mat.channels() == 3)
			cv::cvtColor(_mat, _mat, cv::COLOR_RGB2GRAY);

		//Convert opencv 'MAT' to dlib 'array2d<unsigned char>'
		dlib::array2d<unsigned char> dlib_img;
		dlib::assign_image(dlib_img, dlib::cv_image<unsigned char>(_mat));

		return dlib_img;
	}


	/* -----------------------------------------------------------------
	Function : setRectToImage
	Put all tracking results(new rectangle) on the frame image
	Parameter _rects is stl container(such as vector..) filled with
	cv::Rect
	----------------------------------------------------------------- */
	template <typename Container>
	static void setRectToImage(cv::Mat& _mat_img, Container _rects)
	{
		std::for_each(_rects.begin(), _rects.end(), [&_mat_img](cv::Rect rect) {
			cv::rectangle(_mat_img, rect, cv::Scalar(0, 0, 255));
		});
	}
};

/* ---------------------------------------------------------------------------------

Function : startSingleTracking

Initialize dlib::correlation_tracker tracker using dlib::start_track function

---------------------------------------------------------------------------------*/
int SingleTracker::startSingleTracking(cv::Mat _mat_img)
{
	// Exception
	if (_mat_img.empty())
	{
		std::cout << "====================== Error Occured! =======================" << std::endl;
		std::cout << "Function : int SingleTracker::startSingleTracking" << std::endl;
		std::cout << "Parameter cv::Mat& _mat_img is empty image!" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	// Convert _mat_img to dlib::array2d<unsigned char>
	dlib::array2d<unsigned char> dlib_frame = Util::cvtMatToArray2d(_mat_img);

	// Convert SingleTracker::rect to dlib::drectangle
	dlib::drectangle dlib_rect = Util::cvtRectToDrect(this->getRect());

	// Initialize SingleTracker::tracker
	this->tracker.start_track(dlib_frame, dlib_rect);
	this->setIsTrackingStarted(true);

	return SUCCESS;
}

/*---------------------------------------------------------------------------------

Function : isTargetInsideFrame

Check the target is inside the frame
If the target is going out of the frame, need to SingleTracker stop that target.

---------------------------------------------------------------------------------*/
int SingleTracker::isTargetInsideFrame(int _frame_width, int _frame_height)
{
	int cur_x = this->getCenter().x;
	int cur_y = this->getCenter().y;

	bool is_x_inside = ((0 <= cur_x) && (cur_x < _frame_width));
	bool is_y_inside = ((0 <= cur_y) && (cur_y < _frame_height));

	if (is_x_inside && is_y_inside)
		return TRUE;
	else
		return FALSE;
}

/* ---------------------------------------------------------------------------------

Function : doSingleTracking

Track 'one' target specified by SingleTracker::rect in a frame.
(It means that you need to call doSingleTracking once per a frame)
SingleTracker::rect is initialized to the target position in the constructor of SingleTracker
Using correlation_tracker in dlib, start tracking 'one' target

--------------------------------------------------------------------------------- */
int SingleTracker::doSingleTracking(cv::Mat _mat_img)
{
	//Exception
	if (_mat_img.empty())
	{
		std::cout << "====================== Error Occured! ======================= " << std::endl;
		std::cout << "Function : int SingleTracker::doSingleTracking" << std::endl;
		std::cout << "Parameter cv::Mat& _mat_img is empty image!" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	// Convert _mat_img to dlib::array2d<unsigned char>
	dlib::array2d<unsigned char> dlib_img = Util::cvtMatToArray2d(_mat_img);

	// Track using dlib::update function
	double confidence = this->tracker.update(dlib_img);

	// New position of the target
	dlib::drectangle updated_rect = this->tracker.get_position();

	// Update variables(center, rect, confidence)
	this->setCenter(updated_rect);
	this->setRect(updated_rect);
	this->setConfidence(confidence);

	return SUCCESS;
}




/* -------------------------------------------------------------------------

Function : insertTracker

Create new SingleTracker object and insert it to the vector.
If you are about to track new person, need to use this function.

------------------------------------------------------------------------- */

int TrackerManager::insertTracker(cv::Rect _init_rect, cv::Scalar _color, int _target_id)
{
	// Exceptions
	if (_init_rect.area() == 0)
	{
		std::cout << "======================= Error Occured! ====================== " << std::endl;
		std::cout << "Function : int SingleTracker::initTracker" << std::endl;
		std::cout << "Parameter cv::Rect _init_rect's area is 0" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	// if _target_id is already exists
	int result_idx = findTracker(_target_id);

	if (result_idx != FAIL)
	{
		std::cout << "======================= Error Occured! ======================" << std::endl;
		std::cout << "Function : int SingleTracker::initTracker" << std::endl;
		std::cout << "_target_id already exists!" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	// Create new SingleTracker object and insert it to the vector
	std::shared_ptr<SingleTracker> new_tracker(new SingleTracker(_target_id, _init_rect, _color));
	this->tracker_vec.push_back(new_tracker);

	return SUCCESS;
}

// Overload of insertTracker
int TrackerManager::insertTracker(std::shared_ptr<SingleTracker> new_single_tracker)
{
	//Exception
	if (new_single_tracker == nullptr)
	{
		std::cout << "======================== Error Occured! ===================== " << std::endl;
		std::cout << "Function : int TrackerManager::insertTracker" << std::endl;
		std::cout << "Parameter shared_ptr<SingleTracker> new_single_tracker is nullptr" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	// if _target_id is already exists
	int result_idx = findTracker(new_single_tracker.get()->getTargetID());
	if (result_idx != FAIL)
	{
		std::cout << "====================== Error Occured! =======================" << std::endl;
		std::cout << "Function : int SingleTracker::insertTracker" << std::endl;
		std::cout << "_target_id already exists!" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}

	// Insert new SingleTracker object into the vector
	this->tracker_vec.push_back(new_single_tracker);

	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : findTracker

Find SingleTracker object which has ID : _target_id in the TrackerManager::tracker_vec
If success to find return that iterator, or return TrackerManager::tracker_vec.end()

----------------------------------------------------------------------------------- */
int TrackerManager::findTracker(int _target_id)
{
	auto target = find_if(tracker_vec.begin(), tracker_vec.end(), [&, _target_id](std::shared_ptr<SingleTracker> ptr) -> bool {
		return (ptr.get() -> getTargetID() == _target_id);
	});

	if (target == tracker_vec.end())
		return FAIL;
	else
		return target - tracker_vec.begin();
}

/* -----------------------------------------------------------------------------------

Function : deleteTracker

Delete SingleTracker object which has ID : _target_id in the TrackerManager::tracker_vec

----------------------------------------------------------------------------------- */
int TrackerManager::deleteTracker(int _target_id)
{
	int result_idx = this->findTracker(_target_id);

	if (result_idx == FAIL)
	{
		std::cout << "======================== Error Occured! =====================" << std::endl;
		std::cout << "Function : int TrackerManager::deleteTracker" << std::endl;
		std::cout << "Cannot find given _target_id" << std::endl;
		std::cout << "=============================================================" << std::endl;

		return FAIL;
	}
	else
	{
		// Memory deallocation
		this->tracker_vec[result_idx].reset();

		// Remove SingleTracker object from the vector
		this->tracker_vec.erase(tracker_vec.begin() + result_idx);

		std::cout << "========================== Notice! ==========================" << std::endl;
		std::cout << "Target ID : " << _target_id << " is going out of the frame." << std::endl;
		std::cout << "Target ID : " << _target_id << " is erased!" << std::endl;
		std::cout << "=============================================================" << std::endl;		return SUCCESS;
	}
}

/* -----------------------------------------------------------------------------------

Function : initTrackingSystem()

Insert multiple SingleTracker objects to the manager.tracker_vec in once.
If you want multi-object tracking, call this function just for once like

vector<cv::Rect> rects;
// Insert all rects into the vector

vector<int> ids;
// Insert all target_ids into the vector

initTrackingSystem(ids, rects)

Then, the system is ready to track multiple targets.

----------------------------------------------------------------------------------- */

int TrackingSystem::initTrackingSystem()
{
	int index = 0;
	for( auto && i : init_target){
		if (this->manager.insertTracker(i.first, i.second, index) == FAIL)
		{
			std::cout << "====================== Error Occured! =======================" << std::endl;
			std::cout << "Function : int TrackingSystem::initTrackingSystem" << std::endl;
			std::cout << "Cannot insert new SingleTracker object to the vector" << std::endl;
			std::cout << "=============================================================" << std::endl;
			return FAIL;
		}
		index++;
	}
	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : startTracking()

Track all targets.
You don't need to give target id for tracking.
This function will track all targets.

----------------------------------------------------------------------------------- */
int TrackingSystem::startTracking(cv::Mat& _mat_img)
{
	// Check the image is empty
	if (_mat_img.empty())
	{
		std::cout << "======================= Error Occured! ======================" << std::endl;
		std::cout << "Function : int TrackingSystem::startTracking" << std::endl;
		std::cout << "Input image is empty" << std::endl;
		std::cout << "=============================================================" << std::endl;
		return FAIL;
	}

	// Convert _mat_img to dlib::array2d<unsigned char>
	dlib::array2d<unsigned char> dlib_cur_frame = Util::cvtMatToArray2d(_mat_img);

	// For all SingleTracker, do SingleTracker::startSingleTracking.
	// Function startSingleTracking should be done before doSingleTracking
	std::for_each(manager.getTrackerVec().begin(), manager.getTrackerVec().end(), [&](std::shared_ptr<SingleTracker> ptr) {
		if (!(ptr.get()->getIsTrackingStarted()))
		{
			ptr.get()->startSingleTracking(_mat_img);
			ptr.get()->setIsTrackingStarted(true);
		}
	});

	vector<std::thread> thread_pool;
	
	// Multi thread
	std::for_each(manager.getTrackerVec().begin(), manager.getTrackerVec().end(), [&](std::shared_ptr<SingleTracker> ptr) {
		thread_pool.emplace_back([ptr, &_mat_img]() { 
		 ptr.get()->doSingleTracking(_mat_img); 
		});
	});

	for (int i = 0; i < thread_pool.size(); i++)
		thread_pool[i].join();

	// If target is going out of the frame, delete that tracker.
	/*std::for_each(manager.getTrackerVec().begin(), manager.getTrackerVec().end(), [&](std::shared_ptr<SingleTracker> ptr) {
		if (ptr.get()->isTargetInsideFrame(this->getFrameWidth(), this->getFrameHeight()) == FALSE)
		{
			int target_id = ptr.get()->getTargetID();
			int a = manager.deleteTracker(target_id);
			std::cout << a << std::endl;
			std::cout << "========================== HOLA ==========================" << std::endl;
			std::cout << "Function int TrackingSystem::startTracking" << std::endl;
			std::cout << "Target ID : " << target_id << " is going out of the frame." << std::endl;
			std::cout << "Target ID : " << target_id << " is erased!" << std::endl;
			std::cout << "=============================================================" << std::endl;
			break;
		}
	});*/
	std::vector<int> tracker_erase;
	for(auto && i: manager.getTrackerVec()){
		if (i->isTargetInsideFrame(this->getFrameWidth(), this->getFrameHeight()) == FALSE)
		{
			int target_id = i.get()->getTargetID();
			tracker_erase.push_back(target_id);
		}
	}
	
	for(auto && i : tracker_erase){
		int a = manager.deleteTracker(i);
	}


	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : drawTrackingResult

Deallocate all memory and close the program.

----------------------------------------------------------------------------------- */
int TrackingSystem::drawTrackingResult(cv::Mat& _mat_img)
{
	TrackerManager manager = this->getTrackerManager();

	// Exception
	if (manager.getTrackerVec().size() == 0)
	{
		std::cout << "======================= Error Occured! ======================" << std::endl;
		std::cout << "Function : int TrackingSystem::drawTrackingResult" << std::endl;
		std::cout << "Nothing to draw" << std::endl;
		std::cout << "=============================================================" << std::endl;
		return FAIL;
	}

	std::for_each(manager.getTrackerVec().begin(), manager.getTrackerVec().end(), [&_mat_img](std::shared_ptr<SingleTracker> ptr) {
		// Draw all rectangles
		cv::rectangle(_mat_img, ptr.get()->getRect(), ptr.get()->getColor(), 1);

		cv::String text(std::string("Target ID : ") + std::to_string(ptr.get()->getTargetID()));
		cv::Point text_pos = ptr.get()->getRect().tl();
		text_pos.x = text_pos.x - 10;
		text_pos.y = text_pos.y - 5;

		// Put all target ids
		cv::putText(_mat_img,
			text,
			text_pos,
			cv::FONT_HERSHEY_TRIPLEX,
			1,
			ptr.get()->getColor(),
			2);
	});

	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : terminateSystem

Draw rectangle around the each target and put target id on rectangle.

----------------------------------------------------------------------------------- */
void TrackingSystem::terminateSystem()
{
	std::vector<std::shared_ptr<SingleTracker>> remaining_tracker = manager.getTrackerVec();

	// Memory deallocation
	std::for_each(remaining_tracker.begin(), remaining_tracker.end(),
		[](std::shared_ptr<SingleTracker> ptr) { ptr.reset(); });

	std::cout << "Close Tracking System..." << std::endl;
}

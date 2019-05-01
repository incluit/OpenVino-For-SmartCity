#include "Tracker.h"

/* ==========================================================================

Class : Util

Many useful but not fundamental functions are implemented in this class.
All functions are static functions so don't need to make class Util object
to use these functions.

========================================================================== */
class Util
{
public:
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

Function : calcAvgPos

Calculate positions as an average of last n_frames positions. This reduces noise
from detection stage.

---------------------------------------------------------------------------------*/
void SingleTracker::calcAvgPos()
{
	const int full = 5;
	double delta_x = 0;
	double delta_y = 0;
	cv::Point2f avg = cv::Point2f(0,0);

	if (this->c_q.size() == full) { // Start calculating the averages when the queue is full.
		for (int i = 0; i<5; i++) {
		avg = avg + (cv::Point2f)this->c_q[i];
		}
		avg = avg / 5.0;
		this->saveAvgPos(avg);
	}
}

/* ---------------------------------------------------------------------------------

Function : calcVel

Calculate velocity as an average of last n_frames frames (dX, dY).

---------------------------------------------------------------------------------*/
void SingleTracker::calcVel()
{
	const int full = 5;
	double delta_x = 0;
	double delta_y = 0;
	cv::Point2f avgvel;

	int lim = std::min(full,(int)this->avg_pos.size()-1); // Wait for one frame to calculate speed
	if (lim > 1) {
	for (int i = 0; i < lim; i++) {
		delta_x = delta_x + (this->avg_pos[i].x - this->avg_pos[i+1].x);
		delta_y = delta_y + (this->avg_pos[i].y - this->avg_pos[i+1].y);
	}
	delta_x = delta_x / lim;
	delta_y = delta_y / lim;
	avgvel = cv::Point2f(delta_x,delta_y);
	this->setVel((cv::Point2f)this->getCenter() + avgvel);
	this->saveLastVel(this->getVel_X(), this->getVel_Y(), this->getModVel());
	} else {
	this->setVel((cv::Point2f)this->getCenter());
	}
}

/* ---------------------------------------------------------------------------------

Function : calcAcc

Calculate acceleration as an average of last n_frames frames (dX, dY).

---------------------------------------------------------------------------------*/
void SingleTracker::calcAcc()
{
	const int full = 5;
	double delta_x = 0;
	double delta_y = 0;
	cv::Point2f acc;

	int lim = std::min(full,(int)this->v_q.size()-1); // Wait for one frame to get 2 speeds
	if (lim > 1) { // We need 2 speeds
	for (int i = 0; i < lim; i++) {
		delta_x = delta_x + ((this->v_x_q[i]+1)*1000.0/(double)(this->avg_pos[i].y+10) - (this->v_x_q[i+1]+1)*1000.0/(double)(this->avg_pos[i+1].y+10));
		delta_y = delta_y + ((this->v_y_q[i]+1)*1000.0/(double)(this->avg_pos[i].y+10) - (this->v_y_q[i+1]+1)*1000.0/(double)(this->avg_pos[i+1].y+10));
	}
	delta_x = delta_x / lim;
	delta_y = delta_y / lim;
	acc = cv::Point2f(delta_x,delta_y);
	this->setAcc((cv::Point2f)this->getCenter() + acc);
	this->saveLastAcc(this->getAcc_X(), this->getAcc_Y(), this->getModAcc());
	} else {
	this->setAcc((cv::Point2f)this->getCenter());
	}

}

/* ---------------------------------------------------------------------------------

Function : startSingleTracking

Initialize dlib::correlation_tracker tracker using dlib::start_track function

---------------------------------------------------------------------------------*/

int SingleTracker::startSingleTracking(cv::Mat _mat_img)
{
	// Exception
	if (_mat_img.empty())
	{
		BOOST_LOG_TRIVIAL(error) << "====================== Error Occured! =======================";
		BOOST_LOG_TRIVIAL(error) << "Function : int SingleTracker::startSingleTracking";
		BOOST_LOG_TRIVIAL(error) << "Parameter cv::Mat& _mat_img is empty image!";
		BOOST_LOG_TRIVIAL(error) << "=============================================================";

		return FAIL;
	}
	this->setIsTrackingStarted(true);

	return SUCCESS;
}

/*---------------------------------------------------------------------------------

Function : isTargetInsideFrame

Check the target is inside the frame
If the target is going out of the frame, need to SingleTracker stop that target.

---------------------------------------------------------------------------------*/
int SingleTracker::isTargetInsideFrame(int _frame_width, int _frame_height, cv::Mat *mask)
{
	int cur_x = this->getCenter().x;
	int cur_y = this->getCenter().y;

	if( mask != nullptr ){
		cv::Mat gray;
		cv::cvtColor(*mask, gray, cv::COLOR_BGR2GRAY);
		cv::Canny(gray, gray, 100, 200, 3);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours( gray, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		std::vector<double> r_values;
		for(auto && i : contours){
			double aux;
			aux = cv::pointPolygonTest(i,cv::Point2f(cur_x,cur_y),false);
			r_values.push_back(aux);
		}
		bool band = FALSE;
		for(auto && i :r_values){
			if (i == 1){
				band = TRUE;
			} else {
				band = FALSE;
			}
		}
		return band;
	}

	bool is_x_inside = ((0 <= cur_x) && (cur_x < _frame_width));
	bool is_y_inside = ((0 <= cur_y) && (cur_y < _frame_height));

	if (is_x_inside && is_y_inside)
		return TRUE;
	else
		return FALSE;
}

/* -----------------------------------------------------------------------------------

Function : markForDeletion(std::vector<std::pair<cv::Rect, int>> rois)

Mark trackers to delete.

----------------------------------------------------------------------------------- */

int SingleTracker::markForDeletion()
{
	const int frames = 12; // Arbitrary numbers, adjust if needed
	const double min_vel = 0.01*this->rect.area();

	if (this->no_update_counter >= frames && this->modvel < min_vel)
		this->to_delete = true;

	return SUCCESS;
}

int isInsideMask(cv::Mat * mask, cv::Point2f * pos)
{
	cv::Mat gray;
	cv::cvtColor(*mask, gray, cv::COLOR_BGR2GRAY);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours( gray, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	std::vector<double> r_values;
	for(auto && i : contours){
		double aux;
		aux = cv::pointPolygonTest(i,*pos,false);
		r_values.push_back(aux);
	}
	bool band = FALSE;
	for(auto && i :r_values){
		if (i == 1){
			band = TRUE;
		} else {
			band = FALSE;
		}
	}
	return band;
}

/* ---------------------------------------------------------------------------------

Function : assignArea

Assign area to each object based on the areas drawn by the user.

--------------------------------------------------------------------------------- */
void SingleTracker::assignArea(std::vector<cv::Mat>* mask_sw, std::vector<cv::Mat>* mask_cw, std::vector<std::pair<cv::Mat, int>>* mask_str )
{
	char areas = 0;
	cv::Point2f pos = (cv::Point2f)this->getBottom();
	int ret = 0;
	cv::Mat* mask_ptr = nullptr;

	if (mask_sw != nullptr) {
		for (auto && mask: *mask_sw) {
			ret = isInsideMask(&mask, &pos);
			if (ret) {
				areas = areas | FLAG_SW;
			}
		}
	}
	if (mask_cw != nullptr) {
		for (auto && mask: *mask_cw) {
			ret = isInsideMask(&mask, &pos);
			if (ret) {
				areas = areas | FLAG_CW;
				mask_ptr = &mask;
			}
		}
	}
	if (mask_str != nullptr) {
		for (auto mask: *mask_str) {
			ret = isInsideMask(&mask.first, &pos);
			if (ret) {
				areas = areas | FLAG_STR;
			}
		}
	}

	this->setArea(areas, mask_ptr);
}
/* ---------------------------------------------------------------------------------

Function : doSingleTracking

Track 'one' target specified by SingleTracker::rect in a frame.
(It means that you need to call doSingleTracking once per a frame)
SingleTracker::rect is initialized to the target position in the constructor of SingleTracker
Using correlation_tracker in dlib, start tracking 'one' target

--------------------------------------------------------------------------------- */
int SingleTracker::doSingleTracking(cv::Mat* _mat_img, std::vector<cv::Mat>* mask_sw, 
								std::vector<cv::Mat>* mask_cw, std::vector<std::pair<cv::Mat, int>>* mask_str,
								Pipe* buffer, int* totalFrames, bool dbEnable)
{
	//Exception
	if (_mat_img -> empty())
	{
		BOOST_LOG_TRIVIAL(error) << "====================== Error Occured! ======================= ";
		BOOST_LOG_TRIVIAL(error) << "Function : int SingleTracker::doSingleTracking";
		BOOST_LOG_TRIVIAL(error) << "Parameter cv::Mat& _mat_img is empty image!";
		BOOST_LOG_TRIVIAL(error) << "=============================================================";

		return FAIL;
	}
	cv::Rect updated_rect = this -> rect;
	if (!this->getUpdateFromDetection()) {
		this->setCenter(cv::Point2f(this->vel.x, this->vel.y));
		updated_rect.x = updated_rect.x + vel_x;
		updated_rect.y = updated_rect.y + vel_y;
	}
	this->setUpdateFromDetection(false);
	// New position of the target
	// Update variables(center, rect, confidence)
	this->setCenter(updated_rect);
	this->setRect(updated_rect);
	//this->setConfidence(confidence);
	this->saveLastCenter(this->getCenter());
	this->assignArea(mask_sw, mask_cw, mask_str);
	this->calcAvgPos();
	this->calcVel();
	this->calcAcc();
	this->no_update_counter++;
	this->markForDeletion();

#ifdef ENABLED_DB
	if(dbEnable){
		PipeItem document;
		document.frame = *totalFrames; 
		document.Id = this->getTargetID();
		document.vel_x = this->getVelX_q()[0];
		document.vel_y = this->getVelY_q()[0]; 
		document.vel = this->getVel_q()[0]; 
		document.acc_x = this->getAccX_q()[0];
		document.acc_y = this->getAccY_q()[0]; 
		document.acc = this->getAcc_q()[0]; 
		document.objectClass = getLabelStr(this->getLabel());
		buffer->push_back(document);
	}
#endif
	
	return SUCCESS;
}

/* -------------------------------------------------------------------------

Function : insertTracker

Create new SingleTracker object and insert it to the vector.
If you are about to track new person, need to use this function.

------------------------------------------------------------------------- */

int TrackerManager::insertTracker(cv::Rect* _init_rect, cv::Scalar* _color, int _target_id, int _label, bool update, std::string *last_event, bool* dbEnable, int* totalFrames, Pipe* buffer)
{
	// Exceptions
	if (_init_rect->area() == 0)
	{
		BOOST_LOG_TRIVIAL(error) << "======================= Error Occured! ====================== " << "\n"
					 << "Function : int SingleTracker::initTracker" << "\n"
					 << "Parameter cv::Rect _init_rect's area is 0" << "\n"
					 << "=============================================================";

		return FAIL;
	}

	// if _target_id already exists
	int result_idx = findTrackerByID(_target_id);
	// Create new SingleTracker object and insert it to the vector
	std::shared_ptr<SingleTracker> new_tracker(new SingleTracker(_target_id, *_init_rect, *_color, _label));

	if (result_idx != FAIL)	{
		if (!update) {
			BOOST_LOG_TRIVIAL(error) << "======================= Error Occured! ======================" << "\n"
						 << "Function : int SingleTracker::initTracker" << "\n"
						 << "_target_id already exists!" << "\n"
						 << "=============================================================";

			return FAIL;
		} else {
			this->tracker_vec[result_idx]->setCenter(new_tracker->getCenter());
			this->tracker_vec[result_idx]->setRect(*_init_rect);
			this->tracker_vec[result_idx]->setUpdateFromDetection(update);
			this->tracker_vec[result_idx]->setNoUpdateCounter(0);
			if (tracker_vec[result_idx]->getLabel() == LABEL_UNKNOWN) {
				this->tracker_vec[result_idx]->setLabel(_label);
				this->tracker_vec[result_idx]->setColor(*_color);
			}
		}
	} else {
		this->tracker_vec.push_back(new_tracker);
		this->id_list = _target_id + 1; // Next ID
		
		std::string a,b,c,d,aux_str;
#ifdef ENABLED_DB
		if(*dbEnable){
			PipeItem document;
			document.Id = this->id_list-1;
			document.frame = *totalFrames;
			document.event = "Start being tracked";
			document.objectClass = getLabelStr(_label);
			buffer -> push_back(document); 
		}
#endif
		a = "========================== Notice! ==========================";
		b = "Target ID : " + std::to_string(this->id_list-1) + " is now been tracked";
		BOOST_LOG_TRIVIAL(info) << b;
		c = "=============================================================";
		aux_str = a + '\n' + b + '\n'+ c + '\n';
		*last_event = aux_str;
	}

	return SUCCESS;
}

// Overload of insertTracker
int TrackerManager::insertTracker(std::shared_ptr<SingleTracker> new_single_tracker, bool update)
{
	//Exception
	if (new_single_tracker == nullptr)
	{
		BOOST_LOG_TRIVIAL(error) << "======================== Error Occured! ===================== " << "\n"
					 << "Function : int TrackerManager::insertTracker" << "\n"
					 << "Parameter shared_ptr<SingleTracker> new_single_tracker is nullptr" << "\n"
					 << "=============================================================";

		return FAIL;
	}

	// if _target_id already exists
	int result_idx = findTrackerByID(new_single_tracker.get()->getTargetID());
	if (result_idx != FAIL) {
		if (!update) {
			BOOST_LOG_TRIVIAL(error) << "====================== Error Occured! =======================" << "\n"
						 << "Function : int SingleTracker::insertTracker" << "\n"
						 << "_target_id already exists!" << "\n"
						 << "=============================================================";

			return FAIL;
		} else {
			this->tracker_vec[result_idx]->setCenter(new_single_tracker->getCenter());
			this->tracker_vec[result_idx]->setRect(new_single_tracker->getRect());
			this->tracker_vec[result_idx]->setUpdateFromDetection(update);
			this->tracker_vec[result_idx]->setNoUpdateCounter(0);
		}
	} else {
		// Insert new SingleTracker object into the vector
		this->tracker_vec.push_back(new_single_tracker);
		this->id_list = new_single_tracker.get()->getTargetID() + 1; //Next ID

	}

	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : findTrackerByID

Find SingleTracker object which has ID : _target_id in the TrackerManager::tracker_vec
If success to find return that iterator, or return TrackerManager::tracker_vec.end()

----------------------------------------------------------------------------------- */
int TrackerManager::findTrackerByID(int _target_id)
{
	auto target = find_if(tracker_vec.begin(), tracker_vec.end(), [&_target_id](std::shared_ptr<SingleTracker> ptr) -> bool {
		return (ptr.get() -> getTargetID() == _target_id);
	});

	if (target == tracker_vec.end())
		return FAIL;
	else
		return target - tracker_vec.begin();
}

/* -----------------------------------------------------------------------------------

Function : findTracker

Find SingleTracker object in the TrackerManager::tracker_vec
If success to find return that index, or return new index if no coincidence

----------------------------------------------------------------------------------- */
int TrackerManager::findTracker(cv::Rect rect, int label)
{
	double max_overlap_thresh = 0.75;
	double dist_thresh = rect.height*rect.width>>1; // Pixels^2 -> adjust properly (maybe a proportion of the img size?)
	std::vector<std::shared_ptr<SingleTracker>> selection;
	std::shared_ptr<SingleTracker> best = NULL;
	double min_distance = (rect.height*rect.width)+10; // Init bigger than threshold
	int index;
	bool new_object = true;
	std::vector<double> areas;

	for(auto && s_tracker: this->getTrackerVec()) {
		double in_area = (s_tracker.get()->getRect() & rect).area();
		double max_per_area = std::max(in_area / s_tracker.get()->getRect().area(), in_area/rect.area());
		areas.push_back(max_per_area);
		if ( max_per_area > max_overlap_thresh && (s_tracker->getLabel() == label || s_tracker->getLabel() == LABEL_UNKNOWN) ) {
			selection.push_back(s_tracker);
		}
	}

	for (auto && s_tracker: selection) {
		cv::Point n_center = cv::Point(rect.x + (rect.width) / 2, rect.y + (rect.height) / 2);
		cv::Point diff = s_tracker.get()->getCenter() - n_center;
		double distance = diff.x*diff.x + diff.y*diff.y;
		if ((best == NULL && distance < dist_thresh) || (best != NULL && distance < min_distance)) {
			min_distance = distance;
			best = s_tracker;
		}
	}

	for (auto && area: areas) {
		if (area > 0.2) {
			new_object = false;
			break;
		}
	}

	if ( best == NULL && new_object ) {
		index = this->getNextID();
	} else if ( best != NULL ) {
		index = best.get()->getTargetID();
	} else if (!new_object) {
		index = -1;
	}

	return index;
}
/* -----------------------------------------------------------------------------------

Function : getTrackerLabel

find tracker label
----------------------------------------------------------------------------------- */

int TrackerManager::getTrackerLabel(int result_idx){

	return this->tracker_vec[result_idx]->getLabel();//Could potentially have a bug

}


/* -----------------------------------------------------------------------------------

Function : deleteTracker

Delete SingleTracker object which has ID : _target_id in the TrackerManager::tracker_vec

----------------------------------------------------------------------------------- */
int TrackerManager::deleteTracker(int _target_id, std::string *last_event, bool* dbEnable, int* totalFrames, Pipe* buffer)
{
	int result_idx = this->findTrackerByID(_target_id);
#ifdef ENABLED_DB
	if(*dbEnable){
		PipeItem document;
		document.Id = _target_id;
		document.frame = *totalFrames;
		document.event = "Stop being tracked";
		document.objectClass = getLabelStr(this->getTrackerLabel(result_idx));
		buffer -> push_back(document); 
	}
#endif

	if (result_idx == FAIL)
	{
		BOOST_LOG_TRIVIAL(error) << "======================== Error Occured! =====================";
		BOOST_LOG_TRIVIAL(error) << "Function : int TrackerManager::deleteTracker";
		BOOST_LOG_TRIVIAL(error) << "Cannot find given _target_id";
		BOOST_LOG_TRIVIAL(error) << "=============================================================";

		return FAIL;
	}
	else
	{
		// Memory deallocation
		this->tracker_vec[result_idx].reset();

		// Remove SingleTracker object from the vector
		this->tracker_vec.erase(tracker_vec.begin() + result_idx);



		std::string a,b,c,d,aux_str;

		a = "========================== Notice! ==========================";
		b = "Target ID : " + std::to_string(_target_id) + " is going out of the frame";
		BOOST_LOG_TRIVIAL(info) << b;
		c = "Target ID : " + std::to_string(_target_id) + " is erased!";
		BOOST_LOG_TRIVIAL(info) << c;
		d = "=============================================================";
		aux_str = a + '\n' + b + '\n' + c + '\n' + d + '\n';
		*last_event = aux_str;
		return SUCCESS;
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
	cv::Scalar color = COLOR_UNKNOWN;
	int label = LABEL_UNKNOWN;

	for( auto && i : this->init_target){
		color = getLabelColor(i.second);
		label = i.second;
		if ((double)i.first.area()/(double)(getFrameWidth()*getFrameHeight()) < 0.009 && label == LABEL_CAR)
			continue;
		if (this->manager.insertTracker(&i.first, &color, index, label, false, this->last_event, &this->dbEnable,&this->totalFrames, &this->buffer_events) == FAIL)
		{
			BOOST_LOG_TRIVIAL(error) << "====================== Error Occured! =======================";
			BOOST_LOG_TRIVIAL(error) << "Function : int TrackingSystem::initTrackingSystem";
			BOOST_LOG_TRIVIAL(error) << "Cannot insert new SingleTracker object to the vector";
			BOOST_LOG_TRIVIAL(error) << "=============================================================";
			return FAIL;
		}
		index++;
	}
	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : updateTrackingSystem(std::vector<std::pair<cv::Rect, int>> rois)

Insert new multiple SingleTracker objects to the manager.tracker_vec.
If you want multi-object tracking, call this function just for once like

vector<cv::Rect> rects;
// Insert all rects into the vector

vector<int> ids;
// Insert all target_ids into the vector

initTrackingSystem(ids, rects)

Then, the system is ready to track new targets.

----------------------------------------------------------------------------------- */

int TrackingSystem::updateTrackingSystem(std::vector<std::pair<cv::Rect, int>> updated_results)
{
	cv::Scalar color = COLOR_UNKNOWN;
	int label = LABEL_UNKNOWN;

	//Update init_target to detect new objects
	//this->updated_target = updated_results;

	for( auto && i : updated_results){
		int index;
		color = getLabelColor(i.second);
		label = i.second;
		if ((double)i.first.area()/(double)(getFrameWidth()*getFrameHeight()) < 0.009 && label == LABEL_CAR)
			continue;
		index = this->manager.findTracker(i.first, label);
		if ( index != -1 && this->manager.insertTracker(&i.first, &color, index, label, true,this->last_event, &this->dbEnable,&this->totalFrames, &this->buffer_events) == FAIL ) {
				BOOST_LOG_TRIVIAL(error) << "====================== Error Occured! =======================";
				BOOST_LOG_TRIVIAL(error) << "Function : int TrackingSystem::updateTrackingSystem";
				BOOST_LOG_TRIVIAL(error) << "Sth went wrong";
				BOOST_LOG_TRIVIAL(error) << "=============================================================";
				return FAIL;
		}
	}
#ifdef ENABLED_DB
	this->dbWrite(&this->events, &this->buffer_events);
#endif
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
		BOOST_LOG_TRIVIAL(error) << "======================= Error Occured! ======================";
		BOOST_LOG_TRIVIAL(error) << "Function : int TrackingSystem::startTracking";
		BOOST_LOG_TRIVIAL(error) << "Input image is empty";
		BOOST_LOG_TRIVIAL(error) << "=============================================================";
		return FAIL;
	}

	// For all SingleTracker, do SingleTracker::startSingleTracking.
	// Function startSingleTracking should be done before doSingleTracking
	std::for_each(manager.getTrackerVec().begin(), manager.getTrackerVec().end(), [&_mat_img](std::shared_ptr<SingleTracker> ptr) {
		if (!(ptr.get()->getIsTrackingStarted()))
		{
			ptr.get()->startSingleTracking(_mat_img);
			ptr.get()->setIsTrackingStarted(true);
		}
	});

	std::vector<std::thread> thread_pool;

	std::vector<cv::Mat>* mask_sw = this->mask_sidewalks;

	std::vector<cv::Mat>* mask_cw = this->mask_crosswalks;

	std::vector<std::pair<cv::Mat, int>>* mask_str = this->mask_streets;

	Pipe* buffer = &this->buffer_tracker;

	int* tFrames = &this->totalFrames;

	bool dbEn = this->dbEnable;

	// Multi thread
	std::for_each(manager.getTrackerVec().begin(), manager.getTrackerVec().end(), [&thread_pool, &_mat_img, &mask_sw, &mask_cw, &mask_str, &buffer, &tFrames, &dbEn](std::shared_ptr<SingleTracker> ptr) {
		thread_pool.emplace_back([ptr, &_mat_img, &mask_sw, &mask_cw, &mask_str, &buffer, &tFrames, dbEn]() {
		 ptr.get()->doSingleTracking(&_mat_img, mask_sw, mask_cw, mask_str, buffer, tFrames, dbEn);
		});
	});

	for (int i = 0; i < thread_pool.size(); i++)
		thread_pool[i].join();

#ifdef ENABLED_DB
	std::thread t1(&TrackingSystem::dbWrite, this, &this->tracker, &this->buffer_tracker);
#endif
	std::vector<int> tracker_erase;
	for(auto && i: manager.getTrackerVec()) {
		if (i->isTargetInsideFrame(this->getFrameWidth(), this->getFrameHeight(), this->mask) == FALSE || i->getDelete()) {
			int target_id = i.get()->getTargetID();
			tracker_erase.push_back(target_id);
		}
	}

	for(auto && i : tracker_erase)
		int a = manager.deleteTracker(i,this->last_event, &this->dbEnable, &this->totalFrames, &this->buffer_events);

	if(this -> mask_crosswalks != nullptr){
		for(auto && crosswalk: *this->mask_crosswalks) {
			bool person_cw = false;
			bool car_cw = false;
			for(auto && i: manager.getTrackerVec()) {
				if (i->getAreas().second == &crosswalk && &crosswalk != nullptr) {
					if (i->getLabel() == LABEL_PERSON) {
						person_cw = true;
					}
					if (i->getLabel() == LABEL_CAR) {
						car_cw = true;
					}
					if (car_cw && person_cw) {
						cv::Mat roi(cv::Size(_mat_img.cols, _mat_img.rows), _mat_img.type(), cv::Scalar(0,0,255));
						cv::bitwise_and(roi,crosswalk,roi);
						this->saveCrosswalk(roi);
					}
				}
			}
		}
	}

#ifdef ENABLED_DB
	this->dbWrite(&this->events, &this->buffer_events);
	t1.join();
#endif

	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : drawTrackingResult

Deallocate all memory and close the program.

----------------------------------------------------------------------------------- */
int TrackingSystem::drawTrackingResult(cv::Mat& _mat_img)
{
	TrackerManager l_manager = this->getTrackerManager();

	// Exception
	if (l_manager.getTrackerVec().size() == 0)
	{
		BOOST_LOG_TRIVIAL(error) << "======================= Error Occured! ======================";
		BOOST_LOG_TRIVIAL(error) << "Function : int TrackingSystem::drawTrackingResult";
		BOOST_LOG_TRIVIAL(error) << "Nothing to draw";
		BOOST_LOG_TRIVIAL(error) << "=============================================================";
		return FAIL;
	}

	std::for_each(l_manager.getTrackerVec().begin(), l_manager.getTrackerVec().end(), [&_mat_img](std::shared_ptr<SingleTracker> ptr) {
		// Draw all rectangles
		cv::rectangle(_mat_img, ptr.get()->getRect(), ptr.get()->getColor(), ptr.get()->getRectWidth());
		if (ptr.get()->getCenters_q().size() == 5) {
			// Draw velocities
			cv::Point2f vel_draw = (ptr.get()->getVel() - ptr.get()->getCenter())*20;
			cv::arrowedLine(_mat_img, ptr.get()->getCenter(), (cv::Point2f)ptr.get()->getCenter()+vel_draw, cv::Scalar(0,0,255), 1);
			if (ptr.get()->getAcc_q().size() > 1) {
				cv::Point2f acc_draw = (ptr.get()->getAcc() - ptr.get()->getCenter())*20;
				cv::arrowedLine(_mat_img, ptr.get()->getCenter(), (cv::Point2f)ptr.get()->getCenter()+acc_draw, cv::Scalar(255,0,0), 1);
			}
			// Draw trajectories
			boost::circular_buffer<cv::Point> positions = ptr.get()->getAvgPos();
			for (int i=1; i < (positions.size()); i++) {
				cv::line(_mat_img, positions.at(i), positions.at(i-1), ptr.get()->getColor(), 1);
			}
		}
		std::string str_label;

		str_label = getLabelStr(ptr.get()->getLabel());
		cv::String text(std::string("ID: ") + std::to_string(ptr.get()->getTargetID()) + " Class: " + str_label);
		cv::Point text_pos = ptr.get()->getRect().tl();
		text_pos.x = text_pos.x + 2;
		text_pos.y = text_pos.y - 5;

		int box_width = ptr.get()->getRect().width;
		cv::rectangle(_mat_img,cv::Point(text_pos.x-3,text_pos.y-15),cv::Point(text_pos.x-2+box_width,text_pos.y+5),ptr.get()->getColor(),cv::FILLED);
		// Put all target ids
		cv::putText(_mat_img,
			text,
			text_pos,
			cv::FONT_HERSHEY_SIMPLEX,
			0.5, //Scale
			/*ptr.get()->getColor()*/cv::Scalar(0,0,0),
			2); //Width

		if (ptr.get()->getCollision() || ptr.get()->getNearMiss()) {
			cv::String text_status;
			text_pos = ptr.get()->getRect().tl();
			text_pos.x = text_pos.x + 2;
			text_pos.y = text_pos.y + ptr.get()->getRect().height + 12;

			cv::rectangle(_mat_img,cv::Point(text_pos.x-3,text_pos.y-12),cv::Point(text_pos.x-2+box_width,text_pos.y+3),ptr.get()->getColor(),cv::FILLED);

			if (ptr.get()->getCollision()) {
				text_status = "Collision";
			} else {
				text_status = "Near Miss";
			}
			cv::putText(_mat_img,text_status,text_pos,cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0),2);
		}
	});

	// Draw dangerous (?) crosswalks
	for (auto && cw: this->d_cws)
		cv::addWeighted(cw, 0.5, _mat_img, 1.0, 0.0, _mat_img);

	this->d_cws.clear();

	return SUCCESS;
}

/* -----------------------------------------------------------------------------------

Function : isValidCollision

Helper function for detectCollisions. A collision is valid if the ratios and threshold
between the different objects is the right one. If not, it just means that they get
occluded but they don't really collide.

----------------------------------------------------------------------------------- */
bool isValidCollision(std::pair<double, int> area1, std::pair<double, int> area2)
{
	const double ratioPtoC = 31;
	const double ratioPtoB = 6.9;
	const double ratioBtoC = 4.5;
	const double threshold = 0.2;

	double a1 = area1.first;
	int label1 = area1.second;
	double a2 = area2.first;
	int label2 = area2.second;

	/*if (label1 == LABEL_UNKNOWN || label2 == LABEL_UNKNOWN) {
		return FALSE;
	}*/

	if (label1 != label2) {
		if (label1 == LABEL_CAR) {
			std::swap(label1, label2);
			std::swap(area1, area2);
		}
		//Extend to further cases if needed
	}

	if (label1 == LABEL_PERSON) {
		if (label2 == LABEL_PERSON) {
			if (a1 > a2*(1-threshold) && a1 < a2*(1+threshold)) {
				return TRUE;
			}
		} else if (label2 == LABEL_CAR) {
			if (a1*ratioPtoC > a2*(1-threshold) && a1*ratioPtoC < a2*(1+threshold)) {
				return TRUE;
			}
		}
	} else if (label1 == LABEL_CAR) {
		if (a1 > a2*(1-threshold) && a1 < a2*(1+threshold)) {
			return TRUE;
		}
	}

	return FALSE;
}

#ifdef ENABLED_DB
void TrackingSystem::dbWrite(mongocxx::v_noabi::collection* col, Pipe* buffer_ptr){
	//std::this_thread::sleep_for(std::chrono::microseconds(10));
	if(buffer_ptr->size() != 0){
		
	std::vector<bsoncxx::document::value> documents;
	while(buffer_ptr->size() != 0){
        //std::cout << "thread" << buffer_ptr->size() << std::endl;
		PipeItem aux = buffer_ptr->back();
		buffer_ptr -> pop_back();
		documents.push_back(
			bsoncxx::builder::stream::document{} 
			<< "frame" << aux.frame
			<< "Id" << aux.Id
			<< "vel_x" << aux.vel_x
			<< "vel_y" << aux.vel_y 
			<< "vel" << aux.vel
			<< "acc_x" << aux.acc_x
			<< "acc_y" << aux.acc_y 
			<< "acc" << aux.acc
			<< "ob1" << aux.ob1
			<< "ob2" << aux.ob2
			<< "event" << aux.event
			<< "nearMiss" << aux.nearMiss
			<< "class" << aux.objectClass
			<< bsoncxx::builder::stream::finalize
		);
		}
		//std::cout << documents.size () << std::endl;
		this -> dbWrite_mutex.lock();
		col -> insert_many(documents);
		this -> dbWrite_mutex.unlock();
	}
}

void TrackingSystem::setUpCollections(){
	this -> conn.start_session();
	this -> dbEnable = true;
	this -> tracker = this -> conn["smart_city_metadata"]["tracker_data"];
	this -> tracker.drop();
	this -> tracker.insert_one({});
	this -> collisions = this -> conn["smart_city_metadata"]["collisions_data"];
	this -> collisions.drop();
	this -> collisions.insert_one({});
	this -> events = this -> conn["smart_city_metadata"]["events"];
	this -> events.drop();
	this -> events.insert_one({});

}
#endif

/* -----------------------------------------------------------------------------------

Function : detectCollisions

Draw red circle when collision is detected and write to log.

----------------------------------------------------------------------------------- */
int TrackingSystem::detectCollisions()
{
	TrackerManager l_manager = this->getTrackerManager();

	// Exception
	if (l_manager.getTrackerVec().size() == 0)
	{
		BOOST_LOG_TRIVIAL(error) << "======================= Error Occured! ======================";
		BOOST_LOG_TRIVIAL(error) << "Function : int TrackingSystem::detectCollisions";
		BOOST_LOG_TRIVIAL(error) << "Nothing to detect";
		BOOST_LOG_TRIVIAL(error) << "=============================================================";
		return FAIL;
	}
	std::vector<std::shared_ptr<SingleTracker>> trackerVec = l_manager.getTrackerVec();
	for (auto i = trackerVec.begin(); i != trackerVec.end(); ++i) {
		SingleTracker& iRef = *(*i);
		if (iRef.getLabel() == LABEL_PERSON) {
			continue;
		}
		boost::circular_buffer<double> vel = iRef.getVel_q();
		boost::circular_buffer<double> vel_x = iRef.getVelX_q();
		boost::circular_buffer<double> vel_y = iRef.getVelY_q();
		boost::circular_buffer<double> acc_x = iRef.getAccX_q();
		boost::circular_buffer<double> acc_y = iRef.getAccY_q();

		double avg_acc_x = 0;
		double avg_acc_y = 0;

		// Keep here for plotting purposes
		// Normalize velocity as in calcAcc
		//int y = iRef.getCenter().y+10;
		//double norm_vel = iRef.getModVel()*1000/y;

		bool inc_speed = (vel[0]-vel[5] > 0 ? true : false);

		bool same_sign_x = ((acc_x[0] > 0) - (acc_x[0] < 0)) == ((vel_x[0] > 0) - (vel_x[0] < 0)); // NOSONAR
		bool same_sign_y = ((acc_y[0] > 0) - (acc_y[0] < 0)) == ((vel_y[0] > 0) - (vel_y[0] < 0)); // NOSONAR
		int sign_x = (same_sign_x ? 1 : -1);
		int sign_y = (same_sign_y ? 1 : -1);

		// Get historic avg_acc
		if (acc_x.size() > 1) {
			int lim = std::min((int)acc_x.size(),3);
			for (int i = 1; i < lim; i++) {
				bool same_sign_x_i = ((acc_x[i] > 0) - (acc_x[i] < 0)) == ((vel_x[i] > 0) - (vel_x[i] < 0)); // NOSONAR
				bool same_sign_y_i = ((acc_y[i] > 0) - (acc_y[i] < 0)) == ((vel_y[i] > 0) - (vel_y[i] < 0)); // NOSONAR
				int sign_x_i = (same_sign_x_i ? 1 : -1);
				int sign_y_i = (same_sign_y_i ? 1 : -1);
				avg_acc_x = avg_acc_x + sign_x_i*std::abs(acc_x.at(i));
				avg_acc_y = avg_acc_y + sign_y_i*std::abs(acc_y.at(i));
			}
		avg_acc_x = avg_acc_x / lim;
		avg_acc_y = avg_acc_y / lim;
		}

		double threshold_x = std::abs(sign_x*std::abs(iRef.getAcc_X()) - (avg_acc_x));
		double threshold_y = std::abs(sign_y*std::abs(iRef.getAcc_Y()) - (avg_acc_y));												
		BOOST_LOG_TRIVIAL(info) << "#" << this -> totalFrames << ',' 
								<< iRef.getTargetID() << ',' 
								<< vel_x[0] << ',' 
								<< vel_y[0] << ',' 
								<< vel[0] << ',' 
								<< acc_x[0] << ','
								<< acc_y[0] << ',' 
								<< iRef.getAcc_q()[0] << ',' 
								<< threshold_x << ',' 
								<< threshold_y << ','
								<< sqrt(threshold_x*threshold_x+threshold_y*threshold_y);
		
		if (threshold_x > 4 || threshold_y >= 3 /*&& !same_sign && !inc_speed*/) {
#ifdef ENABLED_DB
			if(this -> dbEnable && !iRef.getNearMiss()){
				PipeItem document;
				document.frame = totalFrames;
				document.Id = iRef.getTargetID();
				document.event = "Strong speed change";
				document.nearMiss = true;
				document.objectClass = getLabelStr(iRef.getLabel());
				this->buffer_events.push_back(document); 
			}
			std::thread t3(&TrackingSystem::dbWrite, this, &this->events, &this->buffer_events);
#endif
			iRef.setNearMiss(true);
			for (auto j = trackerVec.begin(); j != trackerVec.end(); ++j) {
				SingleTracker& jRef = *(*j);
				if (iRef.getTargetID() == jRef.getTargetID())
					continue;
				cv::Rect recti = iRef.getRect();
				cv::Rect rectj = jRef.getRect();
				bool intersects = ((recti & rectj).area() > 0);
				if (intersects) {
					iRef.setRectWidth(2);
					jRef.setRectWidth(2);
					int ob1 = 0;
					int ob2 = 0;
					if(iRef.getTargetID() < jRef.getTargetID()){
						ob1 = iRef.getTargetID();
						ob2 = jRef.getTargetID();
					}else{
						ob2 = iRef.getTargetID();
						ob1 = jRef.getTargetID();
					}
					if(this -> dbEnable){
						PipeItem document;
						document.frame = totalFrames;
						document.ob1 = ob1;
						document.ob2 = ob2;
						this->buffer_collisions.push_back(document); 
					}
#ifdef ENABLED_DB
					std::thread t2(&TrackingSystem::dbWrite, this, &this->collisions, &this->buffer_collisions);
#endif
					BOOST_LOG_TRIVIAL(error)<< "$" << totalFrames << "$Collision between object $"<<iRef.getTargetID()<<"$ and $"<< jRef.getTargetID() << "$";
					
					
					if (jRef.getNearMiss()) {
						iRef.setCollision(true);
						jRef.setCollision(true);
						iRef.setColor(cv::Scalar(0,0,255)); // Red
						jRef.setColor(cv::Scalar(0,0,255));
					} else {
						iRef.setColor(cv::Scalar(0,165,255)); // Orange
						jRef.setColor(cv::Scalar(0,165,255));
					}
#ifdef ENABLED_DB
					t2.join();
#endif
				}
			}
#ifdef ENABLED_DB
			t3.join();
#endif
		}
	}
	
	this -> totalFrames++;

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

	BOOST_LOG_TRIVIAL(error) << "Close Tracking System...";
}

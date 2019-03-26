#pragma once

#include <thread>
#include <string>
#include <chrono>
#include <mutex>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <boost/circular_buffer.hpp>
#include "yolo_labels.hpp"

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <algorithm>

//MongoDB
#ifdef ENABLED_DB
#include <iostream>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#endif

constexpr int FAIL = -1;
constexpr int SUCCESS = 1;
constexpr int FALSE = 0;
constexpr int TRUE =1;

constexpr int OUT_OF_FRAME = 2;

constexpr int ENTER = 13;
constexpr int ESC = 27;

constexpr int FLAG_SW = 1;
constexpr int FLAG_CW = 1<<1;
constexpr int FLAG_STR = 1<<2;

const int n_frames = 5; // Number of positions to save in the circular buffer
const int n_frames_pos = 50; // Number of positions to save in the circular buffer
const int n_frames_vel = 50; // Number of positions to save in the circular buffer

typedef struct
		{
		 	int frame = 0; 
		 	int Id = 0;
		 	float vel_x = 0;
		 	float vel_y = 0; 
		 	float vel = 0; 
		 	float acc_x = 0;
		 	float acc_y = 0;
		 	float acc = 0;
			int ob1 = 0;
			int ob2 = 0;
			std::string event = "";
			bool nearMiss = false;
			std::string objectClass = "";
		}PipeItem;
typedef std::vector<PipeItem> Pipe;
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
	cv::Point	bottom;				// Base center point of target
	bool		is_tracking_started;		// Is tracking started or not? (Is initializing done or not?)
	cv::Scalar	color;				// Box color
	int		rect_width;			// Box width
	int		label;				// Label (LABEL_CAR, LABEL_PERSON)
	boost::circular_buffer<cv::Point> 	c_q;	// Queue with last n_frames centers
	boost::circular_buffer<cv::Point> 	avg_pos;// Queue with last n_frames centers
	cv::Point2f 	vel;				// Final point of Velocity vector (from center)
	double		modvel;				// Velocity's modulus
	double		vel_x;
	double		vel_y;
	cv::Point2f 	acc;				// Final point of Acceleration vector (from center)
	double		modacc;				// Velocity's modulus
	double		acc_x;
	double		acc_y;
	boost::circular_buffer<double> 	v_q;		// Queue with last n_frames velocities' modulus
	boost::circular_buffer<double> 	v_x_q;		// Queue with last n_frames velocities' modulus
	boost::circular_buffer<double> 	v_y_q;		// Queue with last n_frames velocities' modulus
	boost::circular_buffer<double> 	a_q;		// Queue with last n_frames velocities' modulus
	boost::circular_buffer<double> 	a_x_q;		// Queue with last n_frames velocities' modulus
	boost::circular_buffer<double> 	a_y_q;		// Queue with last n_frames velocities' modulus
	bool		update;				// Update from Detection (new rois)
	bool		to_delete;			// Mark for deletion
	int		no_update_counter;		// Counter if object doesn't get updated
	bool		near_miss;			// If in near miss situation
	bool		collision;			// If in collision situation
	std::pair<char, cv::Mat*>	b_areas;	// Areas where the tracker belongs

public:
	/* Member Initializer & Constructor*/
	SingleTracker(int _target_id, cv::Rect _init_rect, cv::Scalar _color, int _label)
		: target_id(_target_id), confidence(0), is_tracking_started(false), c_q(boost::circular_buffer<cv::Point>(n_frames)), modvel(0), vel_x(0), vel_y(0), to_delete(false), no_update_counter(0), v_x_q(boost::circular_buffer<double>(n_frames_vel)), v_y_q(boost::circular_buffer<double>(n_frames_vel)), v_q(boost::circular_buffer<double>(n_frames_vel)), avg_pos(boost::circular_buffer<cv::Point>(n_frames_pos)), a_q(boost::circular_buffer<double>(n_frames_vel)), a_x_q(boost::circular_buffer<double>(n_frames_vel)), a_y_q(boost::circular_buffer<double>(n_frames_vel)), near_miss(false), collision(false), rect_width(1), b_areas(std::make_pair(0, nullptr))
	{
		// Exception
		if (_init_rect.area() == 0)
		{
			std::cout << "======================= Error Occured! ======================" << std::endl
				  << "Function : Constructor of SingleTracker" << std::endl
				  << "Parameter cv::Rect _init_rect's area is 0" << std::endl
				  << "=============================================================" << std::endl;
		}
		else
		{
			// Initialize rect and center using _init_rect
			this->setRect(_init_rect);
			this->setCenter(_init_rect);
			this->setVel(this->getCenter());
			this->setColor(_color);
			this->setLabel(_label);
		}
	}

	/* Get Function */
	int		getTargetID() { return this->target_id; }
	cv::Rect	getRect() { return this->rect; }
	cv::Point	getCenter() { return this->center; }
	cv::Point	getBottom() { return this->bottom; }
	cv::Point	getVel() { return this->vel; }
	double		getVel_X() { return this->vel_x; }
	double		getVel_Y() { return this->vel_y; }
	cv::Point	getAcc() { return this->acc; }
	double		getAcc_X() { return this->acc_x; }
	double		getAcc_Y() { return this->acc_y; }
	double		getConfidence() { return this->confidence; }
	double		getModVel() { return this->modvel; }
	double		getModAcc() { return this->modacc; }
	bool		getIsTrackingStarted() { return this->is_tracking_started; }
	cv::Scalar	getColor() { return this->color; }
	int		getLabel() { return this->label; }
	boost::circular_buffer<cv::Point> getCenters_q() { return this->c_q; }
	boost::circular_buffer<cv::Point> getAvgPos() { return this->avg_pos; }
	boost::circular_buffer<double> getVel_q() { return this->v_q; }
	boost::circular_buffer<double> getVelX_q() { return this->v_x_q; }
	boost::circular_buffer<double> getVelY_q() { return this->v_y_q; }
	boost::circular_buffer<double> getAcc_q() { return this->a_q; }
	boost::circular_buffer<double> getAccX_q() { return this->a_x_q; }
	boost::circular_buffer<double> getAccY_q() { return this->a_y_q; }
	bool		getUpdateFromDetection() { return this->update; }
	bool		getDelete() { return this->to_delete; }
	int		getNoUpdateCounter() { return this->no_update_counter; }
	bool		getNearMiss() { return this->near_miss; }
	bool		getCollision() { return this->collision; }
	int		getRectWidth() { return this->rect_width; }
	std::pair<char, cv::Mat*> getAreas() {return this->b_areas; }

	/* Set Function */
	void setTargetId(int _target_id) { this->target_id = _target_id; }
	void setRect(cv::Rect _rect) { this->rect = _rect; }
	void setCenter(cv::Point _center) { this->center = _center; this->bottom = cv::Point(this->center.x,this->center.y + (this->rect.height / 2)); }
	void setCenter(cv::Rect _rect) { this->center = cv::Point(_rect.x + (_rect.width) / 2, _rect.y + (_rect.height) / 2); this->bottom = cv::Point(this->center.x,this->center.y + (this->rect.height / 2)); }
	void setVel(cv::Point _vel) { this->vel = _vel; updateVel_X(); updateVel_Y(); updateModVel(); }
	void setAcc(cv::Point _acc) { this->acc = _acc; updateAcc_X(); updateAcc_Y(); updateModAcc(); }
	void setConfidence(double _confidence) { this->confidence = _confidence; }
	void setIsTrackingStarted(bool _b) { this->is_tracking_started = _b; }
	void setColor(cv::Scalar _color) { this->color = _color; }
	void setLabel(int _label) { this->label = _label; }
	void setUpdateFromDetection(bool _update) { this->update = _update; }
	void setNoUpdateCounter(int _counter) { this->no_update_counter = _counter; }
	void setNearMiss(bool _near_miss) { this->near_miss = _near_miss; }
	void setCollision(bool _collision) { this->collision = _collision; }
	void setRectWidth(int _rect_width) { this->rect_width = _rect_width; }
	void setArea(char _areas, cv::Mat* _mask) {this->b_areas = std::make_pair(_areas,_mask); } 

	/* Velocity Related */
	void saveLastCenter(cv::Point _center) { this->c_q.push_front(_center); }
	void saveAvgPos(cv::Point _avg) { this->avg_pos.push_front(_avg); }
	void updateVel_X() { this->vel_x = this->getVel().x - this->getCenter().x; }
	void updateVel_Y() { this->vel_y = this->getVel().y - this->getCenter().y; }
	void updateAcc_X() { this->acc_x = this->getAcc().x - this->getCenter().x; }
	void updateAcc_Y() { this->acc_y = this->getAcc().y - this->getCenter().y; }
	void saveLastVel(double _vel_x, double _vel_y, double _vel) { this->v_x_q.push_front(_vel_x); this->v_y_q.push_front(_vel_y); this->v_q.push_front(_vel); }
	void saveLastAcc(double _acc_x, double _acc_y, double _acc) { this->a_x_q.push_front(_acc_x); this->a_y_q.push_front(_acc_y); this->a_q.push_front(_acc); }
	void updateModVel() { this->modvel = sqrt(this->vel_x*this->vel_x + this->vel_y*this->vel_y); }
	void updateModAcc() { this->modacc = sqrt(this->acc_x*this->acc_x + this->acc_y*this->acc_y); }
	void calcVel();
	void calcAvgPos();
	void calcAcc();

	void assignArea(std::vector<cv::Mat>* mask_sw, std::vector<cv::Mat>* mask_cw, std::vector<std::pair<cv::Mat, int>>* mask_str );

	/* Core Function */
	// Initialize
	int startSingleTracking(cv::Mat _mat_img);

	// Do tracking
	int doSingleTracking(cv::Mat* _mat_img, std::vector<cv::Mat>* mask_sw, std::vector<cv::Mat>* mask_cw, std::vector<std::pair<cv::Mat, int>>* mask_str, Pipe* buffer, int* totalFrames, bool dbEnable);

	// Check the target is inside of the frame
	int isTargetInsideFrame(int _frame_width, int _frame_height, cv::Mat *mask);

	// Check if tracker needs to be deleted
	int markForDeletion();
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
	int id_list = 0; // We keep this to be able to apply new ID to new objects in a simple way.

public:
	/* Get Function */
	std::vector<std::shared_ptr<SingleTracker>>& getTrackerVec() { return this->tracker_vec; } // Return reference! not value!
	int getNextID() { return this->id_list; }

	/* Core Function */
	// Insert new SingleTracker shared pointer into the TrackerManager::tracker_vec
	int insertTracker(cv::Rect* _init_rect, cv::Scalar* _color, int _target_id, int _label, bool update, std::string *last_event, bool* dbEnable, int* totalFrames, Pipe* buffer);
	int insertTracker(std::shared_ptr<SingleTracker> new_single_tracker, bool _update);

	// Find SingleTracker by similarity and return id, return new id if no coincidence
	int findTracker(cv::Rect rect, int label);
	// Find SingleTracker in the TrackerManager::tracker_vec using SingleTracker::target_id
	int findTrackerByID(int _target_id);

	// Deleter SingleTracker which has ID : _target_id from TrackerManager::tracker_vec
	int deleteTracker(int _target_id, std::string *last_event, bool* dbEnable, int* totalFrames, Pipe* buffer);
	int getTrackerLabel(int _target_id);

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
		std::vector<std::pair<cv::Rect, int>> init_target;
		std::vector<std::pair<cv::Rect, int>> updated_target;
		std::string 	*last_event;
		TrackerManager		manager;	// TrackerManager
		cv::Mat*		mask;
		std::vector<cv::Mat>*		mask_sidewalks;
		std::vector<std::pair<cv::Mat, int>>*		mask_streets;
		std::vector<cv::Mat>*		mask_crosswalks;
		std::vector<cv::Mat>		d_cws;
		int 			totalFrames;
		bool dbEnable;
#ifdef ENABLED_DB
		mongocxx::instance inst{};
		mongocxx::client conn{mongocxx::uri{}};
		mongocxx::v_noabi::collection  	tracker;
		mongocxx::v_noabi::collection  	collisions;
		mongocxx::v_noabi::collection	events; 	
		std::mutex dbWrite_mutex;
#endif
		Pipe buffer_tracker;
		Pipe buffer_collisions;
		Pipe buffer_events;
	public:
		/* Constructor */
		explicit TrackingSystem(std::string *last_event):last_event(last_event),mask(nullptr),
					mask_sidewalks(nullptr),mask_streets(nullptr),mask_crosswalks(nullptr), totalFrames(0),dbEnable(false){
					};

	/* Get Function */
	int    getFrameWidth() { return this->frame_width; }
	int    getFrameHeight() { return this->frame_height; }
	cv::Mat   getCurrentFrame() { return this->current_frame; }
	TrackerManager getTrackerManager() { return this->manager; }
	std::vector<cv::Mat>* getMask_sw() { return this->mask_sidewalks; }
	std::vector<cv::Mat>* getMask_cw() { return this->mask_crosswalks; }
	std::vector<std::pair<cv::Mat, int>>* getMask_str() { return this->mask_streets; }


	/* Set Function */
	//void   setFramePath(std::string _frame_path) { this->frame_path.assign(_frame_path); }
	void   	setFrameWidth(int _frame_width) { this->frame_width = _frame_width; }
	void   	setFrameHeight(int _frame_height) { this->frame_height = _frame_height; }
	void   	setCurrentFrame(cv::Mat _current_frame) { this->current_frame = _current_frame; }
	void   	setInitTarget(std::vector<std::pair<cv::Rect, int>> _init_target) { this->init_target = _init_target; }
	void   	setMask(cv::Mat* _mask, std::vector<cv::Mat>* _mask_crosswalks, std::vector<cv::Mat>* _mask_sidewalks, std::vector<std::pair<cv::Mat, int>>* _mask_streets){ 
		this ->	mask = _mask;
		this -> mask_sidewalks = _mask_sidewalks;
		this -> mask_streets = _mask_streets;
		this -> mask_crosswalks = _mask_crosswalks;
	}
	void saveCrosswalk(cv::Mat _roi) { this->d_cws.push_back(_roi); }

	/* Core Function */
	// Initialize TrackingSystem
	int initTrackingSystem();

	// Update TrackingSystem
	int updateTrackingSystem(std::vector<std::pair<cv::Rect, int>> new_target);

	// Start tracking
	int startTracking(cv::Mat& _mat_img);

	// Draw tracking result
	int drawTrackingResult(cv::Mat& _mat_img);

	// Detect collisions
	int detectCollisions();

	// Terminate program
	void terminateSystem();

#ifdef ENABLED_DB
	//clear Mongo collections
	void setUpCollections();

	void dbWrite(mongocxx::v_noabi::collection* col, Pipe* buffer_ptr);
#endif
};

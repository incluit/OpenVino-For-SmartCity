#include <opencv2/opencv.hpp>
#include <iostream>
#include "drawer.hpp"

enum {
	STREETS = 0,
	SIDEWALKS,
	CROSSWALKS,
};

const cv::Scalar STREETS_COLOR = cv::Scalar(0, 0, 255);
const cv::Scalar SIDEWALKS_COLOR = cv::Scalar(0, 255, 0);
const cv::Scalar CROSSWALKS_COLOR = cv::Scalar(255, 255, 0);

cv::Scalar getScnColor(RegionsOfInterest scn)
{
	cv::Scalar color = cv::Scalar(0,0,0);
	switch(scn.state) {
		case STREETS:
			color = STREETS_COLOR;
			break;
		case SIDEWALKS:
			color = SIDEWALKS_COLOR;
			break;
		case CROSSWALKS:
			color = CROSSWALKS_COLOR;
			break;
		default:
			std::cout<<"Something is broken"<<std::endl;
			break;
	}
	return color;
}

void CallBCrop(int event, int x, int y, int flags, void *scn)
{
	RegionsOfInterest* scene = (RegionsOfInterest*) scn;
	RegionsOfInterest& sceneRef = *scene;
	static int x1=0, x2=0, y1=0, y2=0;
	static bool release = false;

	if(event==cv::EVENT_LBUTTONDOWN && !release){
		std::cout << "Left mouse button clicked at (" << x << ", " << y << ")" << std::endl;
		x1 = x;
		y1 = y;
		release = true;
	}

	if(event==cv::EVENT_LBUTTONUP && release) {
		std::cout << "Left mouse button release at (" << x << ", " << y << ")" << std::endl;
		x2 = ( x < sceneRef.orig.cols ? x : sceneRef.orig.cols-2);
		x2 = ( x2 > 0 ? x2 : 1);
		y2 = ( y < sceneRef.orig.rows ? y : sceneRef.orig.rows-2);
		y2 = ( y2 > 0 ? y2 : 1);
		release = false;

		std::cout<<x1<<","<<y1<<","<<x2<<","<<y2<<std::endl;
		cv::Mat roi(cv::Size(sceneRef.orig.cols, sceneRef.orig.rows), sceneRef.orig.type(), cv::Scalar(0));
		cv::rectangle(roi,cv::Point(x2,y2),cv::Point(x1,y1),cv::Scalar(255,255,255),cv::FILLED);
		sceneRef.mask = roi;
		cv::bitwise_and(sceneRef.orig,roi,sceneRef.aux);
	}
}

void drawVertices(RegionsOfInterest *scn)
{
	cv::Scalar color = getScnColor(*scn);
	scn->aux = scn->orig.clone();
	if (scn->vertices.size()>0) {
		// Second, or later click, draw all lines to previous vertex
		for (int i = scn->vertices.size()-1; i > 0; i--){
			cv::line(scn->aux,scn->vertices[i-1],scn->vertices[i],color,2);
		}
	}
}

void CallBDraw(int event, int x, int y, int flags, void *scn)
{
	RegionsOfInterest* scene = (RegionsOfInterest*) scn;
	RegionsOfInterest& sceneRef = *scene;

	if(event==cv::EVENT_LBUTTONDOWN){
		std::cout << "Left mouse button clicked at (" << x << ", " << y << ")" << std::endl;
		sceneRef.vertices.push_back(cv::Point(x,y));
		drawVertices(scene);
	}
}

bool closePolygon(RegionsOfInterest *scn)
{
	RegionsOfInterest* scene = (RegionsOfInterest*) scn;
	RegionsOfInterest& sceneRef = *scene;
	cv::Scalar color;
	double alpha = 0.3;

	if(sceneRef.vertices.size()<2){
		std::cout << "You need a minimum of three points!" << std::endl;
		return false;
	}

	color = getScnColor(sceneRef);
	// Close polygon
	cv::line(sceneRef.aux,sceneRef.vertices[sceneRef.vertices.size()-1],sceneRef.vertices[0],color,2);

	// Mask is black with white where our ROI is
	cv::Mat roi(cv::Size(sceneRef.orig.cols, sceneRef.orig.rows), sceneRef.orig.type(), cv::Scalar(0));
	std::vector< std::vector< cv::Point > > pts{sceneRef.vertices};
	cv::Mat roi2 = roi.clone();
	cv::fillPoly(roi2,pts,cv::Scalar(255, 255, 255));
	cv::fillPoly(roi,pts,color);
	int key = 'x';
	switch(sceneRef.state) {
		case STREETS:
			std::cout<<"Define orientation, (n, s, e, w)" << std::endl;
			while (key != 'n' && key != 's' && key != 'e' && key !='w') {
				key = cv::waitKey();
			}
			sceneRef.streets.push_back(std::make_pair(roi, key));
			sceneRef.mask_streets.push_back(std::make_pair(roi2, key));
			break;
		case SIDEWALKS:
			sceneRef.sidewalks.push_back(roi);
			sceneRef.mask_sidewalks.push_back(roi2); 
			break;
		case CROSSWALKS:
			sceneRef.crosswalks.push_back(roi);
			sceneRef.mask_crosswalks.push_back(roi2); 
			break;
		default:
			std::cout<<"Something is broken"<<std::endl;
			break;
	}
	cv::addWeighted(roi, alpha, sceneRef.out, 1.0, 0.0, sceneRef.out);
	sceneRef.vertices.clear();
	return true;
}

int CropFrame(const cv::String & winname, RegionsOfInterest *scn) {
	bool finished = false;
	RegionsOfInterest* scene = (RegionsOfInterest*) scn;
	RegionsOfInterest& sceneRef = *scene;
	cv::Mat roi(cv::Size(sceneRef.orig.cols, sceneRef.orig.rows), sceneRef.orig.type(), cv::Scalar(0));
	cv::rectangle(roi,cv::Point(1,1),cv::Point(sceneRef.orig.cols-2,sceneRef.orig.rows-2),cv::Scalar(255,255,255),cv::FILLED);
	sceneRef.mask = roi;

	std::cout<<"Select rectangle to crop image. Click, drag and drop. Press 'F' to continue." << std::endl;
	while(!finished){
		cv::imshow(winname, sceneRef.aux);
		switch (cv::waitKey(1)) {
			case 'F':
				finished = true;
				break;
			case 8: // Del
				sceneRef.aux = sceneRef.orig.clone();
				sceneRef.mask = roi;
				break;
			case 27: // Esc
				return -1;
			default:
				break;
		}
	}

	return 0;
}

int DrawAreasOfInterest(const cv::String & winname, RegionsOfInterest *scn)
{
	bool finished = false;
	bool can_finish = true;
	RegionsOfInterest* scene = (RegionsOfInterest*) scn;
	RegionsOfInterest& sceneRef = *scene;

	sceneRef.aux = sceneRef.orig;
	std::cout<<"Draw streets (S), sidewalks(W), crosswalks (Z). To draw next area, press (N) or to finish drawing, press (F)." << std::endl;
	while(!finished){
		cv::imshow(winname, sceneRef.aux);
		switch (cv::waitKey(1)) {
			case 'S':
				if(can_finish) {
					sceneRef.state = STREETS;
					can_finish = false;
				}
				break;
			case 'W':
				if(can_finish) {
					sceneRef.state = SIDEWALKS;
					can_finish = false;
				}
				break;
			case 'Z':
				if(can_finish) {
					sceneRef.state = CROSSWALKS;
					can_finish = false;
				}
				break;
			case 'N':
				std::cout<<"Draw streets (S), sidewalks(W), crosswalks (Z). To draw next area, press (N) or to finish drawing, press (F)." << std::endl;
				can_finish = closePolygon(scn);
				break;
			case 'F':
				if (can_finish) {
					finished=true;
				}
				break;
			case 8: // Del
				if (sceneRef.vertices.size() > 0) {
					sceneRef.vertices.pop_back();
				}
				drawVertices(scene);
				break;
			case 27: // Esc
				return -1;
			default:
				break;
		}
	}
	return 0;
}

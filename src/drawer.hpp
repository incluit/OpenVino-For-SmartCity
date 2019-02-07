#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

struct RegionsOfInterest {
	cv::Mat orig;
	cv::Mat aux;
	cv::Mat out;
	int state = 0;
	cv::Mat mask;
	std::vector<std::pair<cv::Mat, int>> mask_streets;
	std::vector<cv::Mat> mask_sidewalks;
	std::vector<cv::Mat> mask_crosswalks;
	std::vector<cv::Point> mask_vertices;
	std::vector<cv::Mat> street_borders;
	std::vector<cv::Point> vertices;
	std::vector<cv::Mat> sidewalks;
	std::vector<cv::Mat> crosswalks;
	std::vector<std::pair<cv::Mat,int>> streets;
	bool drawing_sidewalks = true;
};

void CallBCrop(int event, int x, int y, int flags, void *scn);
void CallBDraw(int event, int x, int y, int flags, void *scn);

int CropFrame(const cv::String & winname, RegionsOfInterest *scn);
int DrawAreasOfInterest(const cv::String & winname, RegionsOfInterest *scn);


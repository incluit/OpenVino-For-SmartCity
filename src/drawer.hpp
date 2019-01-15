#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

struct RegionsOfInterest {
	cv::Mat orig;
	cv::Mat out;
	std::vector<cv::Mat> street_borders;
	std::vector<cv::Point> vertices;
	std::vector<cv::Point> street_vertices;
	std::vector<cv::Mat> sidewalks;
	std::vector<std::pair<cv::Mat,int>> streets;
	bool drawing_sidewalks = true;
};

void CallBackFunc(int event, int x, int y, int flags, void *scn);

void DrawAreasOfInterest(RegionsOfInterest *scn);


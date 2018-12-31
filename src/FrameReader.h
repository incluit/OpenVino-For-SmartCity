#pragma once
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <io.h>

#define FAIL  -1
#define SUCCESS     1
#define FALSE  0
#define TRUE  1

#define FRAME_READER_IS_MADE  3
#define CANNOT_OPEN_DIR    4
#define WAITING_FOR_READING_FRAME 5
#define READING_FRAME_STARTED  6
#define READING_FRAME_DONE   7

using namespace std;

/* ==========================================================================

Class : FrameReader

FrameReader is responsible for reading all images from designated directory

========================================================================== */
class FrameReader
{
private:
	std::string		path;			// Frame path
	std::string		image_type;		// Image type (For now, just "jpg" is available)
	_finddata_t		fd;
	intptr_t		handle;
	int				status;			// Reading status (CANNOT_OPEN_DIR = 3 / WAITING_FOR_READING_FRAME = 4 / READING_FRAME_STARTED = 5 / READING_FRAME_DONE = 6)

public:

	FrameReader() : status(FRAME_READER_IS_MADE) {};

	/* Set Function */
	void setPath(string _path) { this->path.assign(_path); }
	void setImageType(string _image_type) { this->image_type.assign(_image_type); }
	void setStatus(int status) { this->status = status; }


	/* Get Function*/
	std::string getPath() { return this->path; }
	std::string getImageType() { return this->image_type; }
	_finddata_t getFileFinder() { return this->fd; }
	intptr_t	getHandler() { return this->handle; }
	int			getStatus() { return this->status; }

	/* Core Function */
	// Intialize FrameReader object
	int initFrameReader(std::string _path, std::string _image_type);

	// Get next frame image
	int getNextFrame(cv::Mat& mat_img);
};

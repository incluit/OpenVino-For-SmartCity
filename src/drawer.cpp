#include <opencv2/opencv.hpp>
#include <iostream>
#include "drawer.hpp"

void CallBackFunc(int event, int x, int y, int flags, void *scn)
{
   cv::Scalar color;
   RegionsOfInterest* scene = (RegionsOfInterest*) scn;
   RegionsOfInterest& sceneRef = *scene;

   if(sceneRef.drawing_sidewalks)
	   color = cv::Scalar(0,255,0);
   else
	   color = cv::Scalar(0,0,255);

   if(event==cv::EVENT_LBUTTONDOWN){
	  std::cout << "Left mouse button clicked at (" << x << ", " << y << ")" << std::endl;
      if(sceneRef.vertices.size()==0){
         // First click - just draw point
         sceneRef.out.at<cv::Vec3b>(cv::Point(x,y))=cv::Vec3b(255,0,0);
      } else {
         // Second, or later click, draw line to previous vertex
		 cv::line(sceneRef.out,cv::Point(x,y),sceneRef.vertices[sceneRef.vertices.size()-1],color,2);
      }
      sceneRef.vertices.push_back(cv::Point(x,y));
      return;
   }
}

void DrawAreasOfInterest(RegionsOfInterest *scn)
{
   bool finished = false;
   cv::Mat roi;
   RegionsOfInterest* scene = (RegionsOfInterest*) scn;
   RegionsOfInterest& sceneRef = *scene;

   std::cout<<"Select sidewalks, press c to continue, press f to finish." << std::endl;
   while(!finished){
	  cv::imshow("ImageDisplay", sceneRef.out);
      switch (cv::waitKey(1)) {
	  case 'c':
         if(sceneRef.vertices.size()<2){
			 std::cout << "You need a minimum of three points!" << std::endl;
         }
		 else {
         // Close polygon
		 cv::line(sceneRef.out,sceneRef.vertices[sceneRef.vertices.size()-1],sceneRef.vertices[0],cv::Scalar(0,255,0),2);

         // Mask is black with white where our ROI is
		 cv::Mat roi(cv::Size(sceneRef.orig.cols, sceneRef.orig.rows), sceneRef.orig.type(), cv::Scalar(0));
	     std::vector< std::vector< cv::Point > > pts{sceneRef.vertices};
         fillPoly(roi,pts,cv::Scalar(0,125,0));
         sceneRef.sidewalks.push_back(roi);
		 sceneRef.vertices.clear();
         }
		 break;
	  case 'f':
         finished=true;
		 break;
	  }
   }
   finished = false;
   sceneRef.drawing_sidewalks = false;

   std::cout<<"Select streets, press c to continue, press f to finish." << std::endl;
   while(!finished){
	  cv::imshow("ImageDisplay", sceneRef.out);
      switch (cv::waitKey(1)) {
	  case 'c':
         if(sceneRef.vertices.size()<2){
		    std::cout << "You need a minimum of three points!" << std::endl;
         }
		 else {
	        std::cout<<"Define orientation (n, s, e, w)" << std::endl;
            int key = 'x';
		    while (key != 'n' && key != 's' && key != 'e' && key !='w')
			   key = cv::waitKey();
		    // Close polygon
            line(sceneRef.out,sceneRef.vertices[sceneRef.vertices.size()-1],sceneRef.vertices[0],cv::Scalar(0,0,255),2);

            // Mask is black with white where our ROI is
		    cv::Mat roi(cv::Size(sceneRef.orig.cols, sceneRef.orig.rows), sceneRef.orig.type(), cv::Scalar(0));
	        std::vector< std::vector< cv::Point > > pts{sceneRef.vertices};
            fillPoly(roi, pts, cv::Scalar(0,0,125));
		    sceneRef.vertices.clear();
			sceneRef.streets.push_back(std::make_pair(roi, key));
		 }
		 break;
	  case 'f':
		 finished=true;
		 break;
	  }
   }

   double alpha = 0.3;
   for (int i=0; i<sceneRef.sidewalks.size(); i++)
	   cv::addWeighted(sceneRef.sidewalks[i], alpha, sceneRef.out, 1.0, 0.0, sceneRef.out);
   for (int i=0; i<sceneRef.streets.size(); i++) {
      cv::addWeighted(sceneRef.streets[i].first, alpha, sceneRef.out, 1.0, 0.0, sceneRef.out);
	  std::cout << "Orientations: " << i << "," << (char) sceneRef.streets[i].second << std::endl;
   }
}

/**
* Copyright (c) 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine interactive_Vehicle_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <queue>
#include <utility>
#include <stdlib.h> 

#include <opencv2/opencv.hpp>
#include "customflags.hpp"
#include "drawer.hpp"

#include "Tracker.h"
#include "object_detection.hpp"
#include "yolo_detection.hpp"
#include "yolo_labels.hpp"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

// -------------------------Generic routines for detection networks-------------------------------------------------
bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    BOOST_LOG_TRIVIAL(info) << "Parsing input parameters";
    if (FLAGS_i.empty()) {
        throw std::invalid_argument("Parameter -i is not set");
    }
    if (FLAGS_auto_resize) {
        BOOST_LOG_TRIVIAL(warning) << "auto_resize=1, forcing all batch sizes to 1";
        FLAGS_n = 1;
        FLAGS_n_p = 1;
        FLAGS_n_y = 1;
    }
    if (FLAGS_n_async < 1) {
        throw std::invalid_argument("Parameter -n_async must be >= 1");
    }
    return true;
}
std::string return_current_time_and_date(){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    return ss.str();
}

void init_logging(std::string base)
{
    std::stringstream fileName;
    fileName << "log/" << base << "_" << return_current_time_and_date() << ".log";
    boost::log::register_simple_formatter_factory<boost::log::trivial::severity_level, char>("Severity");
    boost::log::add_file_log(
        boost::log::keywords::file_name = fileName.str(),
        boost::log::keywords::format = "[%TimeStamp%] [%Severity%] [%LineID%] - %Message%"
    );
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );
    boost::log::add_common_attributes();
}
int main(int argc, char *argv[]) {
    try {
        // ---------------------------Init Log-------------------------------
        init_logging("test");
        // ---------------------------Parsing and validation of input args--------------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        std::cout << "Incluit - Openvino-for-Smartcity" << std::endl;
        BOOST_LOG_TRIVIAL(info) << "Incluit - Openvino-for-Smartcity";
#ifdef ENABLED_DB
        if(FLAGS_show_graph){
                if(system("../scripts/startupdb.sh") != 0){
                    BOOST_LOG_TRIVIAL(error) << "MongoDB is not installed on this device";
                    std::cerr << "[ ERROR ] - MongoDB is not installed on this device" << '\n';
                    return 1;
                }   
        }
#endif
        /** This sample covers 2 certain topologies and cannot be generalized **/
        BOOST_LOG_TRIVIAL(info) << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion();

        // -----------------------------Read input -----------------------------------------------------
        BOOST_LOG_TRIVIAL(info) << "Reading input";
        cv::VideoCapture cap;
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::invalid_argument("Cannot open input file or camera: " + FLAGS_i);
        }

        // ---------------------Load plugins for inference engine------------------------------------------------
        std::map<std::string, InferenceEngine::Core> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_p, FLAGS_m_p}, {FLAGS_d_y, FLAGS_m_y}, {FLAGS_d_vp, FLAGS_m_vp}
        };

        const bool runningAsync = (FLAGS_n_async > 1);
        BOOST_LOG_TRIVIAL(info) << "FLAGS_n_async=" << FLAGS_n_async << ", inference pipeline will operate "
                << (runningAsync ? "asynchronously" : "synchronously")
               ;

        FramePipelineFifo pipeS0Fifo;
        FramePipelineFifo pipeS0Fifo2;
        FramePipelineFifo pipeS0toS1Fifo;
        FramePipelineFifo pipeS0toS2Fifo;
        FramePipelineFifo pipeS1toS2Fifo;
        FramePipelineFifo pipeS2toS3Fifo;
        FramePipelineFifo pipeS3toS4Fifo;
        FramePipelineFifo pipeS1toS4Fifo;

        //Yolo lane FIFOs
        FramePipelineFifo pipeS0ytoS1yFifo;
        FramePipelineFifo pipeS1ytoS4Fifo;

        FramePipelineFifo news0tos1;

        ObjectDetection VehicleDetection(FLAGS_m, FLAGS_d, "Vehicle Detection", FLAGS_n, FLAGS_n_async, FLAGS_auto_resize, FLAGS_t);
        ObjectDetection PedestriansDetection(FLAGS_m_p, FLAGS_d_p, "Pedestrians Detection", FLAGS_n_p, FLAGS_n_async, FLAGS_auto_resize, FLAGS_t);
        ObjectDetection VPDetection(FLAGS_m_vp, FLAGS_d_vp, "Pedestrians Detection", FLAGS_n_vp, FLAGS_n_async, FLAGS_auto_resize, FLAGS_t);
        YoloDetection   GeneralDetection(FLAGS_m_y, FLAGS_d_y, "Yolo Detection", FLAGS_n_y, FLAGS_n_async, FLAGS_auto_resize, FLAGS_t, FLAGS_iou_t);    

        const bool yolo_enabled = GeneralDetection.enabled();
        const bool vp_enabled = (VehicleDetection.enabled() && PedestriansDetection.enabled());
        const bool vp2_enabled = VPDetection.enabled();

        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            BOOST_LOG_TRIVIAL(info) << "Loading plugin " << deviceName;
            InferenceEngine::Core core;

            /** Printing plugin version **/
            std::cout << core.GetVersions(deviceName);
            /** Load extensions for the CPU plugin **/
            if (deviceName.find("CPU") != std::string::npos) {
                core.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), deviceName);

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
                    core.AddExtension(extension_ptr, deviceName);
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                core.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, deviceName);
            }

            pluginsForDevices[deviceName] = core;
        }

        // --------------------Load networks (Generated xml/bin files)-------------------------------------------
        Load(VehicleDetection).into(pluginsForDevices[FLAGS_d], FLAGS_d, false);
        Load(PedestriansDetection).into(pluginsForDevices[FLAGS_d_p], FLAGS_d_p, false);
        Load(GeneralDetection).into(pluginsForDevices[FLAGS_d_y], FLAGS_d_y, false);
        Load(VPDetection).into(pluginsForDevices[FLAGS_d_vp], FLAGS_d_vp, false);


        // read input (video) frames, need to keep multiple frames stored
        //  for batching and for when using asynchronous API.
        const int maxNumInputFrames = FLAGS_n_async * VehicleDetection.maxBatch + 1;  // +1 to avoid overwrite
        cv::Mat* inputFrames = new cv::Mat[maxNumInputFrames];
        cv::Mat* inputFrames2 = new cv::Mat[maxNumInputFrames];

        std::queue<cv::Mat*> inputFramePtrs, inputFramePtrs_clean;
        for(int fi = 0; fi < maxNumInputFrames; fi++) {
            inputFramePtrs.push(&inputFrames[fi]);
            inputFramePtrs_clean.push(&inputFrames2[fi]);
        }

	//-----------------------Define regions of interest-----------------------------------------------------
        RegionsOfInterest scene;

	cap.read(scene.orig);
	// Do deep copy to preserve original frame
	scene.aux = scene.orig.clone();
	scene.out = scene.orig.clone();
	cv::Mat aux_mask;
	std::vector<cv::Mat> mask_sidewalk;
	std::vector<cv::Mat> mask_crosswalk;
	std::vector<std::pair<cv::Mat, int>> mask_streets;

        cv::Mat first_frame_masked = scene.orig.clone();

        // Add check
	if (FLAGS_show_selection){
		int ret = 0;
		std::string winname;
		winname = "Crop";
		cv::namedWindow(winname);
		cv::moveWindow(winname, 10, 10);
		cv::setMouseCallback(winname, CallBCrop, &scene);
		ret = CropFrame(winname, &scene);
		if (ret < 0) {
			return FAIL;
		}
		cv::destroyWindow(winname);
		winname = "Draw Areas";
		cv::namedWindow(winname);
		cv::moveWindow(winname, 10, 10);
		cv::setMouseCallback(winname, CallBDraw, &scene);
		ret = DrawAreasOfInterest(winname, &scene);
		if (ret < 0) {
			return FAIL;
		}
		cv::destroyWindow(winname);
		winname = "Result";
		cv::namedWindow(winname);
		cv::moveWindow(winname, 10, 10);
		cv::imshow(winname, scene.out);
		std::cout << "Showing selection result, press any key to continue." << std::endl;

		cv::waitKey();
		aux_mask = scene.mask;
		mask_crosswalk = scene.mask_crosswalks;
		mask_sidewalk = scene.mask_sidewalks;
		mask_streets = scene.mask_streets;

		cv::bitwise_and(scene.orig,aux_mask,first_frame_masked);
		cv::imshow(winname, first_frame_masked);
		cv::waitKey();
		cv::destroyWindow(winname);
	}

        
        // ----------------------------Do inference-------------------------------------------------------------
        BOOST_LOG_TRIVIAL(info) << "Start inference ";
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        std::chrono::high_resolution_clock::time_point wallclockStart;
		std::chrono::high_resolution_clock::time_point wallclockEnd;

        bool firstFrame = true;
        bool firstFrameWithDetections = true;
        bool haveMoreFrames = true;
        bool done = false;
        int numFrames = 0;
        int numSyncFrames = 0;
        int totalFrames = 0;
        double ocv_decode_time_vehicle = 0;
		double ocv_decode_time_pedestrians = 0;
		double ocv_render_time = 0;
        cv::Mat* lastOutputFrame;
        std::vector<std::pair<cv::Rect, int>> firstResults;
        const int update_frame = 0;
        int update_counter = 0;
        std::string last_event;
        TrackingSystem tracking_system(&last_event);
        if(FLAGS_show_selection){
            tracking_system.setMask(&aux_mask, &mask_crosswalk, &mask_sidewalk, &mask_streets);
        }
#ifdef ENABLED_DB
        if(FLAGS_show_graph)
            tracking_system.setUpCollections();
#endif
        // structure to hold frame and associated data which are passed along
        //  from stage to stage for each to do its work
        
        // Queues to pass information across pipeline stages
        wallclockStart = std::chrono::high_resolution_clock::now();
        /** Start inference & calc performance **/
        do {
            std::chrono::high_resolution_clock::time_point a = std::chrono::high_resolution_clock::now();
            std::chrono::high_resolution_clock::time_point b = std::chrono::high_resolution_clock::now();
            ms detection_time;
            detection_time = std::chrono::duration_cast<ms>(b -a);
            std::chrono::high_resolution_clock::time_point t0;
			std::chrono::high_resolution_clock::time_point t1;
            //------------------------------------------------------------------------------------
            //------------------- Frame Read Stage -----------------------------------------------
            //------------------------------------------------------------------------------------
            if (haveMoreFrames && (inputFramePtrs.size() >= VehicleDetection.maxBatch)) {
                FramePipelineFifoItem ps0;
                for(numFrames = 0; numFrames < VehicleDetection.maxBatch; numFrames++) {
                    // read in a frame
		            cv::Mat* curFrame = &scene.orig;
                    cv::Mat* curFrame_clean;

                    if (totalFrames > 0) {
                        curFrame = inputFramePtrs.front();
				        curFrame_clean = inputFramePtrs_clean.front();
				        inputFramePtrs.pop();
                        inputFramePtrs_clean.pop();
                        if(FLAGS_show_selection){
                            haveMoreFrames = cap.read(*curFrame_clean);
                            cv::bitwise_and(*curFrame_clean,aux_mask,*curFrame);
                        }else{
                            haveMoreFrames = cap.read(*curFrame);
                            curFrame_clean = curFrame;
                        }
					}else{
                        curFrame = &first_frame_masked;
                        curFrame_clean = &scene.orig;
                    }
                    if (!haveMoreFrames) {
                        break;
                    }
                    totalFrames++;
                    ps0.batchOfInputFrames.push_back(curFrame);
                    ps0.batchOfInputFrames_clean.push_back(curFrame_clean);
                    if (firstFrame && !FLAGS_no_show) {
                        BOOST_LOG_TRIVIAL(info) << "Press 's' key to save a snapshot, press any other key to stop";
                    }

                    firstFrame = false;
                }
                pipeS0Fifo.push(ps0);
            }

            if(vp_enabled){
                VehicleDetection.run_inferrence(&pipeS0Fifo, &pipeS1toS2Fifo);
                VehicleDetection.wait_results(&pipeS1toS4Fifo);
                PedestriansDetection.run_inferrence(&pipeS1toS2Fifo);
                PedestriansDetection.wait_results(&pipeS3toS4Fifo);
            }

            if(vp2_enabled){
                VPDetection.run_inferrence(&pipeS0Fifo);
                VPDetection.wait_results(&pipeS1ytoS4Fifo);
            }

            if(yolo_enabled){
                GeneralDetection.run_inferrence(&pipeS0Fifo);
                GeneralDetection.wait_results(&pipeS1ytoS4Fifo);
            }

            /* *** Pipeline Stage 4: Render Results *** */
            if (((!pipeS3toS4Fifo.empty() && !pipeS1toS4Fifo.empty()) &&  vp_enabled) 
                    ||  (!pipeS1ytoS4Fifo.empty() && yolo_enabled) 
                    || (!pipeS1ytoS4Fifo.empty() && vp2_enabled)) {

                FramePipelineFifoItem ps3s4i;
                FramePipelineFifoItem ps1s4i;
                FramePipelineFifoItem ps1ys4i;

                cv::Mat outputFrame;
                cv::Mat* outputFrame2;
                cv::Mat outputFrame_clean;
                cv::Mat* outputFrame2_clean;

                if(vp_enabled){
                    ps3s4i = pipeS3toS4Fifo.front();
                    pipeS3toS4Fifo.pop();
                    ps1s4i = pipeS1toS4Fifo.front();
                    pipeS1toS4Fifo.pop();

                    outputFrame = *(ps3s4i.outputFrame);
                    outputFrame2 = ps3s4i.outputFrame;
                    outputFrame_clean = *(ps3s4i.outputFrame_clean);
                    outputFrame2_clean = ps3s4i.outputFrame_clean;
                    
                    
                    // draw box around vehicles
                    for (auto && loc : ps1s4i.resultsLocations) {
                        if(!FLAGS_tracking) {
                            cv::rectangle(outputFrame_clean, loc.first, COLOR_CAR, 1);
                        }
                        if (firstFrameWithDetections || update_counter == update_frame){
                            firstResults.push_back(std::make_pair(loc.first, LABEL_CAR));
                        }
                    }
                    // draw box around pedestrians
                    for (auto && loc : ps3s4i.resultsLocations) {
                        if(!FLAGS_tracking) {
                            cv::rectangle(outputFrame_clean, loc.first, COLOR_PERSON, 1);
                        }
                        if (firstFrameWithDetections || update_counter == update_frame){
                            firstResults.push_back(std::make_pair(loc.first, LABEL_PERSON));
                        }
                    }
                }

                if(yolo_enabled){
                    ps1ys4i = pipeS1ytoS4Fifo.front();
                    pipeS1ytoS4Fifo.pop();

                    outputFrame = *(ps1ys4i.outputFrame);
                    outputFrame2 = ps1ys4i.outputFrame;
                    outputFrame_clean = *(ps1ys4i.outputFrame_clean);
                    outputFrame2_clean = ps1ys4i.outputFrame_clean;

                    for (auto && loc : ps1ys4i.resultsLocations) {
                        if(!FLAGS_tracking) {
                            cv::Scalar color_obj;
                            switch (loc.second) {
                            case LABEL_PERSON:
                                color_obj = COLOR_PERSON;
                                break;
                            case LABEL_CAR:
                                color_obj = COLOR_CAR;
                                break;
                            default:
                                color_obj = COLOR_UNKNOWN;
                                break;
                            }
                            cv::rectangle(outputFrame_clean, loc.first, color_obj, 1);
                        }
                        if (firstFrameWithDetections || update_counter == update_frame){
                            firstResults.push_back(loc);
                        }
                    }
                }

                if(vp2_enabled){
                    ps1ys4i = pipeS1ytoS4Fifo.front();
                    pipeS1ytoS4Fifo.pop();

                    outputFrame = *(ps1ys4i.outputFrame);
                    outputFrame2 = ps1ys4i.outputFrame;
                    outputFrame_clean = *(ps1ys4i.outputFrame_clean);
                    outputFrame2_clean = ps1ys4i.outputFrame_clean;

                    for (auto && loc : ps1ys4i.resultsLocations) {
                        if(loc.second == 1){
                            loc.second = LABEL_PERSON;
                        }else if(loc.second == 0){
                            loc.second = LABEL_BICYCLE;
                        }
                        if(!FLAGS_tracking) {
                            cv::Scalar color_obj;
                                            switch (loc.second) {
                                                case LABEL_PERSON:
                                                    color_obj = COLOR_PERSON;
                                                    break;
                                                case LABEL_BICYCLE:
                                                color_obj = COLOR_PERSON;
                                                break;
                                                case LABEL_CAR:
                                                        color_obj = COLOR_CAR;
                                                        break;
                                                default:
                                                        color_obj = COLOR_UNKNOWN;
                                                        break;
                                                }
                            cv::rectangle(outputFrame_clean, loc.first, color_obj, 1);
                        }
                        if (firstFrameWithDetections || update_counter == update_frame ){
                            firstResults.push_back(loc);
                        }
                    }
                }

                if(FLAGS_tracking) {
                    if(firstFrameWithDetections){
                    tracking_system.setFrameWidth(outputFrame.cols);
                    tracking_system.setFrameHeight(outputFrame.rows);
                    tracking_system.setInitTarget(firstResults);
                    tracking_system.initTrackingSystem();
                    }
                    if( update_counter == update_frame ){
                        tracking_system.updateTrackingSystem(firstResults);
                    }
                    int tracking_success = tracking_system.startTracking(outputFrame);
                    if (tracking_success == FAIL){
                        break;
                    }
                    if (tracking_system.getTrackerManager().getTrackerVec().size() != 0){
			if (FLAGS_collision) {
				tracking_system.detectCollisions();
			}
                        tracking_system.drawTrackingResult(outputFrame_clean);
                    }
                }
                if(update_counter == update_frame){
                    int n_person = 0;
                    int n_car = 0;
                    int n_bus = 0;
                    int n_truck = 0;
                    int n_bike = 0;
                    int n_motorbike = 0;
                    int n_ukn = 0;
                    for(auto && i : firstResults){
                        switch (i.second)
                        {
                            case LABEL_PERSON:
                                n_person++;
                                break;
                            case LABEL_CAR:
                                n_car++;
                                break;
                            case LABEL_BUS:
                                n_bus++;
                                break;
                            case LABEL_TRUCK:
                                n_truck++;
                                break;
                            case LABEL_BICYCLE:
                                n_bike++;
                                break;
                            case LABEL_MOTORBIKE:
                                n_motorbike++;
                                break;
                            default:
                                n_ukn++;
                                break;
                        }
                    }
               }

                firstFrameWithDetections = false;
                firstResults.clear();
                update_counter++;
                if (update_counter > update_frame) {
                    update_counter = 0;
		        }
                if(FLAGS_show_selection){
                    cv::addWeighted(aux_mask, 0.05, outputFrame_clean, 1.0, 0.0, outputFrame_clean);
                    cv::polylines(outputFrame_clean,scene.mask_vertices,true, cv::Scalar(255,0,0),1);
                }
		        // ----------------------------Execution statistics -----------------------------------------------------
                std::ostringstream out;
				std::ostringstream out1;
				std::ostringstream out2;

                ocv_decode_time_pedestrians = 0;
                ocv_decode_time_vehicle = 0;

                cv::putText(outputFrame_clean, out1.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
                cv::putText(outputFrame_clean, out2.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0));

                // When running asynchronously, timing metrics are not accurate so do not display them
                if (!runningAsync) {
                    out.str("");
                    out << "Vehicle detection time ";
                    if (VehicleDetection.maxBatch > 1) {
                        out << "(batch size = " << VehicleDetection.maxBatch << ") ";
                    }
                    out << ": " << std::fixed << std::setprecision(2) << detection_time.count()
                        << " ms ("
                        << 1000.F * numSyncFrames / detection_time.count() << " fps)";
                    cv::putText(outputFrame_clean, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                cv::Scalar(255, 0, 0));

                }

                // -----------------------Display Results ---------------------------------------------
                t0 = std::chrono::high_resolution_clock::now();
                if (!FLAGS_no_show) {
                    std::string winname = "Detection result";
                    cv::namedWindow(winname);
                    cv::moveWindow(winname, 10, 10);
                    cv::imshow(winname, outputFrame_clean);
                    lastOutputFrame = &outputFrame_clean;
                }
                t1 = std::chrono::high_resolution_clock::now();
                ocv_render_time += std::chrono::duration_cast<ms>(t1 - t0).count();

                // watch for keypress to stop or snapshot
                int keyPressed;
                if (-1 != (keyPressed = cv::waitKey(1)))
                {
                    if ('s' == keyPressed) {
                        // save screen to output file
                        BOOST_LOG_TRIVIAL(info) << "Saving snapshot of image";
                        cv::imwrite("snapshot.bmp", outputFrame);
                    } else {
                        haveMoreFrames = false;
                    }
                }

                // done with frame buffer, return to queue
                
                inputFramePtrs.push(outputFrame2);
                inputFramePtrs_clean.push(outputFrame2_clean);

            }

            // wait until break from key press after all pipeline stages have completed
            done = !haveMoreFrames && pipeS0toS1Fifo.empty() && pipeS1toS2Fifo.empty() && pipeS2toS3Fifo.empty()
                        && pipeS3toS4Fifo.empty() && pipeS0toS2Fifo.empty() && pipeS1toS4Fifo.empty() 
                        && pipeS0ytoS1yFifo.empty() && pipeS1ytoS4Fifo.empty();
            // end of file we just keep last image/frame displayed to let user check what was shown
            if (done) {
                // done processing, save time
                wallclockEnd = std::chrono::high_resolution_clock::now();

                if (!FLAGS_no_wait && !FLAGS_no_show) {
                    BOOST_LOG_TRIVIAL(info) << "Press 's' key to save a snapshot, press any other key to exit";
                    while (cv::waitKey(0) == 's') {
                        // save screen to output file
                        BOOST_LOG_TRIVIAL(info) << "Saving snapshot of image";
                        cv::imwrite("snapshot.bmp", *lastOutputFrame);
                    }
                    haveMoreFrames = false;
                    break;
                }
            }
        } while(!done);

        // calculate total run time
        ms total_wallclock_time = std::chrono::duration_cast<ms>(wallclockEnd - wallclockStart);

        // report loop time
        BOOST_LOG_TRIVIAL(info) << "     Total main-loop time:" << std::fixed << std::setprecision(2)
                << total_wallclock_time.count() << " ms ";
        BOOST_LOG_TRIVIAL(info) << "           Total # frames:" << totalFrames;
        float avgTimePerFrameMs = total_wallclock_time.count() / (float)totalFrames;
        BOOST_LOG_TRIVIAL(info) << "   Average time per frame:" << std::fixed << std::setprecision(2)
                    << avgTimePerFrameMs << " ms "
                    << "(" << 1000.0F / avgTimePerFrameMs << " fps)";

        delete [] inputFrames;

    }
    catch (const std::exception& error) {
        BOOST_LOG_TRIVIAL(error) << error.what();
        std::cout << error.what() << std::endl;
        std::cout << "If missing -d_y argument, try running ../scripts/setupenv.sh" << std::endl;
        return 1;
    }
    catch (...) {
        BOOST_LOG_TRIVIAL(error) << "Unknown/internal exception happened.";
        return 1;
    }

    BOOST_LOG_TRIVIAL(info) << "Execution successful";
    return 0;
}

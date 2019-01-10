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

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::invalid_argument("Parameter -i is not set");
    }

    /*if (!(FLAGS_m.empty() && FLAGS_m_p.empty() && !FLAGS_m_y.empty()) && !(!FLAGS_m.empty() && !FLAGS_m_p.empty() && FLAGS_m_y.empty())) {
        throw std::invalid_argument("Check the models combinations.");
    }*/

    if (FLAGS_auto_resize) {
	    slog::warn << "auto_resize=1, forcing all batch sizes to 1" << slog::endl;
	    FLAGS_n = 1;
	    FLAGS_n_p = 1;
	    FLAGS_n_y = 1;
    }

    if (FLAGS_n_async < 1) {
        throw std::invalid_argument("Parameter -n_async must be >= 1");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

int main(int argc, char *argv[]) {
    try {
        /** This sample covers 2 certain topologies and cannot be generalized **/
        std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        // ---------------------------Parsing and validation of input args--------------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // -----------------------------Read input -----------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::invalid_argument("Cannot open input file or camera: " + FLAGS_i);
        }
        //const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        //const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // ---------------------Load plugins for inference engine------------------------------------------------
        std::map<std::string, InferenceEngine::InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_p, FLAGS_m_p}, {FLAGS_d_y, FLAGS_m_y}, {FLAGS_d_vp, FLAGS_m_vp}
        };

        const bool runningAsync = (FLAGS_n_async > 1);
        slog::info << "FLAGS_n_async=" << FLAGS_n_async << ", inference pipeline will operate "
                << (runningAsync ? "asynchronously" : "synchronously")
                << slog::endl;

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
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Load extensions for the CPU plugin **/
            if (deviceName.find("CPU") != std::string::npos) {
                plugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }

            pluginsForDevices[deviceName] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}});
            }
        }

        // --------------------Load networks (Generated xml/bin files)-------------------------------------------
        Load(VehicleDetection).into(pluginsForDevices[FLAGS_d], false);
        Load(PedestriansDetection).into(pluginsForDevices[FLAGS_d_p], false);
        Load(GeneralDetection).into(pluginsForDevices[FLAGS_d_y], false);
        Load(VPDetection).into(pluginsForDevices[FLAGS_d_vp], false);


        // read input (video) frames, need to keep multiple frames stored
        //  for batching and for when using asynchronous API.
        const int maxNumInputFrames = FLAGS_n_async * VehicleDetection.maxBatch + 1;  // +1 to avoid overwrite
        cv::Mat* inputFrames = new cv::Mat[maxNumInputFrames];
        std::queue<cv::Mat*> inputFramePtrs;
        for(int fi = 0; fi < maxNumInputFrames; fi++) {
            inputFramePtrs.push(&inputFrames[fi]);
        }
		//-----------------------Define regions of interest-----------------------------------------------------
            RegionsOfInterest scene;

    		cap.read(scene.orig);
    		// Do deep copy to preserve original frame
    		scene.out = scene.orig.clone();
            // Add check
        if (FLAGS_show_selection){
            cv::namedWindow("ImageDisplay",1);
            cv::setMouseCallback("ImageDisplay", CallBackFunc, &scene);
    		DrawAreasOfInterest(&scene);
    		cv::destroyWindow("ImageDisplay");
    		cv::namedWindow("Result",1);
    		cv::imshow("Result", scene.out);
    		cv::waitKey();
        }
        
        // ----------------------------Do inference-------------------------------------------------------------
        slog::info << "Start inference " << slog::endl;
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
        TrackingSystem tracking_system;
        int detected_objects = 0;

	const int update_frame = 5;
	int update_counter = 0;

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
                    if (totalFrames > 0) {
					   curFrame = inputFramePtrs.front();
					   inputFramePtrs.pop();
                       haveMoreFrames = cap.read(*curFrame);
					}
                    if (!haveMoreFrames) {
                        break;
                    }
                    std::cout << "Frame n°:[" << totalFrames << "]" << std::endl;
                    totalFrames++;
                    ps0.batchOfInputFrames.push_back(curFrame);
                    if (firstFrame && !FLAGS_no_show) {
                        slog::info << "Press 's' key to save a snapshot, press any other key to stop" << slog::endl;
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

                if(vp_enabled){
                    ps3s4i = pipeS3toS4Fifo.front();
                    pipeS3toS4Fifo.pop();
                    ps1s4i = pipeS1toS4Fifo.front();
                    pipeS1toS4Fifo.pop();

                    outputFrame = *(ps3s4i.outputFrame);
                    outputFrame2 = ps3s4i.outputFrame;
                    
                    detected_objects = ps1s4i.resultsLocations.size() + ps3s4i.resultsLocations.size();
                    // draw box around vehicles
                    for (auto && loc : ps1s4i.resultsLocations) {
                        cv::rectangle(outputFrame, loc.first, cv::Scalar(0, 255, 0), 1);
                        if (firstFrameWithDetections || update_counter == update_frame){
                            firstResults.push_back(std::make_pair(loc.first, LABEL_CAR));
                        }
                    }
                    // draw box around pedestrians
                    for (auto && loc : ps3s4i.resultsLocations) {
                        cv::rectangle(outputFrame, loc.first, cv::Scalar(255, 255, 255), 1);
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

                    detected_objects = ps1ys4i.resultsLocations.size();

                    for (auto && loc : ps1ys4i.resultsLocations) {
                        cv::rectangle(outputFrame, loc.first, cv::Scalar(255, 255, 255), 1);
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

                    detected_objects = ps1ys4i.resultsLocations.size();

                    for (auto && loc : ps1ys4i.resultsLocations) {
                        cv::rectangle(outputFrame, loc.first, cv::Scalar(255, 255, 255), 1);
                        if(loc.second == 1){
                            loc.second = LABEL_PERSON;
                        }else if(loc.second == 0){
                            loc.second = LABEL_BICYCLE;
                        }
                         if (firstFrameWithDetections || update_counter == update_frame ){
                            firstResults.push_back(loc);
                        }
                    }
                }

                std::cout << "Amount of infered objects: " << detected_objects << std::endl; 
            
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
                        tracking_system.drawTrackingResult(outputFrame);
                        tracking_system.detectCollisions(outputFrame);
                    }
                }

		firstFrameWithDetections = false;
		firstResults.clear();
		update_counter++;
		if (update_counter > update_frame) {
			update_counter = 0;
		}
		        // ----------------------------Execution statistics -----------------------------------------------------
                std::ostringstream out;
				std::ostringstream out1;
				std::ostringstream out2;

				/*if (VehicleDetection.maxBatch > 1) {
                    out1 << "OpenCV cap/render (avg) Vehicles time: " << std::fixed << std::setprecision(2)
                        << (ocv_decode_time_vehicle / numSyncFrames + ocv_render_time / totalFrames) << " ms";
                } else {
                    out1 << "OpenCV cap/render Vehicles time: " << std::fixed << std::setprecision(2)
                        << (ocv_decode_time_vehicle + ocv_render_time) << " ms";
                    ocv_render_time = 0;
                }
                if (PedestriansDetection.maxBatch > 1) {
                    out2 << "OpenCV cap/render (avg) pedestrians time: " << std::fixed << std::setprecision(2)
                        << (ocv_decode_time_pedestrians / numSyncFrames + ocv_render_time / totalFrames) << " ms";
                } else {
                    out2 << "OpenCV cap/render pedestrians time: " << std::fixed << std::setprecision(2)
                        << (ocv_decode_time_pedestrians + ocv_render_time) << " ms";
                    ocv_render_time = 0;
                }*/
                ocv_decode_time_pedestrians = 0;
                ocv_decode_time_vehicle = 0;

                cv::putText(outputFrame, out1.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
                cv::putText(outputFrame, out2.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0));

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
                    cv::putText(outputFrame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                cv::Scalar(255, 0, 0));

                }

                // -----------------------Display Results ---------------------------------------------
                t0 = std::chrono::high_resolution_clock::now();
                if (!FLAGS_no_show) {
                    cv::imshow("Detection results", outputFrame);
                    lastOutputFrame = &outputFrame;
                }
                t1 = std::chrono::high_resolution_clock::now();
                ocv_render_time += std::chrono::duration_cast<ms>(t1 - t0).count();

                // watch for keypress to stop or snapshot
                int keyPressed;
                if (-1 != (keyPressed = cv::waitKey(1)))
                {
                    if ('s' == keyPressed) {
                        // save screen to output file
                        slog::info << "Saving snapshot of image" << slog::endl;
                        cv::imwrite("snapshot.bmp", outputFrame);
                    } else {
                        haveMoreFrames = false;
                    }
                }

                // done with frame buffer, return to queue
                inputFramePtrs.push(outputFrame2);
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
                    slog::info << "Press 's' key to save a snapshot, press any other key to exit" << slog::endl;
                    while (cv::waitKey(0) == 's') {
                        // save screen to output file
                        slog::info << "Saving snapshot of image" << slog::endl;
                        cv::imwrite("snapshot.bmp", *lastOutputFrame);
                    }
                    haveMoreFrames = false;
                    break;
                }
            }
            int clear = std::system("clear");
        } while(!done);

        // calculate total run time
        ms total_wallclock_time = std::chrono::duration_cast<ms>(wallclockEnd - wallclockStart);

        // report loop time
        slog::info << "     Total main-loop time:" << std::fixed << std::setprecision(2)
                << total_wallclock_time.count() << " ms " <<  slog::endl;
        slog::info << "           Total # frames:" << totalFrames <<  slog::endl;
        float avgTimePerFrameMs = total_wallclock_time.count() / (float)totalFrames;
        slog::info << "   Average time per frame:" << std::fixed << std::setprecision(2)
                    << avgTimePerFrameMs << " ms "
                    << "(" << 1000.0F / avgTimePerFrameMs << " fps)" << slog::endl;

        // ---------------------------Some perf data--------------------------------------------------
        if (FLAGS_pc) {
            VehicleDetection.printPerformanceCounts();
        }

        delete [] inputFrames;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}

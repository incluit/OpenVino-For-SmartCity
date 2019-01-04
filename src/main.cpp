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

#include <opencv2/opencv.hpp>
#include "customflags.hpp"
#include "drawer.hpp"

#include "Tracker.h"
#include "object_detection.hpp"

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

    if (FLAGS_m.empty()) {
        throw std::invalid_argument("Parameter -m is not set");
    }

    if (FLAGS_auto_resize) {
	    slog::warn << "auto_resize=1, forcing all batch sizes to 1" << slog::endl;
	    FLAGS_n = 1;
	    FLAGS_n_va = 1;
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
            {FLAGS_d, FLAGS_m}, {FLAGS_d_p, FLAGS_m_p}
        };

        const bool runningAsync = (FLAGS_n_async > 1);
        slog::info << "FLAGS_n_async=" << FLAGS_n_async << ", inference pipeline will operate "
                << (runningAsync ? "asynchronously" : "synchronously")
                << slog::endl;

        ObjectDetection VehicleDetection(FLAGS_m, FLAGS_d, "Vehicle Detection", FLAGS_n, FLAGS_n_async, FLAGS_auto_resize, FLAGS_t);
        ObjectDetection PedestriansDetection(FLAGS_m_p, FLAGS_d_p, "Pedestrians Detection", FLAGS_n_p, FLAGS_n_async, FLAGS_auto_resize, FLAGS_t);

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

        // structure to hold frame and associated data which are passed along
        //  from stage to stage for each to do its work
        typedef struct {
            std::vector<cv::Mat*> batchOfInputFrames;
            bool vehicleDetectionDone;
            bool pedestriansDetectionDone;
            cv::Mat* outputFrame;
            std::vector<cv::Rect> vehicleLocations;
            int numVehiclesInferred;
            std::vector<cv::Rect> pedestriansLocations;
            int numPedestriansInferred;
        } FramePipelineFifoItem;
        typedef std::queue<FramePipelineFifoItem> FramePipelineFifo;
        // Queues to pass information across pipeline stages
        FramePipelineFifo pipeS0Fifo;
        FramePipelineFifo pipeS0toS1Fifo;
        FramePipelineFifo pipeS0toS2Fifo;
        FramePipelineFifo pipeS1toS2Fifo;
        FramePipelineFifo pipeS2toS3Fifo;
        FramePipelineFifo pipeS3toS4Fifo;
        FramePipelineFifo pipeS1toS4Fifo;

        wallclockStart = std::chrono::high_resolution_clock::now();
        /** Start inference & calc performance **/
        do {
            ms detection_time;
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

                    totalFrames++;

                    ps0.batchOfInputFrames.push_back(curFrame);
                    if (firstFrame && !FLAGS_no_show) {
                        slog::info << "Press 's' key to save a snapshot, press any other key to stop" << slog::endl;
                    }

                    firstFrame = false;
                }
                
                pipeS0Fifo.push(ps0);
            }
            /* *** Pipeline Stage 0: Prepare and Start Inferring a Batch of Frames *** */
            // if there are more frames to do and a request available, then prepare and start batch
            if (!pipeS0Fifo.empty() && VehicleDetection.canSubmitRequest()) {
                // prepare a batch of frames
                // MAKE SLOG std::cout << "STAGE 0.1 - OK"  << std::endl;
                FramePipelineFifoItem ps0i = pipeS0Fifo.front();
                pipeS0Fifo.pop();
                
                for(auto &&  i: ps0i.batchOfInputFrames){
				    cv::Mat* curFrame = i;
                    t0 = std::chrono::high_resolution_clock::now();
                    VehicleDetection.enqueue(*curFrame);
                    t1 = std::chrono::high_resolution_clock::now();
                    ocv_decode_time_vehicle += std::chrono::duration_cast<ms>(t1 - t0).count();
                    // queue frame for next pipeline stage
                }
                // ----------------------------Run Vehicle detection inference------------------------------------------
                // if there are frames to be processed, then submit the request
                if (numFrames > 0) {
                    numSyncFrames = numFrames;
                    // start request
                    t0 = std::chrono::high_resolution_clock::now();
                    // start inference
                    VehicleDetection.submitRequest();
                    // queue data for next pipeline stage
                    pipeS0toS1Fifo.push(ps0i);
                    pipeS0toS2Fifo.push(ps0i);
                }
            }

            /* *** Pipeline Stage 1: Process Vehicles Inference Results *** */
            // sync: wait for results if a request was just submitted
            // async: if results are ready, then fetch and process in next stage of pipeline
            if ((!runningAsync && VehicleDetection.requestsInProcess()) || VehicleDetection.resultIsReady()) {
                // wait for results, async will be ready
                // MAKE SLOG std::cout << "STAGE 1.1 - OK"  << std::endl;

                VehicleDetection.wait();
                t1 = std::chrono::high_resolution_clock::now();
                detection_time = std::chrono::duration_cast<ms>(t1 - t0);

                // get associated data from last pipeline stage to use with results
                FramePipelineFifoItem ps0s1i = pipeS0toS1Fifo.front();
                pipeS0toS1Fifo.pop();
                // parse inference results internally (e.g. apply a threshold, etc)
                VehicleDetection.fetchResults(ps0s1i.batchOfInputFrames.size());

                // prepare a FramePipelineFifoItem for each batched frame to get its detection results
                std::vector<FramePipelineFifoItem> batchedFifoItems;
                for (auto && bFrame : ps0s1i.batchOfInputFrames) {
                    FramePipelineFifoItem fpfi;
                    fpfi.outputFrame = bFrame;
                    batchedFifoItems.push_back(fpfi);
                }

                // store results for next pipeline stage
                for (auto && result : VehicleDetection.results) {
                    FramePipelineFifoItem& fpfi = batchedFifoItems[result.batchIndex];
                    fpfi.vehicleLocations.push_back(result.location);
                }

                // done with results, clear them
                VehicleDetection.results.clear();

                // queue up output for next pipeline stage to process
                for (auto && item : batchedFifoItems) {
                    item.numVehiclesInferred = 0;
                    item.vehicleDetectionDone = true;
                    item.pedestriansDetectionDone = false;
                    pipeS1toS4Fifo.push(item);
                }
            }

            /* *** Pipeline Stage 2: Start Inferring Vehicle Attributes *** */
            // ----------------------------Process the results down the pipeline---------------------------------
            if (PedestriansDetection.enabled()) {
                if (!pipeS0toS2Fifo.empty() && PedestriansDetection.canSubmitRequest()) {
                    //MAKE SLOG std::cout << "STAGE 2.1 - OK"  << std::endl;
                    // grab reference to first item in FIFO, but do not pop until done inferring all vehicles in it
                    FramePipelineFifoItem& ps0s2i = pipeS0toS2Fifo.front();
                    for(numFrames = 0; numFrames < PedestriansDetection.maxBatch; numFrames++) {
                        // read in a frame
                        for (auto && bFrame : ps0s2i.batchOfInputFrames) {
                            cv::Mat* curFrame = bFrame;

                            t0 = std::chrono::high_resolution_clock::now();
                            PedestriansDetection.enqueue(*curFrame);
                            t1 = std::chrono::high_resolution_clock::now();
                            ocv_decode_time_pedestrians += std::chrono::duration_cast<ms>(t1 - t0).count();

						}
                    }

                    // ----------------------------Run Pedestrians detection inference------------------------------------------
                    // if there are frames to be processed, then submit the request
                    if (numFrames > 0) {


                        numSyncFrames = numFrames;
                        // start request
                        t0 = std::chrono::high_resolution_clock::now();
                        // start inference
                        PedestriansDetection.submitRequest();

                        // queue data for next pipeline stage
                        pipeS2toS3Fifo.push(ps0s2i);
                        pipeS0toS2Fifo.pop();
                    }
                }
            } else {
                // not running vehicle attributes, just pass along frames
                if (!pipeS0toS2Fifo.empty()) {
                    FramePipelineFifoItem fpfi = pipeS0toS2Fifo.front();
                    pipeS0toS2Fifo.pop();
                    fpfi.pedestriansDetectionDone = true;
                    pipeS2toS3Fifo.push(fpfi);
                }
            }

            /* *** Pipeline Stage 3: Process Vehicle Attribute Inference Results *** */
            if (PedestriansDetection.enabled()) {
                // MAKE SLOG std::cout << "STAGE 3.1 - OK"  << std::endl;
                if (!pipeS2toS3Fifo.empty() && ((!runningAsync && PedestriansDetection.requestsInProcess()) || PedestriansDetection.resultIsReady())) {
                    // wait for results, async will be ready

                    // MAKE SLOG std::cout << "STAGE 3.2 - OK"  << std::endl;
                    PedestriansDetection.wait();
                    t1 = std::chrono::high_resolution_clock::now();
                    detection_time = std::chrono::duration_cast<ms>(t1 - t0);

                    // get associated data from last pipeline stage to use with results
                    FramePipelineFifoItem ps2s3i = pipeS2toS3Fifo.front();
                    pipeS2toS3Fifo.pop();

                    // parse inference results internally (e.g. apply a threshold, etc)
                    PedestriansDetection.fetchResults(ps2s3i.batchOfInputFrames.size());

                    // prepare a FramePipelineFifoItem for each batched frame to get its detection results
                    std::vector<FramePipelineFifoItem> batchedFifoItems;
                    for (auto && bFrame : ps2s3i.batchOfInputFrames) {
                        FramePipelineFifoItem fpfi;
                        fpfi.outputFrame = bFrame;
                        batchedFifoItems.push_back(fpfi);
                    }

                    // store results for next pipeline stage
                    for (auto && result : PedestriansDetection.results) {
                        FramePipelineFifoItem& fpfi = batchedFifoItems[result.batchIndex];
                        fpfi.pedestriansLocations.push_back(result.location);
                    }

                    // done with results, clear them
                    PedestriansDetection.results.clear();

                    // queue up output for next pipeline stage to process
                    for (auto && item : batchedFifoItems) {
                        item.batchOfInputFrames.clear(); // done with batch storage
                        item.numPedestriansInferred = 0;
                        item.pedestriansDetectionDone = true;
                        pipeS3toS4Fifo.push(item);
                    }
                }
            } else {
                // not running pedestrians locations, just pass along frames
                if (!pipeS2toS3Fifo.empty()) {
                    FramePipelineFifoItem fpfi = pipeS2toS3Fifo.front();
                    pipeS2toS3Fifo.pop();
                    fpfi.pedestriansDetectionDone = true;
                    pipeS3toS4Fifo.push(fpfi);
                }
            }

            /* *** Pipeline Stage 4: Render Results *** */
            if (!pipeS3toS4Fifo.empty() && !pipeS1toS4Fifo.empty()) {
                // MAKE SLOG std::cout << "STAGE 4.1 - OK"  << std::endl;

                FramePipelineFifoItem ps3s4i = pipeS3toS4Fifo.front();
                pipeS3toS4Fifo.pop();
                FramePipelineFifoItem ps1s4i = pipeS1toS4Fifo.front();
                pipeS1toS4Fifo.pop();

                cv::Mat& outputFrame = *(ps3s4i.outputFrame);

                // draw box around vehicles
                for (auto && loc : ps1s4i.vehicleLocations) {
                    cv::rectangle(outputFrame, loc, cv::Scalar(0, 255, 0), 1);
                    if (firstFrameWithDetections){
                        firstResults.push_back(std::make_pair(loc, LABEL_CAR));
                    }
                }
                // draw box around pedestrians
                for (auto && loc : ps3s4i.pedestriansLocations) {
                    cv::rectangle(outputFrame, loc, cv::Scalar(255, 255, 255), 1);
                    if (firstFrameWithDetections){
                        firstResults.push_back(std::make_pair(loc, LABEL_PERSON));
                    }
                }

		if(FLAGS_tracking) {
			if(firstFrameWithDetections){
				tracking_system.setFrameWidth(outputFrame.cols);
				tracking_system.setFrameHeight(outputFrame.rows);
				tracking_system.setInitTarget(firstResults);
				tracking_system.initTrackingSystem();
			}
			int tracking_success = tracking_system.startTracking(outputFrame);
			if (tracking_success == FAIL){
				break;
			}
			if (tracking_system.getTrackerManager().getTrackerVec().size() != 0){
				tracking_system.drawTrackingResult(outputFrame);
				tracking_system.detectCollisions(outputFrame);
			}
			firstFrameWithDetections = false;
		}
		// ----------------------------Execution statistics -----------------------------------------------------
                std::ostringstream out;
				std::ostringstream out1;
				std::ostringstream out2;

				if (VehicleDetection.maxBatch > 1) {
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
                }
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
                inputFramePtrs.push(ps3s4i.outputFrame);
            }

            // wait until break from key press after all pipeline stages have completed
            done = !haveMoreFrames && pipeS0toS1Fifo.empty() && pipeS1toS2Fifo.empty() && pipeS2toS3Fifo.empty()
                        && pipeS3toS4Fifo.empty() && pipeS0toS2Fifo.empty();
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

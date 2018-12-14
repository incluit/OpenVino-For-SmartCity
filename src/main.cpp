/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>
#include "car_detection.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_auto_resize) {
    	slog::warn << "auto_resize=1, forcing all batch sizes to 1" << slog::endl;
    	FLAGS_n = 1;
    	FLAGS_n_va = 1;
    }
	
    if (FLAGS_n_async < 1) {
        throw std::logic_error("Parameter -n_async must be >= 1");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ExecutableNetwork net;
    InferenceEngine::InferencePlugin * plugin;

    std::queue<InferRequest::Ptr> submittedRequests;
    std::vector<InferRequest::Ptr> requests;
    int inputRequestIdx;
    InferRequest::Ptr outputRequest;
    std::string & commandLineFlag;
    std::string topoName;
    int maxBatch;
    int maxSubmittedRequests;

    BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch)
        : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch), maxSubmittedRequests(FLAGS_n_async),
		  plugin(nullptr), inputRequestIdx(0), outputRequest(nullptr), requests(FLAGS_n_async) {}

    virtual ~BaseDetection() {}

    ExecutableNetwork* operator ->() {
        return &net;
    }
    virtual InferenceEngine::CNNNetwork read()  = 0;

    virtual void submitRequest() {
        if (!enabled() || nullptr == requests[inputRequestIdx]) return;
        requests[inputRequestIdx]->StartAsync();
        submittedRequests.push(requests[inputRequestIdx]);
        inputRequestIdx++;
        if (inputRequestIdx >= maxSubmittedRequests) {
        	inputRequestIdx = 0;
        }
    }

    // call before wait() to check status
    bool resultIsReady() {
    	if (submittedRequests.size() < 1) return false;
    	StatusCode state = submittedRequests.front()->Wait(IInferRequest::WaitMode::STATUS_ONLY);
		return (StatusCode::OK == state);
    }

    virtual void wait() {
        if (!enabled()) return;

        // get next request to wait on
        if (nullptr == outputRequest) {
        	if (submittedRequests.size() < 1) return;
        	outputRequest = submittedRequests.front();
        	submittedRequests.pop();
        }

        outputRequest->Wait(IInferRequest::WaitMode::RESULT_READY);
    }

    bool requestsInProcess() {
    	// request is in progress if number of outstanding requests is > 0
    	return (submittedRequests.size() > 0);
    }

    bool canSubmitRequest() {
    	// ready when another request can be submitted
    	return (submittedRequests.size() < maxSubmittedRequests);
    }

    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
    void printPerformanceCounts() {
        if (!enabled()) {
            return;
        }
        // use last request used
        int idx = std::max(0, inputRequestIdx-1);
        slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
        ::printPerformanceCounts(requests[idx]->GetPerformanceCounts(), std::cout, false);
    }
};

struct VehicleDetection : BaseDetection{
    std::string input;
    std::string output;
    int maxProposalCount = 0;
    int objectSize = 0;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    using BaseDetection::operator=;

    struct Result {
    	int batchIndex;
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        if (!enquedFrames) return;
        enquedFrames = 0;
        BaseDetection::submitRequest();
    }

    void enqueue(const cv::Mat &frame) {
        if (!enabled()) return;

        if (enquedFrames >= maxBatch) {
            slog::warn << "Number of frames more than maximum(" << maxBatch << ") processed by Vehicles detector" << slog::endl;
            return;
        }

        if (nullptr == requests[inputRequestIdx]) {
        	requests[inputRequestIdx] = net.CreateInferRequestPtr();
        }

        width = frame.cols;
        height = frame.rows;

		InferenceEngine::Blob::Ptr inputBlob;
        if (FLAGS_auto_resize) {
            inputBlob = wrapMat2Blob(frame);
            requests[inputRequestIdx]->SetBlob(input, inputBlob);
        } else {
			inputBlob = requests[inputRequestIdx]->GetBlob(input);
			matU8ToBlob<uint8_t >(frame, inputBlob, enquedFrames);
    	}
        enquedFrames++;
    }


    VehicleDetection() : BaseDetection(FLAGS_m, "Vehicle Detection", FLAGS_n) {}
    InferenceEngine::CNNNetwork read() override {
        slog::info << "Loading network files for VehicleDetection" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Vehicle Detection" << slog::endl;

        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Vehicle Detection inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        
		if (FLAGS_auto_resize) {
	        // set resizing algorithm
	        inputInfoFirst->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
			inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
		} else {
			inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
		}

        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Vehicle Detection outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }
        auto& _output = outputInfo.begin()->second;
        const InferenceEngine::SizeVector outputDims = _output->dims;
        output = outputInfo.begin()->first;
        maxProposalCount = outputDims[1];
        objectSize = outputDims[0];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        slog::info << "Loading Vehicle Detection model to the "<< FLAGS_d << " plugin" << slog::endl;
        input = inputInfo.begin()->first;
        return netReader.getNetwork();
    }

    void fetchResults(int inputBatchSize) {
        if (!enabled()) return;

        if (nullptr == outputRequest) {
        	return;
        }

        results.clear();

        const float *detections = outputRequest->GetBlob(output)->buffer().as<float *>();
        // pretty much regular SSD post-processing
		for (int i = 0; i < maxProposalCount; i++) {
			int proposalOffset = i * objectSize;
			float image_id = detections[proposalOffset + 0];
			Result r;
			r.batchIndex = image_id;
			r.label = static_cast<int>(detections[proposalOffset + 1]);
			r.confidence = detections[proposalOffset + 2];
			if (r.confidence <= FLAGS_t) {
				continue;
			}
			r.location.x = detections[proposalOffset + 3] * width;
			r.location.y = detections[proposalOffset + 4] * height;
			r.location.width = detections[proposalOffset + 5] * width - r.location.x;
			r.location.height = detections[proposalOffset + 6] * height - r.location.y;

			if ((image_id < 0) || (image_id >= inputBatchSize)) {  // indicates end of detections
				break;
			}
			if (FLAGS_r) {
				std::cout << "[bi=" << r.batchIndex << "][" << i << "," << r.label << "] element, prob = " << r.confidence <<
						  "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
						  << r.location.height << ")"
						  << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
			}
			results.push_back(r);
		}
		// done with request
		outputRequest = nullptr;
    }
};

struct VehicleAttribsDetection : BaseDetection {
    std::string inputName;
    std::string outputNameForType;
    std::string outputNameForColor;
    int enquedVehicles = 0;

    using BaseDetection::operator=;
    VehicleAttribsDetection() : BaseDetection(FLAGS_m_va, "Vehicle Attribs", FLAGS_n_va) {}

    void submitRequest() override {
        if (!enquedVehicles) return;
		
        // Use Dynamic Batching to set actual number of inputs in request
        if (FLAGS_dyn_va) {
        	requests[inputRequestIdx]->SetBatch(enquedVehicles);
        }

        BaseDetection::submitRequest();
        enquedVehicles = 0;
    }

    void enqueue(const cv::Mat &Vehicle, cv::Rect roiRect = cv::Rect(0,0,0,0)) {
        if (!enabled()) {
            return;
        }
        if (enquedVehicles >= maxBatch) {
            slog::warn << "Number of detected vehicles more than maximum(" << maxBatch << ") processed by Vehicles Attributes detector" << slog::endl;
            return;
        }
        if (nullptr == requests[inputRequestIdx]) {
        	requests[inputRequestIdx] = net.CreateInferRequestPtr();
        }

        // check if ROI has non-zero dimensions, if so it will be used to crop input
        bool doCrop = (roiRect.width != 0) && (roiRect.height != 0);

        InferenceEngine::Blob::Ptr inputBlob;
        if (FLAGS_auto_resize) {
        	inputBlob = wrapMat2Blob(Vehicle);
        	if (doCrop) {
        		cv::Rect clippedRect = roiRect & cv::Rect(0, 0, Vehicle.cols, Vehicle.rows);
				InferenceEngine::ROI ieRoi;
				ieRoi.posX = clippedRect.x;
				ieRoi.posY = clippedRect.y;
				ieRoi.sizeX = clippedRect.width;
				ieRoi.sizeY = clippedRect.height;

				inputBlob = InferenceEngine::make_shared_blob(inputBlob, ieRoi);
        	}

            requests[inputRequestIdx]->SetBlob(inputName, inputBlob);
        } else {
        	cv::Mat cropped;
        	if (doCrop) {
        		cv::Rect clippedRect = roiRect & cv::Rect(0, 0, Vehicle.cols, Vehicle.rows);
        		cropped = Vehicle(clippedRect);
        	} else {
        		cropped = Vehicle;
        	}
        	inputBlob = requests[inputRequestIdx]->GetBlob(inputName);
			matU8ToBlob<uint8_t >(cropped, inputBlob, enquedVehicles);
    	}
        enquedVehicles++;
    }
	
    struct Attributes { std::string type; std::string color;};

    std::vector<Attributes> results;

    void fetchResults() {
        static const std::string colors[] = {
                "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
                "car", "van", "truck", "bus"
        };

        if (nullptr == outputRequest) {
        	return;
        }

        results.clear();

        for (int bi = 0; bi < maxBatch; bi++) {
			// 7 possible colors for each vehicle and we should select the one with the maximum probability
			const auto colorsValues = outputRequest->GetBlob(outputNameForColor)->buffer().as<float*>() + (bi * 7);
			// 4 possible types for each vehicle and we should select the one with the maximum probability
			const auto typesValues  = outputRequest->GetBlob(outputNameForType)->buffer().as<float*>() + (bi * 4);

			const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
			const auto type_id =  std::max_element(typesValues,  typesValues  + 4) - typesValues;

			Attributes attrib( { types[type_id], colors[color_id] } );

			if (FLAGS_r) {
				std::cout << "[Vehicle Attribute detected: type=" << attrib.type << ","
						<< "color=" << attrib.color << "]" << std::endl;
			}
			results.push_back(attrib);
        }
		// done with request
		outputRequest = nullptr;
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for VehicleAttribs" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_va);
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Vehicle Attribs" << slog::endl;

        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_va) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** Vehicle Attribs network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking VehicleAttribs inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);

		if (FLAGS_auto_resize) {
	        // set resizing algorithm
	        inputInfoFirst->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
			inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
		} else {
			inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
		}

        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking VehicleAttribs outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }
        auto it = outputInfo.begin();
        outputNameForColor = (it++)->second->name;  // color is the first output
        outputNameForType = (it++)->second->name;  // type is the second output

        slog::info << "Loading Vehicle Attribs model to the "<< FLAGS_d_va << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferenceEngine::InferencePlugin & plg, bool enable_dynamic_batch = false) const {
        if (detector.enabled()) {
            std::map<std::string, std::string> config;
            // if specified, enable Dynamic Batching
            if (enable_dynamic_batch) {
                config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
            }
            detector.net = plg.LoadNetwork(detector.read(), config);
            detector.plugin = &plg;
        }
    }
};

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
        const bool isCamera = FLAGS_i == "cam";
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // ---------------------Load plugins for inference engine------------------------------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_va, FLAGS_m_va}
        };

        const bool runningAsync = (FLAGS_n_async > 1);
        slog::info << "FLAGS_n_async=" << FLAGS_n_async << ", inference pipeline will operate "
        		<< (runningAsync ? "asynchronously" : "synchronously")
        		<< slog::endl;

        VehicleDetection VehicleDetection;
        VehicleAttribsDetection VehicleAttribs;
		
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
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Load extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }

            pluginsForDevices[deviceName] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }

        // --------------------Load networks (Generated xml/bin files)-------------------------------------------
        Load(VehicleDetection).into(pluginsForDevices[FLAGS_d], false);
        Load(VehicleAttribs).into(pluginsForDevices[FLAGS_d_va], FLAGS_dyn_va);

        // read input (video) frames, need to keep multiple frames stored
        //  for batching and for when using asynchronous API.
        const int maxNumInputFrames = FLAGS_n_async * VehicleDetection.maxBatch + 1;  // +1 to avoid overwrite
        cv::Mat* inputFrames = new cv::Mat[maxNumInputFrames];
        std::queue<cv::Mat*> inputFramePtrs;
        for(int fi = 0; fi < maxNumInputFrames; fi++) {
        	inputFramePtrs.push(&inputFrames[fi]);
        }

        // ----------------------------Do inference-------------------------------------------------------------
        slog::info << "Start inference " << slog::endl;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        std::chrono::high_resolution_clock::time_point wallclockStart, wallclockEnd;

        bool firstFrame = true;
        bool haveMoreFrames = true;
        bool done = false;
        int numFrames = 0;
        int numSyncFrames = 0;
        int totalFrames = 0;
		double ocv_decode_time = 0, ocv_render_time = 0;
		cv::Mat* lastOutputFrame;

		// structure to hold frame and associated data which are passed along
		//  from stage to stage for each to do its work
		typedef struct {
			std::vector<cv::Mat*> batchOfInputFrames;
			bool vehicleDetectionDone;
			cv::Mat* outputFrame;
			std::vector<cv::Rect> vehicleLocations;
			int numVehiclesInferred;
			std::vector<cv::Rect> licensePlateLocations;
			bool vehicleAttributesDetectionDone;
			std::vector<VehicleAttribsDetection::Attributes> vehicleAttributes;
		} FramePipelineFifoItem;
		typedef std::queue<FramePipelineFifoItem> FramePipelineFifo;
		// Queues to pass information across pipeline stages
		FramePipelineFifo pipeS0toS1Fifo;
		FramePipelineFifo pipeS1toS2Fifo;
		FramePipelineFifo pipeS2toS3Fifo;
		FramePipelineFifo pipeS3toS4Fifo;

		FramePipelineFifoItem accumVehAttribs;
		bool accumVehAttribsIsEmpty = true;

		wallclockStart = std::chrono::high_resolution_clock::now();
        /** Start inference & calc performance **/
        do {
        	ms detection_time;
			std::chrono::high_resolution_clock::time_point t0,t1;

			/* *** Pipeline Stage 0: Prepare and Start Inferring a Batch of Frames *** */
        	// if there are more frames to do and a request available, then prepare and start batch
			if (haveMoreFrames && (inputFramePtrs.size() >= VehicleDetection.maxBatch) && VehicleDetection.canSubmitRequest()) {
				// prepare a batch of frames
        		FramePipelineFifoItem ps0s1i;
				for(numFrames = 0; numFrames < VehicleDetection.maxBatch; numFrames++) {
					// read in a frame
					cv::Mat* curFrame = inputFramePtrs.front();
					inputFramePtrs.pop();
					haveMoreFrames = cap.read(*curFrame);
					if (!haveMoreFrames) {
						break;
					}

					totalFrames++;

					t0 = std::chrono::high_resolution_clock::now();
					VehicleDetection.enqueue(*curFrame);
					t1 = std::chrono::high_resolution_clock::now();
					ocv_decode_time += std::chrono::duration_cast<ms>(t1 - t0).count();

					// queue frame for next pipeline stage
					ps0s1i.batchOfInputFrames.push_back(curFrame);

					if (firstFrame && !FLAGS_no_show) {
						slog::info << "Press 's' key to save a snapshot, press any other key to stop" << slog::endl;
					}

					firstFrame = false;
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
					pipeS0toS1Fifo.push(ps0s1i);
				}
        	}

			/* *** Pipeline Stage 1: Process Vehicles Inference Results *** */
			// sync: wait for results if a request was just submitted
			// async: if results are ready, then fetch and process in next stage of pipeline
			if ((!runningAsync && VehicleDetection.requestsInProcess()) || VehicleDetection.resultIsReady()) {
        		// wait for results, async will be ready
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
					if (result.label == 1) {  // vehicle
						fpfi.vehicleLocations.push_back(result.location);
					} else { // license plate
						fpfi.licensePlateLocations.push_back(result.location);
					}
				}

				// done with results, clear them
				VehicleDetection.results.clear();

				// queue up output for next pipeline stage to process
				for (auto && item : batchedFifoItems) {
					item.batchOfInputFrames.clear(); // done with batch storage
					item.numVehiclesInferred = 0;
					item.vehicleDetectionDone = true;
					item.vehicleAttributesDetectionDone = false;
					pipeS1toS2Fifo.push(item);
				}
        	}

			/* *** Pipeline Stage 2: Start Inferring Vehicle Attributes *** */
            // ----------------------------Process the results down the pipeline---------------------------------
            ms AttribsNetworkTime(0);
            int AttribsInferred = 0;
			if (VehicleAttribs.enabled()) {
				if (!pipeS1toS2Fifo.empty() && VehicleAttribs.canSubmitRequest()) {
					// grab reference to first item in FIFO, but do not pop until done inferring all vehicles in it
					FramePipelineFifoItem& ps1s2i = pipeS1toS2Fifo.front();

					const int totalVehicles = ps1s2i.vehicleLocations.size();

					// enqueue input batch
					for(int rib = ps1s2i.numVehiclesInferred; rib < totalVehicles; rib++) {
						if (VehicleAttribs.enquedVehicles >= VehicleAttribs.maxBatch) {
							break;
						}
						VehicleAttribs.enqueue(*ps1s2i.outputFrame, ps1s2i.vehicleLocations[rib]);
					}

					// ----------------------------Run vehicleResult attribute inference ----------------
					// if there are vehicles to infer, submit a request to start
					if (VehicleAttribs.enquedVehicles > 0) {
						// track how many vehicles have been inferred
						ps1s2i.numVehiclesInferred += VehicleAttribs.enquedVehicles;
						AttribsInferred += VehicleAttribs.enquedVehicles;

						t0 = std::chrono::high_resolution_clock::now();
						VehicleAttribs.submitRequest();

					}
					// make a copy before sending out
					FramePipelineFifoItem ps2s3out(ps1s2i);
					if ( ps2s3out.numVehiclesInferred >= totalVehicles) {
						// done with input FIFO item, pop from input FIFO
						pipeS1toS2Fifo.pop();
					}

					// always queue frame data for next pipeline stage to handle results even without doing inference
					pipeS2toS3Fifo.push(ps2s3out);
				}
			} else {
				// not running vehicle attributes, just pass along frames
				if (!pipeS1toS2Fifo.empty()) {
					FramePipelineFifoItem fpfi = pipeS1toS2Fifo.front();
					pipeS1toS2Fifo.pop();
					fpfi.vehicleAttributesDetectionDone = true;
					pipeS2toS3Fifo.push(fpfi);
				}
			}

			/* *** Pipeline Stage 3: Process Vehicle Attribute Inference Results *** */
			if (VehicleAttribs.enabled()) {
				if (!pipeS2toS3Fifo.empty()) {
					FramePipelineFifoItem ps2s3i = pipeS2toS3Fifo.front();
					int numVehicles = ps2s3i.vehicleLocations.size();

					if ( 0 == numVehicles) {
						// no vehicles are being inferred for this frame, we are done with it
						pipeS2toS3Fifo.pop();
						// queue frame data for next pipeline stage to handle results
						ps2s3i.vehicleAttributesDetectionDone = true;
						pipeS3toS4Fifo.push(ps2s3i);
					} else {
						// expecting inference results, check for them while accumulating results
						// if first FIFO item for results, initialize from first item
						if (accumVehAttribsIsEmpty) {
							accumVehAttribs = ps2s3i;
							accumVehAttribsIsEmpty = false;
						}
						if ((!runningAsync && VehicleAttribs.requestsInProcess()) || VehicleAttribs.resultIsReady()) {
							// wait for results, async will be ready
							VehicleAttribs.wait();
							t1 = std::chrono::high_resolution_clock::now();
							AttribsNetworkTime += std::chrono::duration_cast<ms>(t1 - t0);

							// ----------------------------Process outputs-----------------------------------------------------
							VehicleAttribs.fetchResults();
							int numVAResuls = VehicleAttribs.results.size();

							int batchIndex = 0;
							while(batchIndex < numVAResuls) {
								VehicleAttribsDetection::Attributes& res = VehicleAttribs.results[batchIndex];
								accumVehAttribs.vehicleAttributes.push_back(res);
								batchIndex++;
							}
							accumVehAttribs.numVehiclesInferred = ps2s3i.numVehiclesInferred;

							// only send out frame data when inference is complete
							if ( accumVehAttribs.numVehiclesInferred >= numVehicles) {
								// queue frame data for next pipeline stage to handle results
								accumVehAttribs.vehicleAttributesDetectionDone = true;
								pipeS3toS4Fifo.push(accumVehAttribs);
								accumVehAttribsIsEmpty = true;
							}
							// done with this FIFO item
							pipeS2toS3Fifo.pop();
						}
					}
				}
			} else {
				// not running vehicle attributes, just pass along frames
				if (!pipeS2toS3Fifo.empty()) {
					FramePipelineFifoItem fpfi = pipeS2toS3Fifo.front();
					pipeS2toS3Fifo.pop();
					fpfi.vehicleAttributesDetectionDone = true;
					pipeS3toS4Fifo.push(fpfi);
				}
			}

			/* *** Pipeline Stage 4: Render Results *** */
			if (!pipeS3toS4Fifo.empty()) {
				FramePipelineFifoItem ps3s4i = pipeS3toS4Fifo.front();
				pipeS3toS4Fifo.pop();

				cv::Mat& outputFrame = *(ps3s4i.outputFrame);

				// draw box around vehicles and license plates
				for (auto && loc : ps3s4i.vehicleLocations) {
					cv::rectangle(outputFrame, loc, cv::Scalar(0, 255, 0), 2);
				}
				// draw box around license plates
				for (auto && loc : ps3s4i.licensePlateLocations) {
					cv::rectangle(outputFrame, loc, cv::Scalar(0, 0, 255), 2);
				}

				// label vehicle attributes
				int numVehicles = ps3s4i.vehicleAttributes.size();
				for(int vi = 0; vi < numVehicles; vi++) {
					VehicleAttribsDetection::Attributes& res = ps3s4i.vehicleAttributes[vi];
					cv::Rect vLoc = ps3s4i.vehicleLocations[vi];
					cv::putText(outputFrame,
								res.color,
								cv::Point2f(vLoc.x, vLoc.y + 15),
								cv::FONT_HERSHEY_COMPLEX_SMALL,
								0.8,
								cv::Scalar(255, 255, 255));
					cv::putText(outputFrame,
								res.type,
								cv::Point2f(vLoc.x, vLoc.y + 30),
								cv::FONT_HERSHEY_COMPLEX_SMALL,
								0.8,
								cv::Scalar(255, 255, 255));
					if (FLAGS_r) {
						std::cout << "Vehicle Attributes results:" << res.color << ";" << res.type << std::endl;
					}
				}

				// ----------------------------Execution statistics -----------------------------------------------------
				std::ostringstream out;
				if (VehicleDetection.maxBatch > 1) {
					out << "OpenCV cap/render (avg) time: " << std::fixed << std::setprecision(2)
						<< (ocv_decode_time / numSyncFrames + ocv_render_time / totalFrames) << " ms";
				} else {
					out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
						<< (ocv_decode_time + ocv_render_time) << " ms";
					ocv_render_time = 0;
				}
				ocv_decode_time = 0;
				cv::putText(outputFrame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

				// When running asynchronously, timing metrics are not accurate so do not display them
				if (!runningAsync) {
					out.str("");
					out << "Vehicle detection time ";
					if (VehicleDetection.maxBatch > 1) {
						out << "(batch size = " << VehicleDetection.maxBatch << ") ";
					}
					out << ": " << std::fixed << std::setprecision(2) << detection_time.count()
						<< " ms ("
						<< 1000.f * numSyncFrames / detection_time.count() << " fps)";
					cv::putText(outputFrame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
								cv::Scalar(255, 0, 0));

					if (VehicleAttribs.enabled() && AttribsInferred > 0) {
						float average_time = AttribsNetworkTime.count() / AttribsInferred;
						out.str("");
						out << "Vehicle Attribs time (averaged over " << AttribsInferred << " detections) :" << std::fixed
							<< std::setprecision(2) << average_time << " ms " << "(" << 1000.f / average_time << " fps)";
						cv::putText(outputFrame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_SIMPLEX, 0.5,
									cv::Scalar(255, 0, 0));
					}
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
						&& pipeS3toS4Fifo.empty();
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
					<< "(" << 1000.0f / avgTimePerFrameMs << " fps)" << slog::endl;

        // ---------------------------Some perf data--------------------------------------------------
        if (FLAGS_pc) {
        	VehicleDetection.printPerformanceCounts();
        	VehicleAttribs.printPerformanceCounts();
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

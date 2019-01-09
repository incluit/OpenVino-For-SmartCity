#pragma once
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

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <ext_list.hpp>

typedef struct {
            std::vector<cv::Mat*> batchOfInputFrames;
            bool vehicleDetectionDone;
            bool pedestriansDetectionDone;
            bool generalDetectionDone;
            cv::Mat* outputFrame;
            std::vector<cv::Rect> vehicleLocations;
            int numVehiclesInferred;
            std::vector<cv::Rect> pedestriansLocations;
            int numPedestriansInferred;
            std::vector<cv::Rect> resultsLocations;
} FramePipelineFifoItem;
typedef std::queue<FramePipelineFifoItem> FramePipelineFifo;


class BaseDetection {
  public:
	InferenceEngine::ExecutableNetwork net;
    InferenceEngine::CNNNetwork net_readed;
    std::string & commandLineFlag;
    std::string & deviceName;
    std::string topoName;
    int maxBatch;
    int maxSubmittedRequests;
    InferenceEngine::InferencePlugin * plugin;
    int inputRequestIdx;
    InferenceEngine::InferRequest::Ptr outputRequest;
    std::vector<InferenceEngine::InferRequest::Ptr> requests;
    std::queue<InferenceEngine::InferRequest::Ptr> submittedRequests;
    bool auto_resize;
    bool next_pipe;
    float detection_threshold;
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;
    FramePipelineFifo S1toS2;

    struct Result {
	    int batchIndex;
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    BaseDetection(std::string &commandLineFlag, std::string &deviceName, std::string topoName, 
                    int maxBatch, int FLAGS_n_async, bool auto_resize, float detection_threshold)
        : commandLineFlag(commandLineFlag), deviceName(deviceName),topoName(topoName), 
            maxBatch(maxBatch), maxSubmittedRequests(FLAGS_n_async), plugin(nullptr), 
            inputRequestIdx(0), outputRequest(nullptr), requests(FLAGS_n_async), 
            auto_resize(auto_resize), detection_threshold(detection_threshold) {}

    virtual ~BaseDetection() {}

    InferenceEngine::ExecutableNetwork* operator ->() {
        return &net;
    }
    virtual InferenceEngine::CNNNetwork read()  = 0;

    virtual void submitRequest();

    // call before wait() to check status
    bool resultIsReady();

    virtual void wait();

    virtual void enqueue(const cv::Mat &frame);

    virtual void fetchResults(int inputBatchSize);

    void run_inferrence(FramePipelineFifo *i);
    void run_inferrence(FramePipelineFifo *i, FramePipelineFifo *o2);

    void wait_results(FramePipelineFifo *o);

    bool requestsInProcess();

    bool canSubmitRequest();

    bool enabled() const;
    void printPerformanceCounts();
};

class Load {
  public:
	BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferenceEngine::InferencePlugin & plg, bool enable_dynamic_batch = false) const {
        if (detector.enabled()) {
            std::map<std::string, std::string> config;
            // if specified, enable Dynamic Batching
            if (enable_dynamic_batch) {
                config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
            }
            detector.net = plg.LoadNetwork(detector.read(), config);
            detector.plugin = &plg;
        }
    }
};
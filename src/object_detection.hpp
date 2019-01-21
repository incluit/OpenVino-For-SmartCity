#pragma once

#include "base_detection.hpp"

class ObjectDetection : public BaseDetection{
  public:
	std::string input;
    std::string output;
    int maxProposalCount = 0;
    int objectSize = 0;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    using BaseDetection::operator=;

    void submitRequest() override;

    void enqueue(const cv::Mat &frame) override;


    ObjectDetection(std::string &commandLineFlag, std::string &deviceName, std::string topoName, 
                    int maxBatch, int n_async, bool auto_resize, float detection_threshold) 
            : BaseDetection(commandLineFlag, deviceName, topoName, maxBatch, n_async, auto_resize, 
                    detection_threshold) {}
    
    InferenceEngine::CNNNetwork read() override;

    void fetchResults(int inputBatchSize) override;
};
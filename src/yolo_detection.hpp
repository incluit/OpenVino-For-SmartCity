#pragma once

#include "base_detection.hpp"

constexpr size_t yolo_scale_13 = 13;
constexpr size_t yolo_scale_26 = 26;
constexpr size_t yolo_scale_52 = 52;

void FrameToBlob(const cv::Mat &frame, InferenceEngine::InferRequest::Ptr &inferRequest, const std::string &inputName);

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry);

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2);

void ParseYOLOV3Output(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects);

class YoloDetection : public BaseDetection{
  public:
	std::string input_name;
    std::vector<std::string> output;
    int maxProposalCount = 0;
    int objectSize = 0;
    int enquedFrames = 0;
    std::vector<std::string> labels;
    float olb_threshold; // overlaping boxes threshold
    std::vector<DetectionObject> detected_results;

    unsigned long resized_im_h = 0;
    unsigned long resized_im_w = 0;
        
    float width = 0;
    float height = 0;
    using BaseDetection::operator=;

    // detection_threshold = FLAGS_t; olb_threshold = FLAGS_iou_t
    YoloDetection(std::string &commandLineFlag, std::string &deviceName, std::string topoName, 
                int maxBatch, int n_async, bool auto_resize, float detection_threshold, 
                float olb_threshold) 
                : BaseDetection(commandLineFlag, deviceName, topoName, maxBatch, n_async, 
                    auto_resize , detection_threshold), olb_threshold(olb_threshold) {}
    
    void submitRequest() override;

    void enqueue(const cv::Mat &frame) override;
    
    InferenceEngine::CNNNetwork read() override ;

    void fetchResults(int inputBatchSize) override;

    
};

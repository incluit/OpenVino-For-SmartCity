#include "yolo_detection.hpp"

void FrameToBlob(const cv::Mat &frame, InferenceEngine::InferRequest::Ptr &inferRequest, const std::string &inputName, bool auto_resize) {
    if (auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        InferenceEngine::Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void ParseYOLOV3Output(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------
    auto num = layer->GetParamAsInt("num");
    num = layer->GetParamAsInts("mask").size();
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,156.0, 198.0, 373.0, 326.0};
    //std::vector<float> anchors = {10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0};
    //std::vector<float> anchors = {};
    //anchors = layer->GetParamAsFloats("anchors");
    anchors = layer->GetParamAsFloats("anchors");
    auto side = out_blob_h;
    int anchor_offset = 0;
    if (anchors.size() == 12){ //yolo_v3-tiny
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    } else if (anchors.size() == 18){
        switch (side) {
            case yolo_scale_13:
                anchor_offset = 2 * 6;
                break;
            case yolo_scale_26:
                anchor_offset = 2 * 3;
                break;
            case yolo_scale_52:
                anchor_offset = 2 * 0;
                break;
            default:
                throw std::runtime_error("Invalid output size");
        }
    }
    
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            //std::cout << scale << std::endl;
            if (scale < threshold){
                continue;
            }
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void YoloDetection::submitRequest() {
    if (! this -> enquedFrames) return;
    this -> enquedFrames = 0;
    this -> BaseDetection::submitRequest();
}

void YoloDetection::enqueue(const cv::Mat &frame) {
    if (!this -> enabled()) return;
    if (this -> enquedFrames >= this -> maxBatch) {
        BOOST_LOG_TRIVIAL(warning) << "Number of frames more than maximum(" << this -> maxBatch << ") processed by Vehicles detector" ;
        return;
    }
    if (nullptr == this -> requests[this -> inputRequestIdx]) {
        this -> requests[this -> inputRequestIdx] = this -> net.CreateInferRequestPtr();
    }
    this -> width = frame.cols;
    this -> height = frame.rows;
	InferenceEngine::Blob::Ptr inputBlob;
    if (this -> auto_resize) {
        inputBlob = wrapMat2Blob(frame);
        this -> requests[this -> inputRequestIdx]->SetBlob(this -> input_name, inputBlob);
    } else {
		inputBlob = this -> requests[this -> inputRequestIdx]->GetBlob(this -> input_name);
		matU8ToBlob<uint8_t >(frame, inputBlob, this -> enquedFrames);
    }
    this -> enquedFrames++;
}

InferenceEngine::CNNNetwork YoloDetection::read() {
    BOOST_LOG_TRIVIAL(info) << "Loading network files" ;
    InferenceEngine::CNNNetReader netReader;
    /** Reading network model **/
    netReader.ReadNetwork(this -> commandLineFlag);
    /** Setting batch size to 1 **/
    BOOST_LOG_TRIVIAL(info) << "Batch size is forced to  1." ;
    netReader.getNetwork().setBatchSize(1);
    /** Extracting the model name and loading its weights **/
    std::string binFileName = fileNameNoExt(this -> commandLineFlag) + ".bin";
    netReader.ReadWeights(binFileName);
    /** Reading labels (if specified) **/
    std::string labelFileName = fileNameNoExt(this -> commandLineFlag) + ".labels";
    std::ifstream inputFile(labelFileName);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(labels));
    // -----------------------------------------------------------------------------------------------------
    /** YOLOV3-based network should have one input and three output **/
    // --------------------------- 3. Configuring input and output -----------------------------------------
    // --------------------------------- Preparing input blobs ---------------------------------------------
    BOOST_LOG_TRIVIAL(info) << "Checking that the inputs are as the demo expects" ;
    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }
    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    this -> resized_im_h = input.get()->getTensorDesc().getDims()[0];
    this -> resized_im_w = input.get()->getTensorDesc().getDims()[1];
    auto inputName = inputInfo.begin()->first;
    this -> input_name = inputName;
    input->setPrecision(InferenceEngine::Precision::U8);
    if (this -> auto_resize) {
        input->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
    } else {
        input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    }
            // --------------------------------- Preparing output blobs -------------------------------------------
    BOOST_LOG_TRIVIAL(info) << "Checking that the outputs are as the demo expects" ;
    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    /*if (outputInfo.size() != 3) {
        throw std::logic_error("This demo only accepts networks with three layers");
    }*/
    for (auto &a : outputInfo) {
        a.second->setPrecision(InferenceEngine::Precision::FP32);
        a.second->setLayout(InferenceEngine::Layout::NCHW);
        this -> output.push_back(a.first);
    }
    this -> net_readed = netReader.getNetwork();
    // -----------------------------------------------------------------------------------------------------
    return this -> net_readed;
}

void YoloDetection::fetchResults(int inputBatchSize) {
    if (!this -> enabled()) return;
    if (nullptr == this -> outputRequest) {
	    return;
    }
    this -> results.clear();
    
    std::vector<DetectionObject> objects;
    // Parsing outputs
    for (auto && i : this -> output) {
        InferenceEngine::CNNLayerPtr layer = net_readed.getLayerByName(i.c_str());
        InferenceEngine::Blob::Ptr blob = outputRequest->GetBlob(i);
        ParseYOLOV3Output(layer, blob, this -> resized_im_h, this -> resized_im_w, this -> height, this -> width, this -> detection_threshold, objects);
    }
    // Filtering overlapping boxes
    std::sort(objects.begin(), objects.end());
    for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].confidence == 0)
            continue;
        for (int j = i + 1; j < objects.size(); ++j)
            if (IntersectionOverUnion(objects[i], objects[j]) >= this ->olb_threshold)
                objects[j].confidence = 0;
    }
    int j = 0;
    for(auto && i : objects){
        Result r;
        if(i.confidence < this -> detection_threshold)
            continue;
        r.batchIndex = 0;
        j++;
        r.label = i.class_id;
        r.confidence = i.confidence;
        r.location = cv::Rect(cv::Point2f(i.xmin,i.ymin), cv::Point2f(i.xmax,i.ymax));
        this -> results.push_back(r);
    }
    this -> outputRequest = nullptr;
}

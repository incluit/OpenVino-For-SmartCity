#include "object_detection.hpp"

void ObjectDetection::submitRequest(){
    if (!this -> enquedFrames) return;
    this -> enquedFrames = 0;
    this -> BaseDetection::submitRequest();
}

void ObjectDetection::enqueue(const cv::Mat &frame) {
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
        this -> requests[this -> inputRequestIdx]->SetBlob(this -> input, inputBlob);
    } else {
		inputBlob = this -> requests[this -> inputRequestIdx]->GetBlob(this -> input);
		matU8ToBlob<uint8_t >(frame, inputBlob, this -> enquedFrames);
	}
    this -> enquedFrames++;
}

InferenceEngine::CNNNetwork ObjectDetection::read() {
    BOOST_LOG_TRIVIAL(info) << "Loading network files for " << this -> topoName ;
    InferenceEngine::CNNNetReader netReader;
    /** Read network model **/
    netReader.ReadNetwork(this -> commandLineFlag);
    netReader.getNetwork().setBatchSize(this -> maxBatch);
    BOOST_LOG_TRIVIAL(info) << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for " << this -> topoName ;
    /** Extract model name and load it's weights **/
    std::string binFileName = fileNameNoExt(this -> commandLineFlag) + ".bin";
    netReader.ReadWeights(binFileName);
    // -----------------------------------------------------------------------------------------------------
    /** SSD-based network should have one input and one output **/
    // ---------------------------Check inputs ------------------------------------------------------
    BOOST_LOG_TRIVIAL(info) << "Checking " << this -> topoName << " inputs" ;
    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
        std::string msg = this -> topoName + "network should have only one input";
        throw std::domain_error(msg);
    }
    auto& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
	if (this -> auto_resize) {
        // set resizing algorithm
        inputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
		inputInfoFirst->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
	} else {
		inputInfoFirst->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
	}
    // -----------------------------------------------------------------------------------------------------
    // ---------------------------Check outputs ------------------------------------------------------
    BOOST_LOG_TRIVIAL(info) << "Checking " << this -> topoName << " outputs" ;
    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1) {
        std::string msg = this -> topoName + "network should have only one output";
        throw std::domain_error(msg);
    }
    auto& _output = outputInfo.begin()->second;
    const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
    this -> output = outputInfo.begin()->first;
    this -> maxProposalCount = outputDims[2];
    this -> objectSize = outputDims[3];
    if (this -> objectSize != 7) {
        throw std::domain_error("Output should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::domain_error("Incorrect output dimensions for SSD");
    }
    _output->setPrecision(InferenceEngine::Precision::FP32);
    _output->setLayout(InferenceEngine::Layout::NCHW);
    BOOST_LOG_TRIVIAL(info) << "Loading " << this -> topoName << " model to the "<< this -> deviceName << " plugin" ;
    this -> input = inputInfo.begin()->first;
    this -> net_readed = netReader.getNetwork();
    return net_readed;
}

void ObjectDetection::fetchResults(int inputBatchSize) {
    if (!this -> enabled()) return;
    if (nullptr == this -> outputRequest) {
        return;
    }
    this -> results.clear();
    const float *detections = this -> outputRequest->GetBlob(this -> output)->buffer().as<float *>();
    // pretty much regular SSD post-processing
	for (int i = 0; i < this -> maxProposalCount; i++) {
		int proposalOffset = i * this -> objectSize;
		float image_id = detections[proposalOffset + 0];
		Result r;
		r.batchIndex = image_id;
		r.label = static_cast<int>(detections[proposalOffset + 1]);
		r.confidence = detections[proposalOffset + 2];
		if (r.confidence <= this -> detection_threshold) {
			continue;
		}
		r.location.x = detections[proposalOffset + 3] * this -> width;
		r.location.y = detections[proposalOffset + 4] * this -> height;
		r.location.width = detections[proposalOffset + 5] * this -> width - r.location.x;
		r.location.height = detections[proposalOffset + 6] * this -> height - r.location.y;
		if ((image_id < 0) || (image_id >= inputBatchSize)) {  // indicates end of detections
			break;
		}
		this -> results.push_back(r);
	}
	// done with request
	this -> outputRequest = nullptr;
}

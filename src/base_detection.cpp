#include "base_detection.hpp"

void BaseDetection::submitRequest() 
{
    if (! this -> enabled() || nullptr == this -> requests[this -> inputRequestIdx]) return;
    this -> requests[this -> inputRequestIdx]->StartAsync();
    this -> submittedRequests.push(this -> requests[this -> inputRequestIdx]);
    (this -> inputRequestIdx)++;
    if (this-> inputRequestIdx >= this -> maxSubmittedRequests) {
	   this -> inputRequestIdx = 0;
    }
}

bool BaseDetection::resultIsReady() {
   if (this -> submittedRequests.size() < 1) return false;
   InferenceEngine::StatusCode state = this -> submittedRequests.front()->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
   return (InferenceEngine::StatusCode::OK == state);
}

void BaseDetection::wait() 
{
    if (!this -> enabled()) return;
    // get next request to wait on
    if (nullptr == this -> outputRequest) {
        if (this -> submittedRequests.size() < 1) return;
        this -> outputRequest = this -> submittedRequests.front();
        this -> submittedRequests.pop();
    }
    this -> outputRequest->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

bool BaseDetection::requestsInProcess() {
    // request is in progress if number of outstanding requests is > 0
    return (this -> submittedRequests.size() > 0);
}

 bool BaseDetection::canSubmitRequest() {
    // ready when another request can be submitted
    return (this -> submittedRequests.size() < this -> maxSubmittedRequests);
}

bool BaseDetection::enabled() const  {
    if (!this -> enablingChecked) {
        this -> _enabled = !this -> commandLineFlag.empty();
        if (!this -> _enabled) {
            slog::info << this -> topoName << " DISABLED" << slog::endl;
        }
        this -> enablingChecked = true;
    }
    return this -> _enabled;
}
void BaseDetection::printPerformanceCounts() {
    if (!this -> enabled()) {
        return;
    }
    // use last request used
    int idx = std::max(0, (this -> inputRequestIdx)-1);
    slog::info << "Performance counts for " << this -> topoName << slog::endl << slog::endl;
    ::printPerformanceCounts(this -> requests[idx]->GetPerformanceCounts(), std::cout, false);
}

void BaseDetection::enqueue(const cv::Mat &frame){}

void BaseDetection::fetchResults(int inputBatchSize){}

void BaseDetection::run_inferrence(FramePipelineFifo *in_fifo){
    FramePipelineFifo& in = *in_fifo; 
    if (!in.empty() && (this ->canSubmitRequest())) {
        FramePipelineFifoItem ps0i = in.front();
        in.pop();
        for(auto &&  i: ps0i.batchOfInputFrames){
            cv::Mat* curFrame = i;
            //t0 = std::chrono::high_resolution_clock::now();
            this -> enqueue(*curFrame);
            //t1 = std::chrono::high_resolution_clock::now();
            //ocv_decode_time_vehicle += std::chrono::duration_cast<ms>(t1 - t0).count();
        }
        //t0 = std::chrono::high_resolution_clock::now();
        this -> submitRequest();
        this -> S1toS2.push(ps0i);
        this -> next_pipe = true;
    }
}

void BaseDetection::run_inferrence(FramePipelineFifo *i, FramePipelineFifo *o2){
    this -> run_inferrence(i);
    if(this -> next_pipe == true ){
        this -> next_pipe = false;
        FramePipelineFifo& in = this -> S1toS2; 
        FramePipelineFifo& out2 = *o2;
        FramePipelineFifoItem i2 = in.back();
        out2.push(i2);
    }
}


void BaseDetection::wait_results(FramePipelineFifo *o){
    FramePipelineFifo& in = this -> S1toS2; 
    FramePipelineFifo& out = *o; 
    
    if (((this -> maxSubmittedRequests == 1) && this -> requestsInProcess()) || this -> resultIsReady()) {
        this -> wait();
        //t1 = std::chrono::high_resolution_clock::now();
        //detection_time = std::chrono::duration_cast<ms>(t1 - t0);
        FramePipelineFifoItem ps0s1i = in.front();
        in.pop();
        this -> fetchResults(ps0s1i.batchOfInputFrames.size());
        // prepare a FramePipelineFifoItem for each batched frame to get its detection results
        std::vector<FramePipelineFifoItem> batchedFifoItems;
        /*for (auto && bFrame : ps0s1i.batchOfInputFrames_clean) {
            FramePipelineFifoItem fpfi;
            fpfi.outputFrame = bFrame;
            batchedFifoItems.push_back(fpfi);
        }*/

        for(int i = 0; i < ps0s1i.batchOfInputFrames.size(); i++){
            FramePipelineFifoItem fpfi;
            fpfi.outputFrame = ps0s1i.batchOfInputFrames[i];
            fpfi.outputFrame_clean = ps0s1i.batchOfInputFrames_clean[i];
            batchedFifoItems.push_back(fpfi);
        }
        // store results for next pipeline stage
        for (auto && result : this -> results) {
            FramePipelineFifoItem& fpfi = batchedFifoItems[result.batchIndex];
            fpfi.resultsLocations.push_back(std::make_pair(result.location, result.label));
        }
        // done with results, clear them
        this -> results.clear();
        // queue up output for next pipeline stage to process
        for (auto && item : batchedFifoItems) {
            item.numVehiclesInferred = 0;
            item.vehicleDetectionDone = true;
            item.pedestriansDetectionDone = false;
            out.push(item);
        }
    }
}
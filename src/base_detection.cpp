#include "base_detection.hpp"

void BaseDetection::submitRequest() 
{
    if (! this -> enabled() || nullptr == this -> requests[inputRequestIdx]) return;
    this -> requests[this -> inputRequestIdx]->StartAsync();
    this -> submittedRequests.push(this -> requests[this -> inputRequestIdx]);
    this -> inputRequestIdx++;
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
    int idx = std::max(0, this -> inputRequestIdx-1);
    slog::info << "Performance counts for " << this -> topoName << slog::endl << slog::endl;
    ::printPerformanceCounts(this -> requests[idx]->GetPerformanceCounts(), std::cout, false);
}
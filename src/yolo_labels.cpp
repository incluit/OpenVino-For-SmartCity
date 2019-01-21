#include "yolo_labels.hpp"

std::string getLabelStr(int label){
    return  YOLO_LABELS[label];
}

cv::Scalar getLabelColor(int label){
    cv::Scalar color;
    switch (label){
        case LABEL_PERSON:
            color = COLOR_PERSON;
            break;
        case LABEL_CAR:
            color = COLOR_CAR;
            break;
        case LABEL_BUS:
            color = COLOR_BUS;
            break;
        case LABEL_TRUCK:
            color = COLOR_TRUCK;
            break;
        case LABEL_BICYCLE:
            color = COLOR_BIKE;
            break;
        case LABEL_MOTORBIKE:
            color = COLOR_MOTORBIKE;
            break;
        default:
            color = COLOR_UNKNOWN;
            break;
    }
    return color;
}
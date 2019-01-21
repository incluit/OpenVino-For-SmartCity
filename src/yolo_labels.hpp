#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


const int LABEL_PERSON          = 0;
const int LABEL_BICYCLE         = 1;
const int LABEL_CAR             = 2;
const int LABEL_MOTORBIKE       = 3;
const int LABEL_AEROPLANE       = 4;
const int LABEL_BUS             = 5;
const int LABEL_TRAIN           = 6;
const int LABEL_TRUCK           = 7;
const int LABEL_BOAT            = 8;
const int LABEL_TRAFFIC_LIGHT   = 9;
const int LABEL_FIRE_HYDRANT    = 10;
const int LABEL_STOP_SIGN       = 11;
const int LABEL_PARKING_METER   = 12;
const int LABEL_BENCH           = 13;
const int LABEL_BIRD            = 14;
const int LABEL_CAT             = 15;
const int LABEL_DOG             = 16;
const int LABEL_HORSE           = 17;
const int LABEL_SHEEP           = 18;
const int LABEL_COW             = 19;
const int LABEL_ELEPHANT        = 20;
const int LABEL_BEAR            = 21;
const int LABEL_ZEBRA           = 22;
const int LABEL_GIRAFFE         = 23;
const int LABEL_BACKPACK        = 24;
const int LABEL_UMBRELLA        = 25;
const int LABEL_HANDBAG         = 26;
const int LABEL_TIE             = 27;
const int LABEL_SUITCASE        = 28;
const int LABEL_FRISBEE         = 29;
const int LABEL_SKIS            = 30;
const int LABEL_SNOWBOARD       = 31;
const int LABEL_SPORTS_BALL     = 32;
const int LABEL_KITE            = 33;
const int LABEL_BASEBALL_BAT    = 34;
const int LABEL_BASEBALL_GLOVE  = 35;
const int LABEL_SKATEBOARD      = 36;
const int LABEL_SURFBOARD       = 37;
const int LABEL_TENNIS_RACKET   = 38;
const int LABEL_BOTTLE          = 39;
const int LABEL_WINE_GLASS      = 40;
const int LABEL_CUP             = 41;
const int LABEL_FORK            = 42;
const int LABEL_KNIFE           = 43;
const int LABEL_SPOON           = 44;
const int LABEL_BOWL            = 45;
const int LABEL_BANANA          = 46;
const int LABEL_APPLE           = 47;
const int LABEL_SANDWICH        = 48;
const int LABEL_ORANGE          = 49;
const int LABEL_BROCCOLI        = 50;
const int LABEL_CARROT          = 51;
const int LABEL_HOT_DOG         = 52;
const int LABEL_PIZZA           = 53;
const int LABEL_DONUT           = 54;
const int LABEL_CAKE            = 55;
const int LABEL_CHAIR           = 56;
const int LABEL_SOFA            = 57;
const int LABEL_POTTEDPLANT     = 58;
const int LABEL_BED             = 59;
const int LABEL_DININGTABLE     = 60;
const int LABEL_TOILET          = 61;
const int LABEL_TVMONITOR       = 62;
const int LABEL_LAPTOP          = 63;
const int LABEL_MOUSE           = 64;
const int LABEL_REMOTE          = 65;
const int LABEL_KEYBOARD        = 66;
const int LABEL_CELL_PHONE      = 67;
const int LABEL_MICROWAVE       = 68;
const int LABEL_OVEN            = 69;
const int LABEL_TOASTER         = 70;
const int LABEL_SINK            = 71;
const int LABEL_REFRIGERATOR    = 72;
const int LABEL_BOOK            = 73;
const int LABEL_CLOCK           = 74;
const int LABEL_VASE            = 75;
const int LABEL_SCISSORS        = 76;
const int LABEL_TEDDY_BEAR      = 77;
const int LABEL_HAIR_DRIER      = 78;
const int LABEL_TOOTHBRUSH      = 79;
const int LABEL_UNKNOWN         = 99;

const std::vector<std::string> YOLO_LABELS = {
"Person"
,"Bicycle"
,"Car"
,"Motorbike"
,"Aeroplane"
,"Bus"
,"Train"
,"Truck"
,"Boat"
,"Traffic light"
,"Fire hydrant"
,"Stop sign"
,"Parking meter"
,"Bench"
,"Bird"
,"Cat"
,"Dog"
,"Horse"
,"Sheep"
,"Cow"
,"Elephant"
,"Bear"
,"Zebra"
,"Giraffe"
,"Backpack"
,"Umbrella"
,"Handbag"
,"Tie"
,"Suitcase"
,"Frisbee"
,"Skis"
,"Snowboard"
,"Sports ball"
,"Kite"
,"Baseball bat"
,"Baseball glove"
,"Skateboard"
,"Surfboard"
,"Tennis racket"
,"Bottle"
,"Wine glass"
,"Cup"
,"Fork"
,"Knife"
,"Spoon"
,"Bowl"
,"Banana"
,"Apple"
,"Sandwich"
,"Orange"
,"Broccoli"
,"Carrot"
,"Hot dog"
,"Pizza"
,"Donut"
,"Cake"
,"Chair"
,"Sofa"
,"Pottedplant"
,"Bed"
,"Diningtable"
,"Toilet"
,"Tvmonitor"
,"Laptop"
,"Mouse"
,"Remote"
,"Keyboard"
,"Cell phone"
,"Microwave"
,"Oven"
,"Toaster"
,"Sink"
,"Refrigerator"
,"Book"
,"Clock"
,"Vase"
,"Scissors"
,"Yeddy bear"
,"Hair drier"
,"Toothbrush"
};

const cv::Scalar COLOR_TRUCK = cv::Scalar(255, 0, 255);
const cv::Scalar COLOR_BUS = cv::Scalar(255, 0, 0);
const cv::Scalar COLOR_BIKE = cv::Scalar(0, 255, 0);
const cv::Scalar COLOR_MOTORBIKE = cv::Scalar(0, 0, 255);
const cv::Scalar COLOR_UNKNOWN = cv::Scalar(0, 0, 0);
const cv::Scalar COLOR_CAR = cv::Scalar(0, 255, 255);
const cv::Scalar COLOR_PERSON = cv::Scalar(255, 255, 0);

std::string getLabelStr(int label);

cv::Scalar getLabelColor(int label);
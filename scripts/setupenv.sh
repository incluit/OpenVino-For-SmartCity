# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# check for variable set by setupvars.sh in the SDK, need it to find models
: ${InferenceEngine_DIR:?"Must source the setupvars.sh in the SDK to set InferenceEngine_DIR"}

export INTEL_CVSDK_VER=20

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

PROJECT_PATH=$parent_path/../
modelDir=$parent_path/../models/

### OPENVINO VERSION TO COMPILE ###
# 2019 = 2019
# 2020 = 2020 R1
if (echo $INTEL_CVSDK_DIR | grep -q "openvino_2020"); 
then export OPENVINO_VER=2020
else 
    if (echo $INTEL_CVSDK_DIR | grep -q "openvino_2019"); 
    then export OPENVINO_VER=2019
    else
            export OPENVINO_VER=2019
    fi
fi

# Face Detection
modName=person-vehicle-bike-detection-crossroad-0078
export vehicle216=$modelDir/FP16/$modName.xml
export vehicle232=$modelDir/FP32/$modName.xml

modName=frozen_yolo_v3
export yolo16=$parent_path/../data/$modName.xml

export PROJECT_PATH=${PROJECT_PATH}

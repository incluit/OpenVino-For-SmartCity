# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# check for variable set by setupvars.sh in the SDK, need it to find models
: ${InferenceEngine_DIR:?"Must source the setupvars.sh in the SDK to set InferenceEngine_DIR"}

modelDir=$InferenceEngine_DIR/../../intel_models

# Vehicle and License Plates Detection Model
modName=vehicle-license-plate-detection-barrier-0106
export mVLP16=$modelDir/$modName/FP16/$modName.xml
export mVLP32=$modelDir/$modName/FP32/$modName.xml

# Vehicle-only Detection Model used with the batch size exercise
modName=vehicle-detection-adas-0002
export mVDR16=$modelDir/$modName/FP16/$modName.xml
export mVDR32=$modelDir/$modName/FP32/$modName.xml

# Vehicle Attributes Detection Model
modName=vehicle-attributes-recognition-barrier-0039
export mVA16=$modelDir/$modName/FP16/$modName.xml
export mVA32=$modelDir/$modName/FP32/$modName.xml

modName=person-vehicle-bike-detection-crossroad-0078
export vehicle216=$modelDir/$modName/FP16/$modName.xml
export vehicle232=$modelDir/$modName/FP32/$modName.xml

modName=pedestrian-and-vehicle-detector-adas-0001
export pv16=$modelDir/$modName/FP16/$modName.xml
export pv32=$modelDir/$modName/FP32/$modName.xml

modName=person-detection-retail-0013
export person116=$modelDir/$modName/FP16/$modName.xml
export person132=$modelDir/$modName/FP32/$modName.xml

modName=pedestrian-detection-adas-0002
export person216=$modelDir/$modName/FP16/$modName.xml
export person232=$modelDir/$modName/FP32/$modName.xml

modName=frozen_yolo_v3
export yolo16=../data/$modName.xml


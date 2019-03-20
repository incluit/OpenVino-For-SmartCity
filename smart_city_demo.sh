#PBS
DEVICE=$1
FP_MODEL=$2


if [ "$2" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/computer_vision_sdk/bitstreams/a10_vision_design_bitstreams/5-0_PL1_FP11_ResNet.aocx
fi

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/build/intel64/Release/lib/ 
MODEL_ROOT=/opt/intel/computer_vision_sdk/deployment_tools/intel_models
cd build && ./intel64/Release/smart_city_tutorial -m_vp ${MODEL_ROOT}/person-vehicle-bike-detection-crossroad-0078/${FP_MODEL}/person-vehicle-bike-detection-crossroad-0078.xml -d_vp ${DEVICE} -n_async 16 -tracking -collision -i ../data/video82.mp4 -no_show

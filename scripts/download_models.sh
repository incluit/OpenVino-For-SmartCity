# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# check for variable set by setupvars.sh in the SDK, need it to find models
: ${InferenceEngine_DIR:?"Must source the setupvars.sh in the SDK to set InferenceEngine_DIR"}

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

PRECISION="FP32 FP16 INT8 INT1"
models_path=$parent_path/../models/
for prec in $PRECISION
do
	echo $prec
	mkdir -p ${models_path}/$prec
done

path_to_open_model_zoo=`find ${InferenceEngine_DIR}/../../../ -name open_model_zoo`
path_to_downloader=`find ${path_to_open_model_zoo} -name downloader.py`

echo ${InferenceEngine_DIR}
echo ${parent_path}
echo ${path_to_open_model_zoo}
echo ${path_to_downloader}

mkdir -p ${models_path}/intel
python3 ${path_to_downloader} --list ${parent_path}/list.lst -o ${models_path}/intel

for FP in $PRECISION
do 
	for model_path in `find ${models_path}/intel -name $FP`
	do
		mv $model_path/* ${models_path}/$FP
	done
done

rm -r ${models_path}/intel

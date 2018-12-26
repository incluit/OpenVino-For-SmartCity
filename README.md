# OpenVino-For-SmartCity

[![Build Status](https://travis-ci.org/incluit/OpenVino-For-SmartCity.svg?branch=master)](https://travis-ci.org/incluit/OpenVino-For-SmartCity#) [![Sonarcloud Status](https://sonarcloud.io/api/project_badges/measure?project=incluit_OpenVino-For-SmartCity&metric=alert_status)](https://sonarcloud.io/dashboard?id=incluit_OpenVino-For-SmartCity)


This is a follow-up on the OpenVino's inference tutorials:

https://github.com/intel-iot-devkit/inference-tutorials-generic/tree/openvino_toolkit_r4_0

We will work on and extend this tutorial as a demo app for smart cities, specifically for near misses detection.

## Build

1. Clone the repository at desired location:

```Bash
git clone https://github.com/incluit/OpenVino-For-SmartCity.git
```

2. The first step is to configure the build environment for the OpenCV toolkit by sourcing the "setupvars.sh" script.

```Bash
source  /opt/intel/computer_vision_sdk/bin/setupvars.sh
```

3. Change to the top git repository:

```Bash
cd OpenVino-For-SmartCity
```


4. Create a directory to build the tutorial in and change to it.

```Bash
mkdir build
cd build
```

5. Compile:

```Bash
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

## Usage

### Run

1. Before running each of the following sections, be sure to source the helper script.  That will make it easier to use environment variables instead of long names to the models:

```bash
source ../scripts/setupenv.sh 
```


### CPU

1. First, let us see how it works on a single image file using default synchronous mode.

```bash
./intel64/Release/smart_city_tutorial -m $mVDR32 -m_p $person232 -i ../data/car_1.bmp
```


2. You can also run the command in asynchronous mode using the option "-n_async 2":

```bash
./intel64/Release/smart_city_tutorial -m $mVDR32 -m_p $person232 -i ../data/car_1.bmp -n_async 2
```

3. For video files:

```bash
./intel64/Release/smart_city_tutorial -m $mVDR32 -m_p $person232 -i ../data/cars_768x768.h264 -n_async 1
```

### CPU and GPU

**Note**: In order to run this section, the GPU is required to be present and correctly configured.

1. First we run in synchronous mode then asynchronous with increasing -n_async values using the commands:

```Bash
./intel64/Release/smart_city_tutorial -m $mVDR16 -d GPU -m_p $person232 -d_p GPU -i ../data/cars_768x768.h264 -n_async 1
./intel64/Release/smart_city_tutorial -m $mVDR16 -d GPU -m_p $person232 -d_p GPU -i ../data/cars_768x768.h264 -n_async 2
./intel64/Release/smart_city_tutorial -m $mVDR16 -d GPU -m_p $person232 -d_p GPU -i ../data/cars_768x768.h264 -n_async 4
./intel64/Release/smart_city_tutorial -m $mVDR16 -d GPU -m_p $person232 -d_p GPU -i ../data/cars_768x768.h264 -n_async 8
./intel64/Release/smart_city_tutorial -m $mVDR16 -d GPU -m_p $person232 -d_p GPU -i ../data/cars_768x768.h264 -n_async 16
```

2. Asynchronous mode should be faster by some amount for "-n_async 2" then a little more for “-n_async 4” and “-n_async 8”, then not as noticeable for “-n_async 16”.  The improvements come from the CPU running in parallel more and more with the GPU.  The absence of improvement shows when the CPU is doing less in parallel and is waiting on the other devices.  This is referred to as “diminishing returns” and will vary across devices and inference models.

/**
* Copyright (c) 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
**/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char video_message[] = "Optional. Path to an video file. Default value is \"cam\" to work with camera.";

/// @brief message for model argument
static const char vehicle_detection_model_message[] = "Optional. Path to the Vehicle (.xml) file.";
static const char pedestrians_model_message[] = "Optional. Path to the Pedestrians detection model (.xml) file.";
static const char yolo_model_message[] = "Optional. Path to the Yolo detection model (.xml) file.";
static const char vp_model_message[] = "Optional. Path to the Vehicle and Pedestrian detection model (.xml) file.";

/// @brief message for assigning vehicle detection inference to device
static const char target_device_message[] = "Specify the target device for Vehicle Detection (CPU, GPU, FPGA, MYRIAD, or HETERO). ";

/// @brief message for number of simultaneously vehicle detections using dynamic batch
static const char num_batch_message[] = "Specify number of maximum simultaneously processed frames for Vehicle and Pedestrians Detection ( default is 1).";

/// @brief message for assigning vehicle attributes to device
static const char target_device_message_pedestrians[] = "Specify the target device for Pedestrians (CPU, GPU, FPGA, MYRIAD, or HETERO). ";
static const char target_device_message_yolo[] = "Specify the target device for YOLO v3 model (CPU, GPU, FPGA, MYRIAD, or HETERO). ";
static const char target_device_message_vp[] = "Specify the target device for Vehicle and Pedestrian model (CPU, GPU, FPGA, MYRIAD, or HETERO). ";

/// @brief message for number of simultaneously vehicle attributes detections using dynamic batch
static const char num_batch_va_message[] = "Specify number of maximum simultaneously processed vehicles for Vehicle Attributes Detection ( default is 1).";

/// @brief message for enabling dynamic batching for vehicle detections
static const char dyn_va_message[] = "Enable dynamic batching for Vehicle Attributes Detection ( default is 0).";

/// @brief message auto_resize input flag
static const char auto_resize_message[] = "Enable auto-resize (ROI crop & data resize) of input during inference.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance statistics.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "For clDNN (GPU)-targeted custom kernels, if any. Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "For MKLDNN (CPU)-targeted custom layers, if any. Absolute path to a shared library with the kernels impl.";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Probability threshold for vehicle/licence-plate detections.";

/// @brief message raw output flag
static const char raw_output_message[] = "Output Inference results as raw values.";

/// @brief message async function flag
static const char async_depth_message[] = "Maximum number of outstanding async API calls allowed (1=synchronous=default, >1=asynchronous).";

/// @brief message no wait for keypress after input stream completed
static const char no_wait_for_keypress_message[] = "No wait for key press in the end.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "No show processed video.";

static const char show_interest_areas_selection[] = "Draw interest areas locations.";

static const char do_tracking[] = "Track objects.";

static const char do_collision[] = "Detect collisions between objects.";

static const char run_yolo[] = "Running Yolo v3 as detector.";

static const char intersection_over_union_yolo[] = "Intersection over Yolo ROI threshold";

static const char show_graph_message[] = "Running graph server on 127.0.0.1";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// \brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m, "", vehicle_detection_model_message);

/// \brief device the target device for vehicle detection infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief batch size for running vehicle detection <br>
DEFINE_uint32(n, 1, num_batch_message);

/// \brief Define flag for enabling auto-resize of inputs for all models <br>
DEFINE_bool(auto_resize, false, auto_resize_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Flag to disable keypress exit<br>
/// It is an optional parameter
DEFINE_bool(no_wait, false, no_wait_for_keypress_message);

/// \brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief parameter to set depth (number of outstanding requests) of asynchronous API calls <br>
/// It is an optional parameter
DEFINE_uint32(n_async, 1, async_depth_message);

///
DEFINE_bool(show_graph, false, show_graph_message);
DEFINE_bool(show_selection, false, show_interest_areas_selection);
DEFINE_bool(tracking, false, do_tracking);
DEFINE_bool(collision, false, do_collision);
DEFINE_bool(yolo, false, run_yolo);

DEFINE_string(m_p, "", pedestrians_model_message);
DEFINE_uint32(n_p, 1, num_batch_message);
DEFINE_string(d_p, "CPU", target_device_message_pedestrians);

DEFINE_string(m_y, "", yolo_model_message);
DEFINE_uint32(n_y, 1, num_batch_message);
DEFINE_string(d_y, "CPU", target_device_message_yolo);
DEFINE_double(iou_t, 0.4, intersection_over_union_yolo);

DEFINE_string(m_vp, "", vp_model_message);
DEFINE_uint32(n_vp, 1, num_batch_message);
DEFINE_string(d_vp, "CPU", target_device_message_vp);

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl; // NOSONAR
    std::cout << "car_detection_tutorial [OPTION]" << std::endl; // NOSONAR
    std::cout << "Options:" << std::endl; // NOSONAR
    std::cout << std::endl; // NOSONAR
    std::cout << "\t-h\t\t\t\t\t" << help_message << std::endl; // NOSONAR
    std::cout << "\t-i \"<path>\"\t\t\t" << video_message << std::endl; // NOSONAR
    std::cout << "\t-m \"<path>\"\t\t\t" << vehicle_detection_model_message<< std::endl; // NOSONAR
    std::cout << "\t-m_p \"<path>\"\t\t\t" << pedestrians_model_message << std::endl; // NOSONAR
    std::cout << "\t-m_y \"<path>\"\t\t\t" << yolo_model_message << std::endl; // NOSONAR
    std::cout << "\t-m_vp \"<path>\"\t\t\t" << vp_model_message << std::endl; // NOSONAR
    std::cout << "\t\t-l \"<absolute_path>\"\t" << custom_cpu_library_message << std::endl; // NOSONAR
    std::cout << "\t\t\tOr" << std::endl; // NOSONAR
    std::cout << "\t\t-c \"<absolute_path>\"\t" << custom_cldnn_message << std::endl; // NOSONAR
    std::cout << "\t-d \"<device>\"\t\t\t" << target_device_message << std::endl; // NOSONAR
    std::cout << "\t-n \"<num>\"\t\t\t" << num_batch_message << std::endl; // NOSONAR
    std::cout << "\t-d_p \"<device>\"\t\t\t" << target_device_message_pedestrians << std::endl; // NOSONAR
    std::cout << "\t-n_p \"<num>\"\t\t\t" << num_batch_va_message << std::endl; // NOSONAR
    std::cout << "\t-d_y \"<device>\"\t\t\t" << target_device_message_yolo << std::endl; // NOSONAR
    std::cout << "\t-n_y \"<num>\"\t\t\t" << num_batch_va_message << std::endl; // NOSONAR
    std::cout << "\t-d_vp \"<device>\"\t\t\t" << target_device_message_vp << std::endl; // NOSONAR
    std::cout << "\t-n_vp \"<num>\"\t\t\t" << num_batch_va_message << std::endl; // NOSONAR
    std::cout << "\t-dyn_va\t\t\t\t" << dyn_va_message << std::endl; // NOSONAR
    std::cout << "\t-n_aysnc \"<num>\"\t\t\t" << async_depth_message << std::endl; // NOSONAR
    std::cout << "\t-auto_resize\t\t\t\t" << auto_resize_message << std::endl; // NOSONAR
    std::cout << "\t-no_wait\t\t\t\t" << no_wait_for_keypress_message << std::endl; // NOSONAR
    std::cout << "\t-no_show\t\t\t\t" << no_show_processed_video << std::endl; // NOSONAR
    std::cout << "\t-show_selection\t\t\t\t" << show_interest_areas_selection << std::endl; // NOSONAR
    std::cout << "\t-tracking\t\t\t\t" << do_tracking << std::endl; // NOSONAR
    std::cout << "\t-collision\t\t\t\t" << do_collision << std::endl; // NOSONAR
    std::cout << "\t-show_graph\t\t\t\t" << show_graph_message << std::endl; // NOSONAR
    std::cout << "\t-yolo\t\t\t\t" << run_yolo << std::endl; // NOSONAR
    std::cout << "\t-iou_t\t\t\t\t" << intersection_over_union_yolo << std::endl;
    std::cout << "\t-pc\t\t\t\t" << performance_counter_message << std::endl; // NOSONAR
    std::cout << "\t-r\t\t\t\t" << raw_output_message << std::endl; // NOSONAR
    std::cout << "\t-t\t\t\t\t" << thresh_output_message << std::endl; // NOSONAR
}

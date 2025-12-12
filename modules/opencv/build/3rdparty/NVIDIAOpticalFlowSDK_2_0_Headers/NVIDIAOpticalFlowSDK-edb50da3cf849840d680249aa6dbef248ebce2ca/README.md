# NVIDIAOpticalFlowSDK
NVIDIA GPUs starting with the Turing generation, contain a hardware accelerator(referred to as NVOFA) for computing 
optical flow and stereo disparity between frames. This works independently of graphics/CUDA cores. With end-to-end 
optical flow calculation offloaded to NVOFA, the graphics/CUDA cores and the CPU are free for other operations. 
Optical flow vectors are useful in various use-cases such as object detection and tracking, video frame rate 
up-conversion, depth estimation, stitching etc. Using flow vectors for object detection has been observed to 
improve inference accuracy.

The users must have the following hardware and software for being able to use the functionality:

* NVIDIA Turing GPUs - Refer to the NVIDIA Optical flow developer zone web page
 (https://developer.nvidia.com/opticalflow-sdk) for GPUs which support Optical
 flow and stereo disparity hardware acceleration.
* Windows: Display driver version 445.87 or higher
* Linux:   Display driver version 450.51 or higher

The latest shipping NVIDIA Optical Flow SDK can be downloaded from https://developer.nvidia.com/opticalflow-sdk. 
The NVIDIA Optical Flow SDK provides detailed documentation, sample application illustarting the usage of 
NVIDIA Optical Flow APIs.
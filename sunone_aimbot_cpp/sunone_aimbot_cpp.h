#ifndef SUNONE_AIMBOT_CPP_H
#define SUNONE_AIMBOT_CPP_H

#include "config.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif
#include "dml_detector.h"
#include "mouse.h"
#include "SerialConnection.h"
#include "Kmbox_b.h"
#include "detection_buffer.h"

extern Config config;
#ifdef USE_CUDA
extern TrtDetector trt_detector;
#endif
extern DirectMLDetector* dml_detector;
extern DetectionBuffer detectionBuffer;
extern MouseThread* globalMouseThread;
extern SerialConnection* arduinoSerial;
extern Kmbox_b_Connection* kmboxSerial;
extern std::atomic<bool> input_method_changed;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;

#endif // SUNONE_AIMBOT_CPP_H
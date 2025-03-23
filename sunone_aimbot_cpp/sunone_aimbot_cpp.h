#ifndef SUNONE_AIMBOT_CPP_H
#define SUNONE_AIMBOT_CPP_H

#include "config.h"
#include "detector.h"
#include "directml_detector.h"
#include "mouse.h"
#include "SerialConnection.h"
#include "KmboxConnection.h"
#include "optical_flow.h"

extern Config config;
extern Detector detector;
extern DirectMLDetector* dml_detector;
extern MouseThread* globalMouseThread;
extern SerialConnection* arduinoSerial;
extern KmboxConnection* kmboxSerial;
extern std::atomic<bool> input_method_changed;
extern OpticalFlow opticalFlow;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;

#endif // SUNONE_AIMBOT_CPP_H
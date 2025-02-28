#ifndef SUNONE_AIMBOT_CPP_H
#define SUNONE_AIMBOT_CPP_H

#include "config.h"
#include "detector.h"
#include "mouse.h"
#include "SerialConnection.h"
#include "optical_flow.h"

extern Config config;
extern Detector detector;
extern MouseThread* globalMouseThread;
extern SerialConnection* arduinoSerial;
extern std::atomic<bool> input_method_changed;
extern OpticalFlow opticalFlow;

#endif // SUNONE_AIMBOT_CPP_H
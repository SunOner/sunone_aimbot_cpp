#ifndef SUNONE_AIMBOT_CPP_H
#define SUNONE_AIMBOT_CPP_H

#include "config.h"
#include "detector.h"
#include "mouse.h"
#include "SerialConnection.h"

extern Config config;
extern Detector detector;
extern MouseThread* globalMouseThread;
extern SerialConnection* serial;

#endif // SUNONE_AIMBOT_CPP_H
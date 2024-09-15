#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include "capture.h"
#include "visuals.h"
#include "detector.h"
#include "mouse.h"
#include "target.h"
#include "sunone_aimbot_cpp.h"
#include "detector.h"
#include <Windows.h>

#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib")

using namespace cv;
using namespace std;

Mat latestFrame;
std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);

Detector detector;

int detection_window_width;
int detection_window_height;
double dpi;
double sensitivity;
double fovX;
double fovY;
double minSpeedMultiplier;
double maxSpeedMultiplier;
double predictionInterval;

MouseThread* mouseThread;

// Функция для чтения параметров из файла конфигурации
void loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл конфигурации: " << filename << std::endl;
        exit(EXIT_FAILURE); // Выход, если файл не найден
    }

    std::map<std::string, std::string> configMap;
    std::string line;

    // Читаем файл построчно
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string key, value;

        // Игнорируем комментарии и пустые строки
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        // Разделяем строку на ключ и значение
        if (std::getline(lineStream, key, '=') && std::getline(lineStream, value)) {
            key = key.substr(key.find_first_not_of(" \t")); // убираем пробелы
            value = value.substr(value.find_first_not_of(" \t")); // убираем пробелы
            configMap[key] = value;
        }
    }

    // Чтение значений из конфигурации, выход при отсутствии обязательных параметров
    try {
        detection_window_width = std::stoi(configMap.at("detection_window_width"));
        detection_window_height = std::stoi(configMap.at("detection_window_height"));
        dpi = std::stod(configMap.at("dpi"));
        sensitivity = std::stod(configMap.at("sensitivity"));
        fovX = std::stod(configMap.at("fovX"));
        fovY = std::stod(configMap.at("fovY"));
        minSpeedMultiplier = std::stod(configMap.at("minSpeedMultiplier"));
        maxSpeedMultiplier = std::stod(configMap.at("maxSpeedMultiplier"));
        predictionInterval = std::stod(configMap.at("predictionInterval"));
    } catch (const std::out_of_range& e) {
        std::cerr << "Отсутствует необходимый параметр в конфигурационном файле!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void mouseThreadFunction() {
    int lastDetectionVersion = -1;

    while (!shouldExit) {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        {
            std::unique_lock<std::mutex> lock(detector.detectionMutex);
            detector.detectionCV.wait(lock, [&]() { return detector.detectionVersion > lastDetectionVersion || shouldExit; });
            if (shouldExit) break;

            lastDetectionVersion = detector.detectionVersion;

            boxes = detector.detectedBoxes;
            classes = detector.detectedClasses;
        }

        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) {
            Target* target = sortTargets(boxes, classes, detection_window_width, detection_window_height, false);
            if (target) {
                mouseThread->moveMouseToTarget(*target);
                delete target;
            }
        }
    }
}

int main() {
    // Загружаем конфигурацию из файла config.ini
    loadConfig("config.ini");

    // Инициализируем MouseThread только после загрузки конфигурации
    mouseThread = new MouseThread(detection_window_width, detection_window_height, dpi, sensitivity, fovX, fovY, minSpeedMultiplier, maxSpeedMultiplier, predictionInterval);

    detector.initialize("models/sunxds_0.6.3.engine");

    std::thread capThread(captureThread, detection_window_width, detection_window_height);
    std::thread detThread(&Detector::inferenceThread, &detector);
    std::thread dispThread(displayThread);
    std::thread mouseMovThread(mouseThreadFunction);

    capThread.join();
    detThread.join();
    dispThread.join();
    mouseMovThread.join();

    delete mouseThread; // Освобождаем память
    return 0;
}

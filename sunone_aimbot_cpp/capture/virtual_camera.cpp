#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>
#include <string>

#include "virtual_camera.h"
#include "config.h" // Asegurarse de que config esté disponible

// --- Código de manejo de caché (sin cambios, la lógica es sólida) ---
namespace {
    inline int even(int v) { return (v % 2 == 0) ? v : v + 1; }

    inline std::filesystem::path exe_dir()
    {
        char path[MAX_PATH]{};
        GetModuleFileNameA(nullptr, path, MAX_PATH);
        return std::filesystem::path(path).parent_path();
    }

    const std::filesystem::path kCamCachePath = exe_dir() / "virtual_cameras_cache.txt";

    void EnsureCacheDir()
    {
        std::error_code ec;
        std::filesystem::create_directories(kCamCachePath.parent_path(), ec);
    }

    std::vector<std::string> LoadCamList()
    {
        EnsureCacheDir();
        std::vector<std::string> cams;
        std::ifstream ifs(kCamCachePath);
        std::string line;
        while (std::getline(ifs, line))
        {
            while (!line.empty() && isspace(static_cast<unsigned char>(line.back())))
                line.pop_back();
            while (!line.empty() && isspace(static_cast<unsigned char>(line.front())))
                line.erase(line.begin());
            if (!line.empty())
                cams.emplace_back(std::move(line));
        }
        return cams;
    }

    void SaveCamList(const std::vector<std::string>& cams)
    {
        EnsureCacheDir();
        std::ofstream ofs(kCamCachePath, std::ios::trunc);
        for (const auto& c : cams) ofs << c << '\n';
    }

    static std::vector<std::string>& CamCache()
    {
        static std::vector<std::string> cache = LoadCamList();
        return cache;
    }
}

VirtualCameraCapture::VirtualCameraCapture(int w, int h)
{
    const auto& cams = VirtualCameraCapture::GetAvailableVirtualCameras();
    auto it = std::find(cams.begin(), cams.end(), config.virtual_camera_name);
    int camIdx = 0;

    if (it != cams.end()) {
        camIdx = static_cast<int>(std::distance(cams.begin(), it));
    } else {
        std::cerr << "[VirtualCamera] Camera name not found in cache: " << config.virtual_camera_name << std::endl;
        if (!cams.empty()) {
            camIdx = 0;
            config.virtual_camera_name = cams[0]; // Asigna directamente
            config.saveConfig("config.ini");
            std::cout << "[VirtualCamera] Defaulting to first camera: " << config.virtual_camera_name << std::endl;
        } else {
            throw std::runtime_error("[VirtualCamera] No virtual cameras found or available.");
        }
    }
    
    cap_ = std::make_unique<cv::VideoCapture>(camIdx, cv::CAP_MSMF);
    std::cout << "[VirtualCamera] Opening camera: " << config.virtual_camera_name << " (Index: " << camIdx << ")" << std::endl;

    if (!cap_->isOpened())
        throw std::runtime_error("[VirtualCamera] Unable to open the specified capture device.");

    // Configurar propiedades de la cámara para baja latencia
    cap_->set(cv::CAP_PROP_FRAME_WIDTH, even(w));
    cap_->set(cv::CAP_PROP_FRAME_HEIGHT, even(h));
    if (config.capture_fps > 0)
        cap_->set(cv::CAP_PROP_FPS, config.capture_fps);
    
    cap_->set(cv::CAP_PROP_BUFFERSIZE, 0); // Crucial para baja latencia

    // Leer las dimensiones reales después de configurarlas
    roiW_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
    roiH_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));

    if (config.verbose)
        std::cout << "[VirtualCamera] Actual capture resolution: " << roiW_ << 'x' << roiH_ 
                  << " @ " << cap_->get(cv::CAP_PROP_FPS) << " FPS\n";
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    if (cap_ && cap_->isOpened()) {
        cap_->release();
    }
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!cap_ || !cap_->isOpened()) return {};

    // Leer directamente en la variable miembro para evitar una copia/movimiento extra
    if (!cap_->read(frameCpu) || frameCpu.empty()) {
        return {};
    }

    // Asegurar que el formato sea BGR de 3 canales
    if (frameCpu.channels() == 4) {
        cv::cvtColor(frameCpu, frameCpu, cv::COLOR_BGRA2BGR);
    } else if (frameCpu.channels() == 1) {
        cv::cvtColor(frameCpu, frameCpu, cv::COLOR_GRAY2BGR);
    }

    // Redimensionar si es necesario
    const int target_width = config.virtual_camera_width;
    const int target_height = config.virtual_camera_heigth;
    if (target_width > 0 && target_height > 0 && (frameCpu.cols != target_width || frameCpu.rows != target_height)) {
        cv::resize(frameCpu, frameCpu, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
    }
    
    // *** OPTIMIZACIÓN CLAVE (YA PRESENTE): Mover los datos del frame sin copia profunda. ***
    // La variable miembro frameCpu queda en un estado válido pero vacío, lista para el siguiente 'read'.
    return std::move(frameCpu);
}

// --- Métodos de utilidad (sin cambios) ---
std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras(bool forceRescan)
{
    auto& cache = CamCache();
    if (!forceRescan && !cache.empty()) return cache;

    cache.clear();
    // Escanear los primeros 10 dispositivos es una heurística razonable
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture test(i, cv::CAP_MSMF);
        if (test.isOpened()) {
            // En un futuro, se podrían obtener nombres más descriptivos usando Media Foundation
            cache.push_back("Camera " + std::to_string(i));
        }
    }
    SaveCamList(cache);
    return cache;
}

void VirtualCameraCapture::ClearCachedCameraList()
{
    CamCache().clear();
    std::error_code ec;
    std::filesystem::remove(kCamCachePath, ec);
} // virtual_camera.cpp
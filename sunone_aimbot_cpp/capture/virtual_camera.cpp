#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility> // Necesario para std::move

#include "virtual_camera.h"

namespace {

    inline int even(int v) { return (v % 2 == 0) ? v : v + 1; }

    inline std::filesystem::path exe_dir()
    {
        char path[MAX_PATH]{};
        GetModuleFileNameA(nullptr, path, MAX_PATH);
        return std::filesystem::path(path).parent_path();
    }

    const std::filesystem::path kCamCachePath =
        exe_dir() / "virtual_cameras_cache.txt";

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
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t'))
                line.pop_back();

            while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
                line.erase(line.begin());

            if (!line.empty())
            {
                cams.emplace_back(std::move(line));
            }
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
        static std::vector<std::string> cache = [] {
            auto list = LoadCamList();
            return list;
            }();
        return cache;
    }
}

VirtualCameraCapture::VirtualCameraCapture(int w, int h)
{
    int camIdx = 0;
    const auto& cams = VirtualCameraCapture::GetAvailableVirtualCameras();
    auto it = std::find(cams.begin(), cams.end(), config.virtual_camera_name);
    if (it != cams.end())
    {
        camIdx = static_cast<int>(std::distance(cams.begin(), it));
    }
    else
    {
        std::cerr << "[VirtualCamera] Camera name not found in cache: " << config.virtual_camera_name << std::endl;
        if (!cams.empty())
        {
            camIdx = 0;
            config.virtual_camera_name = cams[0];
            config.saveConfig("config.ini");
        }
    }
    
    cap_ = std::make_unique<cv::VideoCapture>(camIdx, cv::CAP_MSMF);
	std::cout << "[VirtualCamera] Opening camera: " << config.virtual_camera_name << std::endl;
    // cap_->set(cv::CAP_PROP_FOURCC, 0);

    if (!cap_->isOpened())
        throw std::runtime_error("[VirtualCamera] Unable to open any capture device");

    bool autoMode = (w <= 0 || h <= 0);

    if (autoMode)
    {
        w = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
        h = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    }
    else
    {
        cap_->set(cv::CAP_PROP_FRAME_WIDTH, even(w));
        cap_->set(cv::CAP_PROP_FRAME_HEIGHT, even(h));
		w = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
		h = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    if (config.capture_fps > 0)
        cap_->set(cv::CAP_PROP_FPS, config.capture_fps);
    
    cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);

    roiW_ = even(w);
    roiH_ = even(h);

    if (config.verbose)
        std::cout << "[VirtualCamera] Actual capture: "
        << roiW_ << 'x' << roiH_ << " @ "
            << cap_->get(cv::CAP_PROP_FPS) << " FPS\n";
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    if (cap_)
    {
        if (cap_->isOpened())
        {
            cap_->release();
        }
        cap_.reset();
    }
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!cap_ || !cap_->isOpened())
        return cv::Mat();

    // La variable 'frame' es local, no hay necesidad de usar la variable miembro 'frameCpu' aquí
    cv::Mat frame;
    if (!cap_->read(frame) || frame.empty())
    {
        return cv::Mat();
    }

    switch (frame.channels())
    {
    case 1: cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR); break;
    case 4: cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR); break;
    case 3:                                                 break;
    default:
        std::cerr << "[VirtualCamera] Unexpected channel count: "
            << frame.channels() << std::endl;
        return cv::Mat();
    }

    int target_width = config.virtual_camera_width;
    int target_height = config.virtual_camera_heigth;

    // Asignar el frame procesado a la variable miembro 'frameCpu'
    frameCpu = frame;

    if (target_width > 0 && target_height > 0 && !frameCpu.empty())
    {
        cv::resize(frameCpu, frameCpu, cv::Size(target_width, target_height));
    }

    // *** OPTIMIZACIÓN ***
    // Se utiliza std::move para transferir la propiedad de los datos del frame sin
    // realizar una copia profunda (clone). Esto es más eficiente.
    // La variable miembro frameCpu quedará en un estado válido pero vacío,
    // y será rellenada en la siguiente llamada a GetNextFrameCpu.
    return std::move(frameCpu);
}

std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras(bool forceRescan)
{
    auto& cache = CamCache();

    if (!forceRescan)
    {
        auto diskList = LoadCamList();
        if (!diskList.empty())
        {
            cache = std::move(diskList);
            return cache;
        }
    }

    if (!forceRescan && !cache.empty())
        return cache;

    cache.clear();
    for (int i = 0; i < 10; ++i)
    {
        cv::VideoCapture test(i, cv::CAP_MSMF);
        if (test.isOpened())
            cache.emplace_back("Camera " + std::to_string(i));
    }
    SaveCamList(cache);
    return cache;
}

void VirtualCameraCapture::ClearCachedCameraList()
{
    CamCache().clear();
    std::error_code ec;
    std::filesystem::remove(kCamCachePath, ec);
}
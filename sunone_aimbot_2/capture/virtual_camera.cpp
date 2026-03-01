#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <dshow.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "virtual_camera.h"

#pragma comment(lib, "strmiids.lib")

namespace
{
inline int even(int v) { return (v % 2 == 0) ? v : v + 1; }

struct CameraCandidate
{
    std::string displayName;
    int index = -1;
    int backend = cv::CAP_ANY;
};

std::mutex& CameraCacheMutex()
{
    static std::mutex m;
    return m;
}

std::vector<CameraCandidate>& CameraCandidateCache()
{
    static std::vector<CameraCandidate> cache;
    return cache;
}

std::vector<std::string>& CameraNameCache()
{
    static std::vector<std::string> cache;
    return cache;
}

std::string TrimCopy(const std::string& s)
{
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;
    return s.substr(b, e - b);
}

std::string ToLowerCopy(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

bool ContainsCaseInsensitive(const std::string& haystack, const std::string& needle)
{
    const std::string h = ToLowerCopy(haystack);
    const std::string n = ToLowerCopy(needle);
    return h.find(n) != std::string::npos;
}

std::string WideToUtf8(const std::wstring& w)
{
    if (w.empty())
        return {};

    const int required = WideCharToMultiByte(
        CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr
    );
    if (required <= 0)
        return {};

    std::string out(static_cast<size_t>(required), '\0');
    const int converted = WideCharToMultiByte(
        CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), out.data(), required, nullptr, nullptr
    );
    if (converted <= 0)
        return {};

    return out;
}

std::string BackendToString(int backend)
{
    if (backend == cv::CAP_DSHOW)
        return "DSHOW";
    if (backend == cv::CAP_MSMF)
        return "MSMF";
    return "ANY";
}

std::vector<CameraCandidate> EnumerateDirectShowCandidates()
{
    std::vector<CameraCandidate> out;

    HRESULT hrCo = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool shouldUninit = SUCCEEDED(hrCo);
    if (FAILED(hrCo) && hrCo != RPC_E_CHANGED_MODE)
        return out;

    ICreateDevEnum* devEnum = nullptr;
    IEnumMoniker* enumMoniker = nullptr;

    HRESULT hr = CoCreateInstance(
        CLSID_SystemDeviceEnum,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_ICreateDevEnum,
        reinterpret_cast<void**>(&devEnum)
    );

    if (SUCCEEDED(hr) && devEnum)
        hr = devEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMoniker, 0);

    if (hr == S_OK && enumMoniker)
    {
        IMoniker* moniker = nullptr;
        ULONG fetched = 0;
        int index = 0;

        while (enumMoniker->Next(1, &moniker, &fetched) == S_OK)
        {
            std::string name = "Camera " + std::to_string(index);

            IPropertyBag* propBag = nullptr;
            if (SUCCEEDED(moniker->BindToStorage(0, 0, IID_IPropertyBag, reinterpret_cast<void**>(&propBag))) && propBag)
            {
                VARIANT varName;
                VariantInit(&varName);

                bool gotName = false;
                if (SUCCEEDED(propBag->Read(L"FriendlyName", &varName, 0)) && varName.vt == VT_BSTR && varName.bstrVal)
                {
                    name = WideToUtf8(varName.bstrVal);
                    gotName = !name.empty();
                }
                VariantClear(&varName);

                if (!gotName)
                {
                    VARIANT varDesc;
                    VariantInit(&varDesc);
                    if (SUCCEEDED(propBag->Read(L"Description", &varDesc, 0)) && varDesc.vt == VT_BSTR && varDesc.bstrVal)
                    {
                        std::string desc = WideToUtf8(varDesc.bstrVal);
                        if (!desc.empty())
                            name = std::move(desc);
                    }
                    VariantClear(&varDesc);
                }

                propBag->Release();
            }

            moniker->Release();
            moniker = nullptr;

            name = TrimCopy(name);
            if (name.empty())
                name = "Camera " + std::to_string(index);

            CameraCandidate c;
            c.displayName = std::move(name);
            c.index = index;
            c.backend = cv::CAP_DSHOW;
            out.push_back(std::move(c));
            ++index;
        }
    }

    if (enumMoniker)
        enumMoniker->Release();
    if (devEnum)
        devEnum->Release();

    if (shouldUninit)
        CoUninitialize();

    return out;
}

bool CandidateAlreadyExists(
    const std::vector<CameraCandidate>& list,
    int index,
    int backend)
{
    return std::any_of(list.begin(), list.end(), [&](const CameraCandidate& c) {
        return c.index == index && c.backend == backend;
    });
}

void AppendScannedBackendCandidates(
    std::vector<CameraCandidate>& out,
    int backend,
    int maxIndex)
{
    for (int i = 0; i < maxIndex; ++i)
    {
        if (CandidateAlreadyExists(out, i, backend))
            continue;

        cv::VideoCapture test(i, backend);
        if (!test.isOpened())
            continue;

        CameraCandidate c;
        c.displayName = BackendToString(backend) + " Camera " + std::to_string(i);
        c.index = i;
        c.backend = backend;
        out.push_back(std::move(c));
        test.release();
    }
}

void EnsureDisplayNamesAreUnique(std::vector<CameraCandidate>& list)
{
    std::unordered_map<std::string, int> seen;
    for (auto& c : list)
    {
        if (c.displayName.empty())
            c.displayName = "Camera " + std::to_string(c.index);

        const std::string key = ToLowerCopy(c.displayName);
        int& count = seen[key];
        if (count > 0)
        {
            c.displayName += " [" + BackendToString(c.backend) + " #" + std::to_string(c.index) + "]";
        }
        ++count;
    }
}

std::vector<CameraCandidate> EnumerateCameraCandidates()
{
    std::vector<CameraCandidate> out = EnumerateDirectShowCandidates();

    // Fallback entries from OpenCV probing. Useful when DSHOW listing fails
    // or when only MSMF backend can open the camera.
    AppendScannedBackendCandidates(out, cv::CAP_MSMF, 20);
    if (out.empty())
        AppendScannedBackendCandidates(out, cv::CAP_DSHOW, 20);

    EnsureDisplayNamesAreUnique(out);
    return out;
}

std::vector<CameraCandidate> GetCameraCandidates(bool forceRescan)
{
    std::lock_guard<std::mutex> lock(CameraCacheMutex());

    auto& candidateCache = CameraCandidateCache();
    auto& nameCache = CameraNameCache();
    if (forceRescan || candidateCache.empty())
    {
        candidateCache = EnumerateCameraCandidates();
        nameCache.clear();
        nameCache.reserve(candidateCache.size());
        for (const auto& c : candidateCache)
            nameCache.push_back(c.displayName);
    }

    return candidateCache;
}

int ScoreAutoCameraChoice(const CameraCandidate& c)
{
    const std::string n = ToLowerCopy(c.displayName);
    int score = 0;

    if (n.find("obs virtual camera") != std::string::npos)
        score += 300;
    else if (n.find("obs-camera") != std::string::npos)
        score += 260;
    else if (n.find("obs") != std::string::npos)
        score += 180;

    if (n.find("virtual") != std::string::npos)
        score += 80;

    if (c.backend == cv::CAP_DSHOW)
        score += 20;

    return score;
}

int FindExactNameMatch(const std::vector<CameraCandidate>& candidates, const std::string& requestedName)
{
    const std::string key = ToLowerCopy(TrimCopy(requestedName));
    if (key.empty())
        return -1;

    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (ToLowerCopy(candidates[i].displayName) == key)
            return static_cast<int>(i);
    }
    return -1;
}

int FindPartialNameMatch(const std::vector<CameraCandidate>& candidates, const std::string& requestedName)
{
    const std::string key = ToLowerCopy(TrimCopy(requestedName));
    if (key.empty())
        return -1;

    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (ContainsCaseInsensitive(candidates[i].displayName, key))
            return static_cast<int>(i);
    }
    return -1;
}

std::vector<int> BuildOpenOrder(const std::vector<CameraCandidate>& candidates, const std::string& requestedName)
{
    std::vector<int> order;
    order.reserve(candidates.size());

    auto addUnique = [&](int idx)
    {
        if (idx < 0 || idx >= static_cast<int>(candidates.size()))
            return;
        if (std::find(order.begin(), order.end(), idx) == order.end())
            order.push_back(idx);
    };

    const std::string trimmedRequest = TrimCopy(requestedName);
    const std::string lowerRequest = ToLowerCopy(trimmedRequest);
    const bool autoSelect = trimmedRequest.empty() || lowerRequest == "none" || lowerRequest == "auto";

    if (!autoSelect)
    {
        int exact = FindExactNameMatch(candidates, trimmedRequest);
        if (exact >= 0)
            addUnique(exact);
        else
        {
            int partial = FindPartialNameMatch(candidates, trimmedRequest);
            if (partial >= 0)
                addUnique(partial);
        }
    }

    int bestAutoIdx = -1;
    int bestScore = -1;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
    {
        const int score = ScoreAutoCameraChoice(candidates[i]);
        if (score > bestScore)
        {
            bestScore = score;
            bestAutoIdx = i;
        }
    }
    addUnique(bestAutoIdx);

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
        addUnique(i);

    return order;
}

bool TryOpenCandidate(
    const CameraCandidate& candidate,
    cv::VideoCapture& cap,
    bool verbose)
{
    cap.release();
    cap.open(candidate.index, candidate.backend);
    if (!cap.isOpened())
        return false;

    if (verbose)
    {
        std::cout << "[VirtualCamera] Opened '" << candidate.displayName
                  << "' using " << BackendToString(candidate.backend)
                  << " index=" << candidate.index << std::endl;
    }
    return true;
}
} // namespace

VirtualCameraCapture::VirtualCameraCapture(
    int w,
    int h,
    const std::string& cameraName,
    int captureFps,
    bool verbose)
    : targetWidth_(w)
    , targetHeight_(h)
    , selectedCameraName_(cameraName)
    , captureFps_(captureFps)
    , verbose_(verbose)
{
    auto candidates = GetCameraCandidates(false);
    if (candidates.empty())
        candidates = GetCameraCandidates(true);

    if (candidates.empty())
    {
        throw std::runtime_error("[VirtualCamera] No camera devices found");
    }

    const std::string requestedName = selectedCameraName_;
    if (!requestedName.empty() && ToLowerCopy(TrimCopy(requestedName)) != "none")
    {
        int exact = FindExactNameMatch(candidates, requestedName);
        int partial = (exact >= 0) ? exact : FindPartialNameMatch(candidates, requestedName);
        if (exact < 0 && partial < 0)
        {
            std::cerr << "[VirtualCamera] Requested camera not found: " << requestedName
                      << ". Will use fallback search." << std::endl;
        }
    }

    cap_ = std::make_unique<cv::VideoCapture>();
    bool opened = false;

    auto openByOrder = [&](const std::vector<CameraCandidate>& list) -> bool
    {
        const auto order = BuildOpenOrder(list, requestedName);
        for (int idx : order)
        {
            if (idx < 0 || idx >= static_cast<int>(list.size()))
                continue;

            if (TryOpenCandidate(list[idx], *cap_, verbose_))
            {
                selectedCameraName_ = list[idx].displayName;
                return true;
            }
        }
        return false;
    };

    opened = openByOrder(candidates);
    if (!opened)
    {
        candidates = GetCameraCandidates(true);
        opened = openByOrder(candidates);
    }

    if (!opened || !cap_ || !cap_->isOpened())
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

    if (captureFps_ > 0)
        cap_->set(cv::CAP_PROP_FPS, captureFps_);

    cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);

    roiW_ = even(std::max(2, w));
    roiH_ = even(std::max(2, h));

    if (verbose_)
    {
        std::cout << "[VirtualCamera] Selected camera: " << selectedCameraName_ << std::endl;
        std::cout << "[VirtualCamera] Actual capture: "
                  << roiW_ << 'x' << roiH_ << " @ "
                  << cap_->get(cv::CAP_PROP_FPS) << " FPS" << std::endl;
    }
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    if (cap_)
    {
        if (cap_->isOpened())
            cap_->release();
        cap_.reset();
    }
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!cap_ || !cap_->isOpened())
        return cv::Mat();

    cv::Mat frame;
    if (!cap_->read(frame) || frame.empty())
        return cv::Mat();

    switch (frame.channels())
    {
    case 1:
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        break;
    case 4:
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        break;
    case 3:
        break;
    default:
        std::cerr << "[VirtualCamera] Unexpected channel count: " << frame.channels() << std::endl;
        return cv::Mat();
    }

    frameCpu = frame;

    if (targetWidth_ > 0 && targetHeight_ > 0 && !frameCpu.empty())
        cv::resize(frameCpu, frameCpu, cv::Size(targetWidth_, targetHeight_));

    return frameCpu.clone();
}

std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras(bool forceRescan)
{
    auto candidates = GetCameraCandidates(forceRescan);
    std::vector<std::string> names;
    names.reserve(candidates.size());
    for (const auto& c : candidates)
        names.push_back(c.displayName);
    return names;
}

void VirtualCameraCapture::ClearCachedCameraList()
{
    std::lock_guard<std::mutex> lock(CameraCacheMutex());
    CameraCandidateCache().clear();
    CameraNameCache().clear();
}

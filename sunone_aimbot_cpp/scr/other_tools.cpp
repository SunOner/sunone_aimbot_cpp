#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <string>
#include <iostream>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <array>
#include <fstream>
#include <cstdlib>
#include <random>
#include <set>
#include <unordered_set>
#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <comdef.h>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <VersionHelpers.h>

#include "other_tools.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

#pragma comment(lib, "windowscodecs.lib")

using Microsoft::WRL::ComPtr;

static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

static inline bool is_base64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string WideToUtf8(const std::wstring& ws)
{
    if (ws.empty())
        return {};

    int len = WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()),
        nullptr, 0, nullptr, nullptr);
    if (len <= 0)
        return {};

    std::string out(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()),
        out.data(), len, nullptr, nullptr);
    return out;
}

static std::string hr_to_string(HRESULT hr)
{
    _com_error err(hr);
    const wchar_t* ws = err.ErrorMessage();
    int len = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
    std::string s(len > 0 ? len - 1 : 0, '\0');
    if (len > 0)
        WideCharToMultiByte(CP_UTF8, 0, ws, -1, s.data(), len, nullptr, nullptr);
    return s;
}

bool IsValidImageFile(const std::wstring& wpath, UINT& outW, UINT& outH, std::string& outErr)
{
    outW = outH = 0;
    outErr.clear();

    static std::once_flag coinit_flag;
    static HRESULT coinit_hr = S_OK;
    std::call_once(coinit_flag, [] {
        coinit_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        });

    ComPtr<IWICImagingFactory> factory;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { outErr = "WIC factory error: " + hr_to_string(hr); return false; }

    ComPtr<IWICBitmapDecoder> decoder;
    hr = factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) { outErr = "DecoderFromFilename failed: " + hr_to_string(hr); return false; }

    ComPtr<IWICBitmapFrameDecode> frame;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) { outErr = "GetFrame(0) failed: " + hr_to_string(hr); return false; }

    UINT w = 0, h = 0;
    hr = frame->GetSize(&w, &h);
    if (FAILED(hr)) { outErr = "GetSize failed: " + hr_to_string(hr); return false; }

    const UINT MAX_DIM = 16384;
    if (w == 0 || h == 0 || w > MAX_DIM || h > MAX_DIM)
    {
        outErr = "Invalid image size: " + std::to_string(w) + "x" + std::to_string(h);
        return false;
    }

    ComPtr<IWICFormatConverter> conv;
    hr = factory->CreateFormatConverter(&conv);
    if (FAILED(hr)) { outErr = "CreateFormatConverter failed: " + hr_to_string(hr); return false; }

    hr = conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone, nullptr, 0.0f, WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) { outErr = "Converter Initialize failed: " + hr_to_string(hr); return false; }

    const UINT probe_rows = (std::min)(h, 8u);
    std::vector<uint8_t> probe;
    probe.resize((size_t)w * probe_rows * 4);
    WICRect rect{ 0, 0, (INT)w, (INT)probe_rows };
    hr = conv->CopyPixels(&rect, (UINT)(w * 4), (UINT)probe.size(), probe.data());
    if (FAILED(hr)) { outErr = "CopyPixels failed: " + hr_to_string(hr); return false; }

    outW = w; outH = h;
    return true;
}

std::vector<unsigned char> Base64Decode(const std::string& encoded_string)
{
    int in_len = static_cast<int>(encoded_string.size());
    int i = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<unsigned char> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(static_cast<unsigned char>(encoded_string[in_])))
    {
        char_array_4[i++] = static_cast<unsigned char>(encoded_string[in_]); in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
                char_array_4[i] = static_cast<unsigned char>(base64_chars.find(char_array_4[i]));

            char_array_3[0] = static_cast<unsigned char>((char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4));
            char_array_3[1] = static_cast<unsigned char>(((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2));
            char_array_3[2] = static_cast<unsigned char>(((char_array_4[2] & 0x3) << 6) + char_array_4[3]);

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i)
    {
        int j;
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = static_cast<unsigned char>(base64_chars.find(char_array_4[j]));

        char_array_3[0] = static_cast<unsigned char>((char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4));
        char_array_3[1] = static_cast<unsigned char>(((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2));
        char_array_3[2] = static_cast<unsigned char>(((char_array_4[2] & 0x3) << 6) + char_array_4[3]);

        for (j = 0; (j < i - 1); j++)
            ret.push_back(char_array_3[j]);
    }

    return ret;
}

bool fileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string replace_extension(const std::string& filename, const std::string& new_extension)
{
    size_t last_dot = filename.find_last_of(".");
    if (last_dot == std::string::npos)
    {
        return filename + new_extension;
    }
    else
    {
        return filename.substr(0, last_dot) + new_extension;
    }
}

std::string intToString(int value)
{
    return std::to_string(value);
}

std::vector<std::string> getModelFiles()
{
    std::vector<std::string> modelsFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".engine" ||
            entry.is_regular_file() && entry.path().extension() == ".onnx")
        {
            modelsFiles.push_back(entry.path().filename().string());
        }
    }
    return modelsFiles;
}

std::vector<std::string> getEngineFiles()
{
    std::vector<std::string> engineFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".engine" ||
            entry.is_regular_file() && entry.path().extension() == ".onnx")
        {
            engineFiles.push_back(entry.path().filename().string());
        }
    }
    return engineFiles;
}

std::vector<std::string> getOnnxFiles()
{
    std::vector<std::string> onnxFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx")
        {
            onnxFiles.push_back(entry.path().filename().string());
        }
    }
    return onnxFiles;
}

std::vector<std::string>::difference_type getModelIndex(std::vector<std::string> engine_models)
{
    auto it = std::find(engine_models.begin(), engine_models.end(), config.ai_model);

    if (it != engine_models.end())
    {
        return std::distance(engine_models.begin(), it);
    }
    else
    {
        return 0; // not found
    }
}

bool LoadTextureFromFile(const char* filename, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height)
{
    int image_width = 0;
    int image_height = 0;
    int channels = 0;
    unsigned char* image_data = stbi_load(filename, &image_width, &image_height, &channels, 4);
    if (image_data == NULL)
        return false;

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = image_width;
    desc.Height = image_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    ID3D11Texture2D* pTexture = NULL;
    D3D11_SUBRESOURCE_DATA subResource;
    subResource.pSysMem = image_data;
    subResource.SysMemPitch = desc.Width * 4;
    subResource.SysMemSlicePitch = 0;
    device->CreateTexture2D(&desc, &subResource, &pTexture);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    device->CreateShaderResourceView(pTexture, &srvDesc, out_srv);
    pTexture->Release();

    *out_width = image_width;
    *out_height = image_height;
    stbi_image_free(image_data);

    return true;
}

bool LoadTextureFromMemory(const std::string& imageBase64, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height)
{
    std::vector<unsigned char> decodedData = Base64Decode(imageBase64);

    int image_width = 0;
    int image_height = 0;
    int channels = 0;

    unsigned char* image_data = stbi_load_from_memory(
        decodedData.data(),
        static_cast<int>(decodedData.size()),
        &image_width, &image_height, &channels, 4);

    if (image_data == NULL)
    {
        std::cerr << "Can't load image trom memory." << std::endl;
        return false;
    }

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = image_width;
    desc.Height = image_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    ID3D11Texture2D* pTexture = NULL;
    D3D11_SUBRESOURCE_DATA subResource;
    subResource.pSysMem = image_data;
    subResource.SysMemPitch = desc.Width * 4;
    subResource.SysMemSlicePitch = 0;
    device->CreateTexture2D(&desc, &subResource, &pTexture);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    device->CreateShaderResourceView(pTexture, &srvDesc, out_srv);
    pTexture->Release();

    *out_width = image_width;
    *out_height = image_height;
    stbi_image_free(image_data);

    return true;
}

std::string get_ghub_version()
{
    std::string line;
    std::ifstream in("C:\\Program Files\\LGHUB\\version");
    if (in.is_open())
    {
        while (std::getline(in, line)) { }
    }
    in.close();

    if (line.data())
    {
        return line;
    }

	return "unknown";
}

bool contains_tensorrt(const std::string& path)
{
    std::string lowercase_path = path;
    std::transform(lowercase_path.begin(), lowercase_path.end(), lowercase_path.begin(), ::tolower);
    return lowercase_path.find("tensorrt") != std::string::npos;
}

int get_active_monitors()
{
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)))
    {
        return -1;
    }

    int monitorCount = 0;

    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        IDXGIOutput* output = nullptr;
        for (UINT j = 0; adapter->EnumOutputs(j, &output) != DXGI_ERROR_NOT_FOUND; ++j)
        {
            monitorCount++;
            output->Release();
        }
        adapter->Release();
    }

    factory->Release();
    return monitorCount;
}

HMONITOR GetMonitorHandleByIndex(int monitorIndex)
{
    struct MonitorSearch
    {
        int targetIndex;
        int currentIndex;
        HMONITOR targetMonitor;
    };

    MonitorSearch search = { monitorIndex, 0, nullptr };

    EnumDisplayMonitors(nullptr, nullptr,
        [](HMONITOR hMonitor, HDC, LPRECT, LPARAM lParam) -> BOOL
        {
            MonitorSearch* search = reinterpret_cast<MonitorSearch*>(lParam);
            if (search->currentIndex == search->targetIndex)
            {
                search->targetMonitor = hMonitor;
                return FALSE;
            }
            search->currentIndex++;
            return TRUE;
        },
        reinterpret_cast<LPARAM>(&search));

    return search.targetMonitor;
}

std::vector<std::string> getAvailableModels()
{
    std::vector<std::string> availableModels;

    std::vector<std::string> engineFiles;
    std::vector<std::string> onnxFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const auto ext = entry.path().extension();
        if (ext == ".engine")
        {
            engineFiles.push_back(entry.path().filename().string());
        }
        else if (ext == ".onnx")
        {
            onnxFiles.push_back(entry.path().filename().string());
        }
    }

    if (config.backend == "TRT")
    {
        std::set<std::string> engineModels;
        for (const auto& file : engineFiles)
        {
            engineModels.insert(std::filesystem::path(file).stem().string());
        }

        for (const auto& file : engineFiles)
        {
            availableModels.push_back(file);
        }

        for (const auto& file : onnxFiles)
        {
            std::string modelName = std::filesystem::path(file).stem().string();
            if (engineModels.find(modelName) == engineModels.end())
            {
                availableModels.push_back(file);
            }
        }
    }
    else
    {
        for (const auto& file : engineFiles)
        {
            availableModels.push_back(file);
        }

        for (const auto& file : onnxFiles)
        {
            availableModels.push_back(file);
        }
    }

    return availableModels;
}

void SetRandomConsoleTitle()
{
    static constexpr std::array<const wchar_t*, 10> kTitles = {
        L"Microsoft Edge",
        L"Google Chrome",
        L"Notepad",
        L"Windows Terminal",
        L"PowerShell",
        L"Visual Studio Code",
        L"Task Manager",
        L"File Explorer",
        L"Calculator",
        L"Command Prompt",
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, kTitles.size() - 1);

    ::SetConsoleTitleW(kTitles[dist(gen)]);
}

bool checkwin1903()
{
    return IsWindows10OrGreater() && IsWindowsVersionOrGreater(10, 0, 18362);
}

void welcome_message()
{
    std::cout <<
    "\n\nAimbot is started!\n" <<
    config.joinStrings(config.button_targeting) << " -> Targeting\n" <<
    config.joinStrings(config.button_exit) << " -> Exit\n" <<
    config.joinStrings(config.button_pause) << " -> Pause Aiming\n" <<
    config.joinStrings(config.button_reload_config) << " -> Reload Config\n" <<
    config.joinStrings(config.button_open_overlay) << " -> Overlay (OPTIONS)" <<
    std::endl;
}

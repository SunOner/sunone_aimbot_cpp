#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <Windows.h>

#include <string>
#include <iostream>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <array>
#include <fstream>
#include <random>
#include <set>
#include <vector>

#include <d3d11.h>
#include <dxgi.h>
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

class ScopedComInit
{
public:
    explicit ScopedComInit(DWORD coinit) noexcept
        : hr(CoInitializeEx(nullptr, coinit))
    {
    }

    ~ScopedComInit()
    {
        if (hr == S_OK || hr == S_FALSE)
        {
            CoUninitialize();
        }
    }

    bool ok() const noexcept
    {
        return hr == S_OK || hr == S_FALSE || hr == RPC_E_CHANGED_MODE;
    }

    HRESULT hr;
};

static std::filesystem::path MakePathFromUtf8(const char* filename)
{
    if (!filename || !*filename)
    {
        return {};
    }

    try
    {
        return std::filesystem::u8path(filename);
    }
    catch (...)
    {
        return std::filesystem::path(filename);
    }
}

static bool ReadFileToBuffer(const std::filesystem::path& path, std::vector<unsigned char>& out)
{
    out.clear();

    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec)
    {
        return false;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        return false;
    }

    std::streamsize size = file.tellg();
    if (size <= 0)
    {
        return false;
    }

    out.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(out.data()), size))
    {
        out.clear();
        return false;
    }

    return true;
}

static std::vector<std::string> GetModelFilesByExtInDir(const std::string& dir, const std::vector<std::string>& exts)
{
    std::vector<std::string> files;

    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec)
    {
        return files;
    }

    for (const auto& entry : it)
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        const bool match = std::any_of(exts.begin(), exts.end(),
            [&ext](const std::string& e) { return ext == e; });
        if (match)
        {
            files.push_back(entry.path().filename().string());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<std::string> GetModelFilesByExt(const std::vector<std::string>& exts)
{
    return GetModelFilesByExtInDir("models/", exts);
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

    ScopedComInit com(COINIT_MULTITHREADED);
    if (!com.ok())
    {
        outErr = "CoInitializeEx failed: " + hr_to_string(com.hr);
        return false;
    }

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
    std::vector<unsigned char> ret;

    if (encoded_string.empty())
    {
        return ret;
    }

    size_t start = 0;
    size_t base64_pos = encoded_string.find("base64,");
    if (base64_pos != std::string::npos)
    {
        start = base64_pos + 7;
    }

    std::string filtered;
    filtered.reserve(encoded_string.size() - start);
    for (size_t idx = start; idx < encoded_string.size(); ++idx)
    {
        unsigned char c = static_cast<unsigned char>(encoded_string[idx]);
        if (c == '=')
        {
            filtered.push_back('=');
            continue;
        }
        if (is_base64(c))
        {
            filtered.push_back(static_cast<char>(c));
            continue;
        }
        if (std::isspace(c))
        {
            continue;
        }
    }

    if (filtered.empty())
    {
        return ret;
    }

    int in_len = static_cast<int>(filtered.size());
    int i = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (filtered[in_] != '=') && is_base64(static_cast<unsigned char>(filtered[in_])))
    {
        char_array_4[i++] = static_cast<unsigned char>(filtered[in_]); in_++;
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
    std::filesystem::path fsPath;
    try
    {
        fsPath = std::filesystem::u8path(path);
    }
    catch (...)
    {
        fsPath = std::filesystem::path(path);
    }

    std::error_code ec;
    return std::filesystem::exists(fsPath, ec) && !ec;
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

std::vector<std::string> getModelFiles()
{
    return GetModelFilesByExt({ ".engine", ".onnx" });
}

std::vector<std::string> getEngineFiles()
{
    return GetModelFilesByExt({ ".engine" });
}

std::vector<std::string> getOnnxFiles()
{
    return GetModelFilesByExt({ ".onnx" });
}

std::vector<std::string> getAvailableDepthModels()
{
    return GetModelFilesByExtInDir("models/depth", { ".engine", ".onnx", ".trt", ".plan" });
}

std::vector<std::string>::difference_type getModelIndex(const std::vector<std::string>& engine_models)
{
    auto it = std::find(engine_models.begin(), engine_models.end(), config.ai_model);

    if (it != engine_models.end())
    {
        return std::distance(engine_models.begin(), it);
    }
    else
    {
        return -1; // not found
    }
}

bool LoadTextureFromFile(const char* filename, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height)
{
    if (out_srv)
    {
        *out_srv = nullptr;
    }
    if (out_width)
    {
        *out_width = 0;
    }
    if (out_height)
    {
        *out_height = 0;
    }

    if (!filename || !device || !out_srv || !out_width || !out_height)
        return false;

    int image_width = 0;
    int image_height = 0;
    int channels = 0;
    const std::filesystem::path path = MakePathFromUtf8(filename);
    std::vector<unsigned char> fileData;
    if (path.empty() || !ReadFileToBuffer(path, fileData))
    {
        return false;
    }

    unsigned char* image_data = stbi_load_from_memory(
        fileData.data(),
        static_cast<int>(fileData.size()),
        &image_width,
        &image_height,
        &channels,
        4);
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
    HRESULT hr = device->CreateTexture2D(&desc, &subResource, &pTexture);
    if (FAILED(hr))
    {
        stbi_image_free(image_data);
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    hr = device->CreateShaderResourceView(pTexture, &srvDesc, out_srv);
    if (FAILED(hr))
    {
        pTexture->Release();
        stbi_image_free(image_data);
        return false;
    }
    pTexture->Release();

    *out_width = image_width;
    *out_height = image_height;
    stbi_image_free(image_data);

    return true;
}

bool LoadTextureFromMemory(const std::string& imageBase64, ID3D11Device* device, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height)
{
    if (out_srv)
    {
        *out_srv = nullptr;
    }
    if (out_width)
    {
        *out_width = 0;
    }
    if (out_height)
    {
        *out_height = 0;
    }

    if (!device || !out_srv || !out_width || !out_height)
        return false;

    std::vector<unsigned char> decodedData = Base64Decode(imageBase64);
    if (decodedData.empty())
    {
        return false;
    }

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
    HRESULT hr = device->CreateTexture2D(&desc, &subResource, &pTexture);
    if (FAILED(hr))
    {
        stbi_image_free(image_data);
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    hr = device->CreateShaderResourceView(pTexture, &srvDesc, out_srv);
    if (FAILED(hr))
    {
        pTexture->Release();
        stbi_image_free(image_data);
        return false;
    }
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

    if (!line.empty())
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

    std::vector<std::string> engineFiles = getEngineFiles();
    std::vector<std::string> onnxFiles = getOnnxFiles();

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

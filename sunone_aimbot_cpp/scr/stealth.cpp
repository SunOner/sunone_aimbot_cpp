#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <string>
#include <random>
#include <array>
#include <filesystem>
#include <iostream>
#include <algorithm>

#include "stealth.h"

namespace Stealth
{
    // =========================================================================
    // Internal RNG
    // =========================================================================
    static std::mt19937& GetRNG()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

    // =========================================================================
    // Random string generators
    // =========================================================================
    std::string GenerateRandomString(int minLen, int maxLen)
    {
        static const char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789";
        auto& gen = GetRNG();
        std::uniform_int_distribution<int> lenDist(minLen, maxLen);
        std::uniform_int_distribution<int> charDist(0, sizeof(charset) - 2);

        int len = lenDist(gen);
        std::string result;
        result.reserve(len);
        for (int i = 0; i < len; ++i)
            result += charset[charDist(gen)];
        return result;
    }

    std::wstring GenerateRandomWString(int minLen, int maxLen)
    {
        auto s = GenerateRandomString(minLen, maxLen);
        return std::wstring(s.begin(), s.end());
    }

    // =========================================================================
    // Generate realistic process name
    // =========================================================================
    std::string GenerateProcessName()
    {
        static const std::array<const char*, 24> prefixes = {
            "svc", "runtime", "host", "service", "agent", "updater",
            "helper", "daemon", "monitor", "broker", "provider", "manager",
            "worker", "bridge", "handler", "driver", "compat", "diag",
            "secure", "system", "core", "module", "native", "platform"
        };

        static const std::array<const char*, 16> suffixes = {
            "host", "svc", "mgr", "ctl", "mon", "app",
            "proc", "exec", "run", "sys", "io", "net",
            "cfg", "init", "eng", "lib"
        };

        auto& gen = GetRNG();
        std::uniform_int_distribution<size_t> prefDist(0, prefixes.size() - 1);
        std::uniform_int_distribution<size_t> sufDist(0, suffixes.size() - 1);
        std::uniform_int_distribution<int> numDist(0, 99);
        std::uniform_int_distribution<int> styleDist(0, 4);

        std::string name;
        int style = styleDist(gen);

        switch (style)
        {
        case 0: // "RuntimeBroker" - CamelCase
            {
                std::string p = prefixes[prefDist(gen)];
                std::string s = suffixes[sufDist(gen)];
                p[0] = static_cast<char>(toupper(p[0]));
                s[0] = static_cast<char>(toupper(s[0]));
                name = p + s;
            }
            break;
        case 1: // "svchost32" - prefix + number
            name = std::string(prefixes[prefDist(gen)]) + std::to_string(numDist(gen));
            break;
        case 2: // "host_service" - underscore
            name = std::string(prefixes[prefDist(gen)]) + "_" + std::string(suffixes[sufDist(gen)]);
            break;
        case 3: // "diagproc" - concat lowercase
            name = std::string(prefixes[prefDist(gen)]) + std::string(suffixes[sufDist(gen)]);
            break;
        case 4: // "NativeHost64"
            {
                std::string p = prefixes[prefDist(gen)];
                std::string s = suffixes[sufDist(gen)];
                p[0] = static_cast<char>(toupper(p[0]));
                s[0] = static_cast<char>(toupper(s[0]));
                name = p + s + "64";
            }
            break;
        }

        return name;
    }

    // =========================================================================
    // Self-rename exe - Windows allows renaming a running executable
    // =========================================================================
    bool SelfRenameExe()
    {
        wchar_t exePath[MAX_PATH]{};
        if (GetModuleFileNameW(nullptr, exePath, MAX_PATH) == 0)
            return false;

        std::filesystem::path currentPath(exePath);
        std::string currentStem = currentPath.stem().string();

        std::string lowerStem = currentStem;
        std::transform(lowerStem.begin(), lowerStem.end(), lowerStem.begin(), ::tolower);

        // Only rename if name contains suspicious keywords
        static const std::array<std::string, 8> suspiciousKeywords = {
            "ai", "aim", "bot", "cheat", "hack", "inject", "hook", "sunone"
        };

        bool needsRename = false;
        for (const auto& keyword : suspiciousKeywords)
        {
            if (lowerStem.find(keyword) != std::string::npos)
            {
                needsRename = true;
                break;
            }
        }

        if (!needsRename)
            return false; // already clean name

        // Generate new name
        std::string newName = GenerateProcessName();
        std::filesystem::path newPath = currentPath.parent_path() / (newName + ".exe");

        // Avoid collision
        int attempts = 0;
        while (std::filesystem::exists(newPath) && attempts < 10)
        {
            newName = GenerateProcessName();
            newPath = currentPath.parent_path() / (newName + ".exe");
            ++attempts;
        }

        if (std::filesystem::exists(newPath))
            return false;

        // Rename (Windows allows renaming a running exe)
        std::error_code ec;
        std::filesystem::rename(currentPath, newPath, ec);
        if (ec)
        {
            if (!MoveFileW(currentPath.c_str(), newPath.c_str()))
            {
                std::cerr << "[Stealth] Rename failed, error code: " << GetLastError() << std::endl;
                return false;
            }
        }

        std::cout << "[Init] Process: " << newName << ".exe" << std::endl;
        return true;
    }

    // =========================================================================
    // Randomize console title
    // =========================================================================
    void RandomizeConsoleTitle()
    {
        static const std::array<const wchar_t*, 20> baseTitles = {
            L"Windows PowerShell",
            L"Command Prompt",
            L"Terminal",
            L"Administrator: Windows PowerShell",
            L"Windows Update",
            L"Microsoft Edge",
            L"Settings",
            L"System Configuration",
            L"Windows Security",
            L"Device Manager",
            L"Resource Monitor",
            L"Performance Monitor",
            L"Local Group Policy Editor",
            L"Windows Defender",
            L"Disk Management",
            L"Event Viewer",
            L"Registry Editor",
            L"Component Services",
            L"Microsoft Management Console",
            L"System Information"
        };

        auto& gen = GetRNG();
        std::uniform_int_distribution<size_t> dist(0, baseTitles.size() - 1);
        ::SetConsoleTitleW(baseTitles[dist(gen)]);
    }

    // =========================================================================
    // Random window class name (looks like legit Windows internals)
    // =========================================================================
    std::wstring GenerateWindowClass()
    {
        static const std::array<const wchar_t*, 12> classTemplates = {
            L"DWM_Composition_",
            L"Windows.UI.Core.",
            L"MSCTFIME_",
            L"DirectUI_",
            L"Shell_",
            L"Explorer_",
            L"Afx_",
            L"WinUI_",
            L"CoreWindow_",
            L"RenderWidget_",
            L"Chrome_",
            L"Gecko_"
        };

        auto& gen = GetRNG();
        std::uniform_int_distribution<size_t> dist(0, classTemplates.size() - 1);
        return std::wstring(classTemplates[dist(gen)]) + GenerateRandomWString(6, 8);
    }

    // =========================================================================
    // Random window title
    // =========================================================================
    std::wstring GenerateWindowTitle()
    {
        static const std::array<const wchar_t*, 10> titles = {
            L"",
            L"Untitled",
            L"Desktop Window Manager",
            L"Shell Infrastructure Host",
            L"Microsoft Text Input Application",
            L"Windows Input Experience",
            L"Windows Shell Experience Host",
            L"Search",
            L"Start",
            L"Widgets"
        };

        auto& gen = GetRNG();
        std::uniform_int_distribution<size_t> dist(0, titles.size() - 1);
        return std::wstring(titles[dist(gen)]);
    }

    // =========================================================================
    // Init all stealth features at startup
    // =========================================================================
    void InitStealth()
    {
        SelfRenameExe();
        RandomizeConsoleTitle();
    }
}

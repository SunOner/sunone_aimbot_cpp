#include <iostream>
#include <string>

#include "ghub.h"

// =============================================================================
// Win32 fallback helpers
// =============================================================================

UINT GhubMouse::ghub_SendInput(UINT nInputs, LPINPUT pInputs)
{
    return SendInput(nInputs, pInputs, sizeof(INPUT));
}

INPUT GhubMouse::ghub_Mouse(DWORD flags, LONG x, LONG y, DWORD data)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dx = x;
    input.mi.dy = y;
    input.mi.mouseData = data;
    input.mi.dwFlags = flags;
    return input;
}

// =============================================================================
// Legacy method: ghub_mouse.dll
// =============================================================================

bool GhubMouse::initLegacy()
{
    auto dllPath = basedir / "ghub_mouse.dll";
    hLegacy = LoadLibraryA(dllPath.string().c_str());
    if (!hLegacy)
    {
        std::cerr << "[Ghub/Legacy] Failed to load ghub_mouse.dll" << std::endl;
        return false;
    }

    pLegacyMouseOpen  = reinterpret_cast<LegacyMouseOpenFunc>(GetProcAddress(hLegacy, "mouse_open"));
    pLegacyMoveR      = reinterpret_cast<LegacyMoveRFunc>(GetProcAddress(hLegacy, "moveR"));
    pLegacyPress      = reinterpret_cast<LegacyPressFunc>(GetProcAddress(hLegacy, "press"));
    pLegacyRelease    = reinterpret_cast<LegacyReleaseFunc>(GetProcAddress(hLegacy, "release"));
    pLegacyMouseClose = reinterpret_cast<LegacyMouseCloseFunc>(GetProcAddress(hLegacy, "mouse_close"));

    if (!pLegacyMouseOpen)
    {
        std::cerr << "[Ghub/Legacy] Failed to get mouse_open from ghub_mouse.dll" << std::endl;
        FreeLibrary(hLegacy);
        hLegacy = NULL;
        return false;
    }

    bool opened = pLegacyMouseOpen();
    if (!opened)
    {
        std::cerr << "[Ghub/Legacy] mouse_open() returned false" << std::endl;
        FreeLibrary(hLegacy);
        hLegacy = NULL;
        return false;
    }

    std::cout << "[Ghub/Legacy] Initialized successfully using ghub_mouse.dll" << std::endl;
    return true;
}

void GhubMouse::cleanupLegacy()
{
    if (hLegacy)
    {
        if (pLegacyMouseClose)
        {
            pLegacyMouseClose();
        }
        FreeLibrary(hLegacy);
        hLegacy = NULL;
    }
    pLegacyMouseOpen  = nullptr;
    pLegacyMoveR      = nullptr;
    pLegacyPress      = nullptr;
    pLegacyRelease    = nullptr;
    pLegacyMouseClose = nullptr;
}

// =============================================================================
// New method: IbInputSimulator.dll
// =============================================================================

bool GhubMouse::initNew()
{
    auto dllPath = basedir / "IbInputSimulator.dll";
    hIbSim = LoadLibraryA(dllPath.string().c_str());
    if (!hIbSim)
    {
        std::cerr << "[Ghub/New] Failed to load IbInputSimulator.dll (Error: "
                  << GetLastError() << ")" << std::endl;
        return false;
    }

    // Load function pointers
    pIbSendInit       = reinterpret_cast<IbSendInitFunc>(GetProcAddress(hIbSim, "IbSendInit"));
    pIbSendDestroy    = reinterpret_cast<IbSendDestroyFunc>(GetProcAddress(hIbSim, "IbSendDestroy"));
    pIbSendInput      = reinterpret_cast<IbSendInputFunc>(GetProcAddress(hIbSim, "IbSendInput"));
    pIbSendMouseMove  = reinterpret_cast<IbSendMouseMoveFunc>(GetProcAddress(hIbSim, "IbSendMouseMove"));
    pIbSendMouseClick = reinterpret_cast<IbSendMouseClickFunc>(GetProcAddress(hIbSim, "IbSendMouseClick"));

    if (!pIbSendInit || !pIbSendDestroy)
    {
        std::cerr << "[Ghub/New] Failed to get IbSendInit/IbSendDestroy from IbInputSimulator.dll" << std::endl;
        FreeLibrary(hIbSim);
        hIbSim = NULL;
        pIbSendInit = nullptr;
        pIbSendDestroy = nullptr;
        pIbSendInput = nullptr;
        pIbSendMouseMove = nullptr;
        pIbSendMouseClick = nullptr;
        return false;
    }

    if (!pIbSendInput && !pIbSendMouseMove)
    {
        std::cerr << "[Ghub/New] Failed to get IbSendInput or IbSendMouseMove" << std::endl;
        FreeLibrary(hIbSim);
        hIbSim = NULL;
        pIbSendInit = nullptr;
        pIbSendDestroy = nullptr;
        pIbSendInput = nullptr;
        pIbSendMouseMove = nullptr;
        pIbSendMouseClick = nullptr;
        return false;
    }

    // Initialize with LogitechGHubNew (type=6), flags=0, arg=nullptr
    // Send::SendType::LogitechGHubNew = 6
    int result = pIbSendInit(6, 0, nullptr);
    if (result != 0)
    {
        // Error codes: 1=InvalidArgument, 2=LibraryNotFound, 3=LibraryLoadFailed,
        //              4=LibraryError, 5=DeviceCreateFailed, 6=DeviceNotFound, 7=DeviceOpenFailed
        const char* errorNames[] = {
            "Success", "InvalidArgument", "LibraryNotFound", "LibraryLoadFailed",
            "LibraryError", "DeviceCreateFailed", "DeviceNotFound", "DeviceOpenFailed"
        };
        const char* errName = (result >= 0 && result <= 7) ? errorNames[result] : "Unknown";

        std::cerr << "[Ghub/New] IbSendInit(LogitechGHubNew) failed: "
                  << errName << " (code=" << result << ")" << std::endl;

        // If LogitechGHubNew fails, try old Logitech driver (type=2)
        std::cout << "[Ghub/New] Trying Logitech legacy driver via IbInputSimulator..." << std::endl;
        result = pIbSendInit(2, 0, nullptr);
        if (result != 0)
        {
            errName = (result >= 0 && result <= 7) ? errorNames[result] : "Unknown";
            std::cerr << "[Ghub/New] IbSendInit(Logitech) also failed: "
                      << errName << " (code=" << result << ")" << std::endl;
            FreeLibrary(hIbSim);
            hIbSim = NULL;
            pIbSendInit = nullptr;
            pIbSendDestroy = nullptr;
            pIbSendInput = nullptr;
            pIbSendMouseMove = nullptr;
            pIbSendMouseClick = nullptr;
            return false;
        }
        std::cout << "[Ghub/New] Initialized successfully using IbInputSimulator (Logitech Legacy driver)" << std::endl;
    }
    else
    {
        std::cout << "[Ghub/New] Initialized successfully using IbInputSimulator (LogitechGHubNew)" << std::endl;
    }

    return true;
}

void GhubMouse::cleanupNew()
{
    if (hIbSim)
    {
        if (pIbSendDestroy)
        {
            pIbSendDestroy();
        }
        FreeLibrary(hIbSim);
        hIbSim = NULL;
    }
    pIbSendInit       = nullptr;
    pIbSendDestroy    = nullptr;
    pIbSendInput      = nullptr;
    pIbSendMouseMove  = nullptr;
    pIbSendMouseClick = nullptr;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

GhubMouse::GhubMouse(const std::string& method)
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    basedir = std::filesystem::path(buffer).parent_path();

    if (method == "NEW")
    {
        // Only try IbInputSimulator
        if (initNew())
        {
            gmok = true;
            activeMethod = ActiveMethod::NEW;
            activeMethodName = "NEW (IbInputSimulator)";
            statusMessage = "IbInputSimulator initialized successfully";
        }
        else
        {
            gmok = false;
            statusMessage = "Failed to initialize IbInputSimulator.dll";
            std::cerr << "[Ghub] " << statusMessage << std::endl;
        }
    }
    else if (method == "LEGACY")
    {
        // Only try ghub_mouse.dll
        if (initLegacy())
        {
            gmok = true;
            activeMethod = ActiveMethod::LEGACY;
            activeMethodName = "LEGACY (ghub_mouse.dll)";
            statusMessage = "ghub_mouse.dll initialized successfully";
        }
        else
        {
            gmok = false;
            statusMessage = "Failed to initialize ghub_mouse.dll";
            std::cerr << "[Ghub] " << statusMessage << std::endl;
        }
    }
    else // "AUTO" or default
    {
        // Try IbInputSimulator first (works with newer G Hub)
        std::cout << "[Ghub/AUTO] Trying IbInputSimulator.dll (new method)..." << std::endl;
        if (initNew())
        {
            gmok = true;
            activeMethod = ActiveMethod::NEW;
            activeMethodName = "NEW (IbInputSimulator - Auto)";
            statusMessage = "Auto-detected: IbInputSimulator";
        }
        else
        {
            // Fall back to ghub_mouse.dll
            std::cout << "[Ghub/AUTO] Falling back to ghub_mouse.dll (legacy method)..." << std::endl;
            if (initLegacy())
            {
                gmok = true;
                activeMethod = ActiveMethod::LEGACY;
                activeMethodName = "LEGACY (ghub_mouse.dll - Auto)";
                statusMessage = "Auto-detected: ghub_mouse.dll";
            }
            else
            {
                gmok = false;
                statusMessage = "Both IbInputSimulator.dll and ghub_mouse.dll failed";
                std::cerr << "[Ghub] " << statusMessage << std::endl;
            }
        }
    }
}

GhubMouse::~GhubMouse()
{
    mouse_close();
}

// =============================================================================
// Mouse operations
// =============================================================================

bool GhubMouse::mouse_xy(int x, int y)
{
    if (!gmok)
    {
        // Fallback to Win32 SendInput
        INPUT input = ghub_Mouse(MOUSEEVENTF_MOVE, x, y);
        return ghub_SendInput(1, &input) == 1;
    }

    if (activeMethod == ActiveMethod::NEW)
    {
        // Prefer API3 IbSendMouseMove (mode=1 for relative)
        if (pIbSendMouseMove)
        {
            // For relative movement, cast signed int to uint32_t
            // IbSendMouseMove handles sign extension internally
            return pIbSendMouseMove(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 1);
        }
        // Fallback to API1 IbSendInput
        if (pIbSendInput)
        {
            INPUT input = ghub_Mouse(MOUSEEVENTF_MOVE, x, y);
            return pIbSendInput(1, &input, sizeof(INPUT)) == 1;
        }
    }
    else if (activeMethod == ActiveMethod::LEGACY)
    {
        if (pLegacyMoveR)
        {
            return pLegacyMoveR(x, y);
        }
    }

    // Final fallback
    INPUT input = ghub_Mouse(MOUSEEVENTF_MOVE, x, y);
    return ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_down(int key)
{
    if (!gmok)
    {
        DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
        INPUT input = ghub_Mouse(flag);
        return ghub_SendInput(1, &input) == 1;
    }

    if (activeMethod == ActiveMethod::NEW)
    {
        if (pIbSendMouseClick)
        {
            // MouseButton::LeftDown = 0x02, RightDown = 0x08
            uint32_t button = (key == 1) ? 0x02 : 0x08;
            return pIbSendMouseClick(button);
        }
        if (pIbSendInput)
        {
            DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
            INPUT input = ghub_Mouse(flag);
            return pIbSendInput(1, &input, sizeof(INPUT)) == 1;
        }
    }
    else if (activeMethod == ActiveMethod::LEGACY)
    {
        if (pLegacyPress)
        {
            return pLegacyPress(key);
        }
    }

    DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
    INPUT input = ghub_Mouse(flag);
    return ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_up(int key)
{
    if (!gmok)
    {
        DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTUP : MOUSEEVENTF_RIGHTUP;
        INPUT input = ghub_Mouse(flag);
        return ghub_SendInput(1, &input) == 1;
    }

    if (activeMethod == ActiveMethod::NEW)
    {
        if (pIbSendMouseClick)
        {
            // MouseButton::LeftUp = 0x04, RightUp = 0x10
            uint32_t button = (key == 1) ? 0x04 : 0x10;
            return pIbSendMouseClick(button);
        }
        if (pIbSendInput)
        {
            DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTUP : MOUSEEVENTF_RIGHTUP;
            INPUT input = ghub_Mouse(flag);
            return pIbSendInput(1, &input, sizeof(INPUT)) == 1;
        }
    }
    else if (activeMethod == ActiveMethod::LEGACY)
    {
        if (pLegacyRelease)
        {
            return pLegacyRelease();
        }
    }

    DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTUP : MOUSEEVENTF_RIGHTUP;
    INPUT input = ghub_Mouse(flag);
    return ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_close()
{
    bool result = false;

    if (activeMethod == ActiveMethod::LEGACY)
    {
        cleanupLegacy();
        result = true;
    }
    else if (activeMethod == ActiveMethod::NEW)
    {
        cleanupNew();
        result = true;
    }

    activeMethod = ActiveMethod::NONE;
    gmok = false;
    return result;
}

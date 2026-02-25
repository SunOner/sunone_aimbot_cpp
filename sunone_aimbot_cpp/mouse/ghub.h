#ifndef GHUB_H
#define GHUB_H

#include <filesystem>
#include <windows.h>
#include <string>

// GhubMethod determines which DLL/driver to use
// "AUTO"   - Try IbInputSimulator (new) first, fallback to ghub_mouse.dll (legacy)
// "LEGACY" - Use ghub_mouse.dll only (requires old G Hub / LGS)
// "NEW"    - Use IbInputSimulator.dll only (works with new G Hub versions)
class GhubMouse
{
public:
    GhubMouse(const std::string& method = "AUTO");
    ~GhubMouse();
    bool mouse_xy(int x, int y);
    bool mouse_down(int key = 1);
    bool mouse_up(int key = 1);
    bool mouse_close();

    bool isInitialized() const { return gmok; }
    std::string getActiveMethod() const { return activeMethodName; }
    std::string getStatusMessage() const { return statusMessage; }

private:
    std::filesystem::path basedir;
    bool gmok = false;
    std::string activeMethodName = "NONE";
    std::string statusMessage;

    // --- Legacy method (ghub_mouse.dll) ---
    HMODULE hLegacy = NULL;

    typedef bool(*LegacyMouseOpenFunc)();
    typedef bool(*LegacyMoveRFunc)(int, int);
    typedef bool(*LegacyPressFunc)(int);
    typedef bool(*LegacyReleaseFunc)();
    typedef bool(*LegacyMouseCloseFunc)();

    LegacyMouseOpenFunc  pLegacyMouseOpen  = nullptr;
    LegacyMoveRFunc      pLegacyMoveR      = nullptr;
    LegacyPressFunc      pLegacyPress      = nullptr;
    LegacyReleaseFunc    pLegacyRelease    = nullptr;
    LegacyMouseCloseFunc pLegacyMouseClose = nullptr;

    bool initLegacy();
    void cleanupLegacy();

    // --- New method (IbInputSimulator.dll) ---
    HMODULE hIbSim = NULL;

    // IbSendInit(int type, int flags, void* arg) -> int (0=success)
    typedef int(__stdcall* IbSendInitFunc)(int, int, void*);
    // IbSendDestroy()
    typedef void(__stdcall* IbSendDestroyFunc)();
    // IbSendInput(UINT nInputs, LPINPUT pInputs, int cbSize) -> UINT
    typedef UINT(WINAPI* IbSendInputFunc)(UINT, LPINPUT, int);
    // IbSendMouseMove(uint32_t x, uint32_t y, uint32_t mode) -> bool
    typedef bool(__stdcall* IbSendMouseMoveFunc)(uint32_t, uint32_t, uint32_t);
    // IbSendMouseClick(uint32_t button) -> bool
    typedef bool(__stdcall* IbSendMouseClickFunc)(uint32_t);

    IbSendInitFunc       pIbSendInit       = nullptr;
    IbSendDestroyFunc    pIbSendDestroy    = nullptr;
    IbSendInputFunc      pIbSendInput      = nullptr;
    IbSendMouseMoveFunc  pIbSendMouseMove  = nullptr;
    IbSendMouseClickFunc pIbSendMouseClick = nullptr;

    bool initNew();
    void cleanupNew();

    // --- Fallback (Win32 SendInput) ---
    static UINT ghub_SendInput(UINT nInputs, LPINPUT pInputs);
    static INPUT ghub_Mouse(DWORD flags, LONG x = 0, LONG y = 0, DWORD data = 0);

    // Active method flag
    enum class ActiveMethod { NONE, LEGACY, NEW };
    ActiveMethod activeMethod = ActiveMethod::NONE;
};

#endif // GHUB_H

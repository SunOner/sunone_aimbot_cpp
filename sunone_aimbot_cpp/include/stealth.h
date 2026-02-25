#ifndef STEALTH_H
#define STEALTH_H

#include <string>
#include <Windows.h>

namespace Stealth
{
    // Generate random string (lowercase + digits)
    std::string GenerateRandomString(int minLen = 6, int maxLen = 10);
    std::wstring GenerateRandomWString(int minLen = 6, int maxLen = 10);

    // Generate a realistic-looking process name (no extension)
    std::string GenerateProcessName();

    // Self-rename the running exe if its name contains suspicious keywords
    // Windows allows renaming a running executable
    bool SelfRenameExe();

    // Set console title to a random realistic Windows title
    void RandomizeConsoleTitle();

    // Generate a random window class name (looks like legit Windows class)
    std::wstring GenerateWindowClass();

    // Generate a random window title
    std::wstring GenerateWindowTitle();

    // Init all stealth features (call once at startup)
    void InitStealth();
}

#endif // STEALTH_H

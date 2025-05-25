#include <iostream>
#include <string>

#include "ghub.h"

UINT GhubMouse::_ghub_SendInput(UINT nInputs, LPINPUT pInputs)
{
    return SendInput(nInputs, pInputs, sizeof(INPUT));
}

INPUT GhubMouse::_ghub_Input(MOUSEINPUT mi)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi = mi;
    return input;
}

MOUSEINPUT GhubMouse::_ghub_MouseInput(DWORD flags, LONG x, LONG y, DWORD data)
{
    MOUSEINPUT mi = { 0 };
    mi.dx = x;
    mi.dy = y;
    mi.mouseData = data;
    mi.dwFlags = flags;
    return mi;
}

INPUT GhubMouse::_ghub_Mouse(DWORD flags, LONG x, LONG y, DWORD data)
{
    return _ghub_Input(_ghub_MouseInput(flags, x, y, data));
}

GhubMouse::GhubMouse()
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    basedir = std::filesystem::path(buffer).parent_path();
    dlldir = basedir / "ghub_mouse.dll";
    gm = LoadLibraryA(dlldir.string().c_str());
    if (gm == NULL)
    {
        std::cerr << "[Ghub] Failed to load DLL" << std::endl;
        gmok = false;
    }
    else
    {
        auto mouse_open = reinterpret_cast<bool(*)()>(GetProcAddress(gm, "mouse_open"));
        if (mouse_open == NULL)
        {
            std::cerr << "[Ghub] Failed to open mouse!" << std::endl;
            gmok = false;
        }
        else
        {
            gmok = mouse_open();
        }
    }
}

GhubMouse::~GhubMouse()
{
    if (gm != NULL)
    {
        FreeLibrary(gm);
    }
}

bool GhubMouse::mouse_xy(int x, int y)
{
    if (gmok)
    {
        auto moveR = reinterpret_cast<bool(*)(int, int)>(GetProcAddress(gm, "moveR"));
        if (moveR != NULL)
        {
            return moveR(x, y);
        }
    }
    INPUT input = _ghub_Mouse(MOUSEEVENTF_MOVE, x, y);
    return _ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_down(int key)
{
    if (gmok)
    {
        auto press = reinterpret_cast<bool(*)(int)>(GetProcAddress(gm, "press"));
        if (press != NULL)
        {
            return press(key);
        }
    }
    DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
    INPUT input = _ghub_Mouse(flag);
    return _ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_up(int key)
{
    if (gmok)
    {
        auto release = reinterpret_cast<bool(*)()>(GetProcAddress(gm, "release"));
        if (release != NULL)
        {
            return release();
        }
    }
    DWORD flag = (key == 1) ? MOUSEEVENTF_LEFTUP : MOUSEEVENTF_RIGHTUP;
    INPUT input = _ghub_Mouse(flag);
    return _ghub_SendInput(1, &input) == 1;
}

bool GhubMouse::mouse_close()
{
    if (gmok)
    {
        auto mouse_close = reinterpret_cast<bool(*)()>(GetProcAddress(gm, "mouse_close"));
        if (mouse_close != NULL)
        {
            return mouse_close();
        }
    }
    return false;
}
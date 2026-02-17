#include "KmboxAConnection.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include "kmboxA.h"

bool KmboxAConnection::parsePidVid(const std::string& pidvid, unsigned short& pid, unsigned short& vid)
{
    std::string hex;
    hex.reserve(pidvid.size());

    for (unsigned char c : pidvid)
    {
        if (std::isxdigit(c))
        {
            hex.push_back(static_cast<char>(std::toupper(c)));
        }
    }

    if (hex.size() != 8)
    {
        return false;
    }

    try
    {
        // Single field format: PIDVID (PPPPVVVV).
        pid = static_cast<unsigned short>(std::stoul(hex.substr(0, 4), nullptr, 16));
        vid = static_cast<unsigned short>(std::stoul(hex.substr(4, 4), nullptr, 16));
        return true;
    }
    catch (...)
    {
        return false;
    }
}

KmboxAConnection::KmboxAConnection(const std::string& pidvid)
    : is_open_(false)
{
    unsigned short pid = 0;
    unsigned short vid = 0;

    if (!parsePidVid(pidvid, pid, vid))
    {
        std::cerr << "[KmboxA] Invalid PIDVID format. Expected 8 hex chars (PPPPVVVV)." << std::endl;
        return;
    }

    const int ret = KM_init(vid, pid);
    is_open_ = (ret == 0);
    if (!is_open_)
    {
        std::cerr << "[KmboxA] Connection failed, ret=" << ret << " (VID=0x"
            << std::hex << std::uppercase << vid << ", PID=0x" << pid << std::dec << ")" << std::endl;
    }
}

KmboxAConnection::~KmboxAConnection()
{
    if (!is_open_) return;
    KM_close();
    is_open_ = false;
}

void KmboxAConnection::move(int x, int y)
{
    if (!is_open_) return;
    KM_move(static_cast<short>(x), static_cast<short>(y));
}

void KmboxAConnection::leftDown()
{
    if (!is_open_) return;
    KM_left(1);
}

void KmboxAConnection::leftUp()
{
    if (!is_open_) return;
    KM_left(0);
}

void KmboxAConnection::rightDown()
{
    if (!is_open_) return;
    KM_right(1);
}

void KmboxAConnection::rightUp()
{
    if (!is_open_) return;
    KM_right(0);
}

void KmboxAConnection::middleDown()
{
    if (!is_open_) return;
    KM_middle(1);
}

void KmboxAConnection::middleUp()
{
    if (!is_open_) return;
    KM_middle(0);
}

void KmboxAConnection::wheel(int delta)
{
    if (!is_open_) return;
    const int clamped = std::clamp(delta, -127, 127);
    const signed char wheel_delta = static_cast<signed char>(clamped);
    KM_wheel(static_cast<unsigned char>(wheel_delta));
}


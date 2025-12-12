#include <iostream>
#include <thread>
#include <chrono>

#include "kmbox_net/kmboxNet.h"
#include "KmboxNetConnection.h"
#include <sunone_aimbot_cpp.h>

KmboxNetConnection::KmboxNetConnection(const std::string& ip, const std::string& port, const std::string& uuid)
    : is_open_(false), ip_(ip), port_(port), uuid_(uuid), monitor_(false)
{
    int ret = kmNet_init((char*)ip.c_str(), (char*)port.c_str(), (char*)uuid.c_str());
    is_open_ = (ret == 0);
    if (!is_open_)
    {
        std::cerr << "[KmboxNet] Connection failed, ret=" << ret << std::endl;
        return;
    }

    aiming_active = false;
    shooting_active = false;

    monitor_ = false;
    if (monitor_thread_.joinable())
        monitor_thread_.join();

    monitor_thread_ = std::thread(&KmboxNetConnection::monitorThread, this);
}

void KmboxNetConnection::monitorThread()
{
    kmNet_monitor(10000);

    while (monitor_running_)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

KmboxNetConnection::~KmboxNetConnection()
{
    monitor_running_ = false;
    if (monitor_thread_.joinable())
        monitor_thread_.join();
    kmNet_monitor(0);
    WSACleanup();
}

void KmboxNetConnection::move(int x, int y)
{
    if (!is_open_) return;
    kmNet_mouse_move((short)x, (short)y);
}

void KmboxNetConnection::moveAuto(int x, int y, int ms)
{
    if (!is_open_) return;
    kmNet_mouse_move_auto(x, y, ms);
}

void KmboxNetConnection::moveBezier(int x, int y, int ms, int x1, int y1, int x2, int y2)
{
    if (!is_open_) return;
    kmNet_mouse_move_beizer(x, y, ms, x1, y1, x2, y2);
}

void KmboxNetConnection::leftDown()
{
    if (!is_open_) return;
    kmNet_mouse_left(1);
}

void KmboxNetConnection::leftUp()
{
    if (!is_open_) return;
    kmNet_mouse_left(0);
}

void KmboxNetConnection::rightDown()
{
    if (!is_open_) return;
    kmNet_mouse_right(1);
}

void KmboxNetConnection::rightUp()
{
    if (!is_open_) return;
    kmNet_mouse_right(0);
}

void KmboxNetConnection::middleDown()
{
    if (!is_open_) return;
    kmNet_mouse_middle(1);
}

void KmboxNetConnection::middleUp()
{
    if (!is_open_) return;
    kmNet_mouse_middle(0);
}

void KmboxNetConnection::wheel(int wheel)
{
    if (!is_open_) return;
    kmNet_mouse_wheel(wheel);
}

void KmboxNetConnection::mouseAll(int button, int x, int y, int wheel)
{
    if (!is_open_) return;
    kmNet_mouse_all(button, x, y, wheel);
}

void KmboxNetConnection::keyDown(int vkey)
{
    if (!is_open_) return;
    kmNet_keydown(vkey);
}

void KmboxNetConnection::keyUp(int vkey)
{
    if (!is_open_) return;
    kmNet_keyup(vkey);
}

void KmboxNetConnection::monitor(short port)
{
    if (!is_open_) return;
    kmNet_monitor(port);
}

int KmboxNetConnection::monitorMouseLeft()
{
    if (!is_open_) return -1;
    return kmNet_monitor_mouse_left();
}

int KmboxNetConnection::monitorMouseRight()
{
    if (!is_open_) return -1;
    return kmNet_monitor_mouse_right();
}

int KmboxNetConnection::monitorMouseMiddle()
{
    if (!is_open_) return -1;
    return kmNet_monitor_mouse_middle();
}

int KmboxNetConnection::monitorMouseSide1()
{
    if (!is_open_) return -1;
    return kmNet_monitor_mouse_side1();
}

int KmboxNetConnection::monitorMouseSide2()
{
    if (!is_open_) return -1;
    return kmNet_monitor_mouse_side2();
}

int KmboxNetConnection::monitorKeyboard(short vkey)
{
    if (!is_open_) return -1;
    return kmNet_monitor_keyboard(vkey);
}

void KmboxNetConnection::maskMouseLeft(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_left(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseRight(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_right(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseMiddle(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_middle(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseSide1(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_side1(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseSide2(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_side2(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseX(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_x(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseY(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_y(enable ? 1 : 0);
}
void KmboxNetConnection::maskMouseWheel(bool enable)
{
    if (!is_open_) return;
    kmNet_mask_mouse_wheel(enable ? 1 : 0);
}
void KmboxNetConnection::maskKeyboard(short vkey)
{
    if (!is_open_) return;
    kmNet_mask_keyboard(vkey);
}
void KmboxNetConnection::unmaskKeyboard(short vkey)
{
    if (!is_open_) return;
    kmNet_unmask_keyboard(vkey);
}
void KmboxNetConnection::unmaskAll()
{
    if (!is_open_) return;
    kmNet_unmask_all();
}

void KmboxNetConnection::reboot()
{
    if (!is_open_) return;
    kmNet_reboot();
}

void KmboxNetConnection::setConfig(const std::string& ip, unsigned short port)
{
    if (!is_open_) return;
    kmNet_setconfig((char*)ip.c_str(), port);
}

void KmboxNetConnection::debug(short port, char enable)
{
    if (!is_open_) return;
    kmNet_debug(port, enable);
}

void KmboxNetConnection::lcdColor(unsigned short rgb565)
{
    if (!is_open_) return;
    kmNet_lcd_color(rgb565);
}
void KmboxNetConnection::lcdPictureBottom(unsigned char* buff_128_80)
{
    if (!is_open_) return;
    kmNet_lcd_picture_bottom(buff_128_80);
}
void KmboxNetConnection::lcdPicture(unsigned char* buff_128_160)
{
    if (!is_open_) return;
    kmNet_lcd_picture(buff_128_160);
}
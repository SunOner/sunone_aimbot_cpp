#pragma once
#include <string>
#include <atomic>
#include <mutex>
#include <thread>

#include "kmbox_net/kmboxNet.h"

class KmboxNetConnection
{
public:
    KmboxNetConnection(const std::string& ip, const std::string& port, const std::string& uuid);
    ~KmboxNetConnection();

    void monitorThread();
    bool isOpen() const { return is_open_; }

    void move(int x, int y);
    void moveAuto(int x, int y, int ms);
    void moveBezier(int x, int y, int ms, int x1, int y1, int x2, int y2);
    void leftDown();
    void leftUp();
    void rightDown();
    void rightUp();
    void middleDown();
    void middleUp();
    void wheel(int wheel);
    void mouseAll(int button, int x, int y, int wheel);

    void keyDown(int vkey);
    void keyUp(int vkey);

    void monitor(short port);
    int monitorMouseLeft();
    int monitorMouseRight();
    int monitorMouseMiddle();
    int monitorMouseSide1();
    int monitorMouseSide2();
    int monitorKeyboard(short vkey);

    void maskMouseLeft(bool enable);
    void maskMouseRight(bool enable);
    void maskMouseMiddle(bool enable);
    void maskMouseSide1(bool enable);
    void maskMouseSide2(bool enable);
    void maskMouseX(bool enable);
    void maskMouseY(bool enable);
    void maskMouseWheel(bool enable);
    void maskKeyboard(short vkey);
    void unmaskKeyboard(short vkey);
    void unmaskAll();

    void reboot();
    void setConfig(const std::string& ip, unsigned short port);
    void debug(short port, char enable);

    void lcdColor(unsigned short rgb565);
    void lcdPictureBottom(unsigned char* buff_128_80);
    void lcdPicture(unsigned char* buff_128_160);

    std::atomic<bool> aiming_active;
    std::atomic<bool> shooting_active;

private:
    bool is_open_;
    bool monitor_;
    std::thread monitor_thread_;
    std::atomic<bool> monitor_running_{ true };
    std::string ip_, port_, uuid_;
};
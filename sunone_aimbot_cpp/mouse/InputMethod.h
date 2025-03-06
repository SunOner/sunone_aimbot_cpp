#ifndef INPUT_METHOD_H
#define INPUT_METHOD_H

#include "SerialConnection.h"
#include "ghub.h"

// 마우스 입력 방식에 대한 인터페이스
class InputMethod
{
public:
    virtual ~InputMethod() = default;
    virtual void move(int x, int y) = 0;
    virtual void press() = 0;
    virtual void release() = 0;
    virtual bool isValid() const = 0;
};

// 시리얼 연결(Arduino)을 통한 마우스 입력 구현
class SerialInputMethod : public InputMethod
{
public:
    explicit SerialInputMethod(SerialConnection *serial) : serial_(serial) {}
    ~SerialInputMethod() override
    {
        // 참조만 유지하므로 소멸자에서 삭제하지 않음
    }

    void move(int x, int y) override
    {
        if (serial_ && serial_->isOpen())
        {
            serial_->move(x, y);
        }
    }

    void press() override
    {
        if (serial_ && serial_->isOpen())
        {
            serial_->press();
        }
    }

    void release() override
    {
        if (serial_ && serial_->isOpen())
        {
            serial_->release();
        }
    }

    bool isValid() const override
    {
        return serial_ && serial_->isOpen();
    }

private:
    SerialConnection *serial_;
};

// Logitech G HUB을 통한 마우스 입력 구현
class GHubInputMethod : public InputMethod
{
public:
    explicit GHubInputMethod(GhubMouse *ghub) : ghub_(ghub) {}
    ~GHubInputMethod() override
    {
        // 참조만 유지하므로 소멸자에서 삭제하지 않음
    }

    void move(int x, int y) override
    {
        if (ghub_)
        {
            ghub_->mouse_xy(x, y);
        }
    }

    void press() override
    {
        if (ghub_)
        {
            ghub_->mouse_down();
        }
    }

    void release() override
    {
        if (ghub_)
        {
            ghub_->mouse_up();
        }
    }

    bool isValid() const override
    {
        return ghub_ != nullptr;
    }

private:
    GhubMouse *ghub_;
};

// Windows API를 통한 기본 마우스 입력 구현
class Win32InputMethod : public InputMethod
{
public:
    Win32InputMethod() = default;

    void move(int x, int y) override
    {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = x;
        input.mi.dy = y;
        SendInput(1, &input, sizeof(INPUT));
    }

    void press() override
    {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
    }

    void release() override
    {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
    }

    bool isValid() const override
    {
        return true; // Win32 API는 항상 유효하다고 가정
    }
};

#endif // INPUT_METHOD_H
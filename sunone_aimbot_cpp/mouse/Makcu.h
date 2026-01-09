#ifndef MAKCU_CONNECTION_H
#define MAKCU_CONNECTION_H

#include <string>
#include <atomic>
#include <mutex>

#include "../modules/makcu/include/makcu.h"

class MakcuConnection
{
public:
    MakcuConnection(const std::string& port, unsigned int baud_rate);
    ~MakcuConnection();

    bool isOpen() const;

    void click(int button);
    void press(int button);
    void release(int button);
    void move(int x, int y);

    bool aiming_active;
    bool shooting_active;
    bool zooming_active;

private:
    void onButtonCallback(makcu::MouseButton button, bool pressed);

private:
    makcu::Device device_;
    std::atomic<bool> is_open_;
    std::mutex write_mutex_;
};

#endif // MAKCU_CONNECTION_H
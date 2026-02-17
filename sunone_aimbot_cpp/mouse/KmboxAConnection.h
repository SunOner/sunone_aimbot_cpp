#pragma once

#include <string>

class KmboxAConnection
{
public:
    explicit KmboxAConnection(const std::string& pidvid);
    ~KmboxAConnection();

    bool isOpen() const { return is_open_; }

    void move(int x, int y);
    void leftDown();
    void leftUp();
    void rightDown();
    void rightUp();
    void middleDown();
    void middleUp();
    void wheel(int delta);

private:
    static bool parsePidVid(const std::string& pidvid, unsigned short& pid, unsigned short& vid);

    bool is_open_;
};


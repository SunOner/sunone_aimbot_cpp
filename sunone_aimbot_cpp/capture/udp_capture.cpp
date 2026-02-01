#include "udp_capture.h"

#include <chrono>
#include <iostream>

UDPCapture::UDPCapture(int width, int height, const std::string& ip, int port)
    : width_(width)
    , height_(height)
    , ip_(ip)
    , port_(port)
    , socket_(INVALID_SOCKET)
    , is_connected_(false)
    , should_stop_(false)
    , received_frames_(0)
    , dropped_frames_(0)
{
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        std::cerr << "[UDPCapture] WSAStartup failed" << std::endl;
        return;
    }

    Initialize();
}

UDPCapture::~UDPCapture()
{
    Cleanup();
    WSACleanup();
}

bool UDPCapture::Initialize()
{
    if (socket_ != INVALID_SOCKET)
        closesocket(socket_);

    socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_ == INVALID_SOCKET)
    {
        std::cerr << "[UDPCapture] Failed to create socket: " << WSAGetLastError() << std::endl;
        return false;
    }

    int buffer_size = MAX_FRAME_SIZE;
    if (setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, (char*)&buffer_size, sizeof(buffer_size)) == SOCKET_ERROR)
    {
        std::cerr << "[UDPCapture] Failed to set receive buffer size: " << WSAGetLastError() << std::endl;
    }

    u_long mode = 1;
    if (ioctlsocket(socket_, FIONBIO, &mode) == SOCKET_ERROR)
    {
        std::cerr << "[UDPCapture] Failed to set non-blocking mode: " << WSAGetLastError() << std::endl;
    }

    memset(&server_addr_, 0, sizeof(server_addr_));
    server_addr_.sin_family = AF_INET;
    server_addr_.sin_port = htons(port_);
    if (inet_pton(AF_INET, ip_.c_str(), &server_addr_.sin_addr) <= 0)
    {
        std::cerr << "[UDPCapture] Invalid IP address: " << ip_ << std::endl;
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
        return false;
    }

    sockaddr_in local_addr;
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = INADDR_ANY;
    local_addr.sin_port = htons(port_);
    if (bind(socket_, (sockaddr*)&local_addr, sizeof(local_addr)) == SOCKET_ERROR)
    {
        std::cerr << "[UDPCapture] Failed to bind socket: " << WSAGetLastError() << std::endl;
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
        return false;
    }

    should_stop_ = false;
    is_connected_ = true;
    received_frames_ = 0;
    dropped_frames_ = 0;

    receive_thread_ = std::thread(&UDPCapture::ReceiveThread, this);

    std::cout << "[UDPCapture] Listening on UDP " << ip_ << ":" << port_ << std::endl;
    return true;
}

void UDPCapture::Cleanup()
{
    should_stop_ = true;
    is_connected_ = false;

    if (receive_thread_.joinable())
    {
        receive_thread_.join();
    }

    if (socket_ != INVALID_SOCKET)
    {
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
    }
}

void UDPCapture::SetUDPParams(const std::string& ip, int port)
{
    if (ip_ != ip || port_ != port)
    {
        ip_ = ip;
        port_ = port;

        if (is_connected_)
        {
            Cleanup();
            Initialize();
        }
    }
}

cv::Mat UDPCapture::GetNextFrameCpu()
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (frame_queue_.empty())
        return cv::Mat();

    cv::Mat frame = frame_queue_.front();
    frame_queue_.pop();
    return frame;
}

void UDPCapture::ReceiveThread()
{
    try
    {
        std::vector<uint8_t> buffer(MAX_FRAME_SIZE);
        std::vector<uint8_t> frame_data;

        while (!should_stop_)
        {
            sockaddr_in from_addr;
            int from_len = sizeof(from_addr);

            int bytes_received = recvfrom(
                socket_,
                (char*)buffer.data(),
                (int)buffer.size(),
                0,
                (sockaddr*)&from_addr,
                &from_len
            );

            if (bytes_received == SOCKET_ERROR)
            {
                int error = WSAGetLastError();
                if (error == WSAEWOULDBLOCK)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                std::cerr << "[UDPCapture] Receive error: " << error << std::endl;
                break;
            }

            if (bytes_received <= 0)
                continue;

            if (server_addr_.sin_addr.s_addr != 0 &&
                from_addr.sin_addr.s_addr != server_addr_.sin_addr.s_addr)
            {
                continue;
            }

            frame_data.insert(frame_data.end(), buffer.begin(), buffer.begin() + bytes_received);
            if (frame_data.size() > MAX_FRAME_SIZE * 2)
            {
                frame_data.clear();
                continue;
            }

            cv::Mat frame;
            if (ParseMJPEGFrame(frame_data, frame))
            {
                if (!frame.empty())
                {
                    if (frame.cols != width_ || frame.rows != height_)
                        cv::resize(frame, frame, cv::Size(width_, height_));

                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    while (frame_queue_.size() >= MAX_QUEUE_SIZE)
                    {
                        frame_queue_.pop();
                        dropped_frames_++;
                    }

                    frame_queue_.push(frame.clone());
                    received_frames_++;
                }

                frame_data.clear();
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[UDPCapture] Receive thread crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[UDPCapture] Receive thread crashed: unknown exception." << std::endl;
    }
}

bool UDPCapture::ParseMJPEGFrame(const std::vector<uint8_t>& data, cv::Mat& frame)
{
    if (data.size() < 4)
        return false;

    size_t start_pos = 0;
    bool found_start = false;
    for (size_t i = 0; i + 1 < data.size(); ++i)
    {
        if (data[i] == 0xFF && data[i + 1] == 0xD8)
        {
            start_pos = i;
            found_start = true;
            break;
        }
    }

    if (!found_start)
        return false;

    size_t end_pos = data.size();
    bool found_end = false;
    for (size_t i = start_pos + 2; i + 1 < data.size(); ++i)
    {
        if (data[i] == 0xFF && data[i + 1] == 0xD9)
        {
            end_pos = i + 2;
            found_end = true;
            break;
        }
    }

    if (!found_end)
        return false;

    std::vector<uint8_t> jpeg_data(data.begin() + start_pos, data.begin() + end_pos);
    try
    {
        frame = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
        return !frame.empty();
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[UDPCapture] JPEG decode error: " << e.what() << std::endl;
        return false;
    }
}

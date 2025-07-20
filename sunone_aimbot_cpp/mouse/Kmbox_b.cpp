#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cctype>

#include "Kmbox_b.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

const std::vector<uint8_t> Kmbox_b_Connection::BinaryPacketHeader = {0x6B, 0x6D, 0x2E}; // "km."

Kmbox_b_Connection::Kmbox_b_Connection(const std::string &port, unsigned int baud_rate)
    : is_open_(false), listening_(false), aiming_active(false), shooting_active(false), zooming_active(false), last_button_mask_(0x00), left_button_(false), right_button_(false)
{
    try
    {
        serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate); // Conectar directamente al baud rate deseado
        serial_.setTimeout(timeout);

        serial_.open();

        if (serial_.isOpen())
        {
            is_open_ = true;
            std::cout << "[Kmbox_b] Port " << port << " opened successfully at " << baud_rate << " baud." << std::endl;

            const std::vector<uint8_t> BAUD_CHANGE_COMMAND = {
            0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00};
            serial_.write(BAUD_CHANGE_COMMAND);
			serial_.flush();
			serial_.close(); // Cerrar antes de cambiar el baud rate
            serial_.setPort(port);
            serial_.setBaudrate(4000000);
            serial_.setTimeout(timeout);
            std::cout << "[Kmbox_b] Port " << port << " super opened successfully at " << 4000000 << " baud." << std::endl;

            serial_.open();

            // Verificar comunicación a este baud rate (opcional pero recomendado)
            // Esto asume que km.version() responde a baud_rate
            std::string version_cmd = "km.version()";
            // sendCommand(version_cmd); // sendCommand añade \r\n
            serial_.write(version_cmd + "\r\n"); // Envío directo para controlar el flush
            serial_.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));

            std::string version_response_str;
            if (serial_.available() > 0)
            {
                version_response_str = serial_.read(serial_.available());
                version_response_str.erase(std::remove_if(version_response_str.begin(), version_response_str.end(),
                                                          [](char c)
                                                          { return !isprint(c) && c != '\r' && c != '\n'; }),
                                           version_response_str.end());
                if (version_response_str.empty() || !(version_response_str.find("KMBOX") != std::string::npos || version_response_str.find("MAKCU") != std::string::npos || version_response_str[0] == 'v' || isdigit(version_response_str[0])))
                {
                    std::cerr << "[Kmbox_b] Warning: 'km.version()' response at " << baud_rate << " baud seems invalid or missing. Communication might be problematic." << std::endl;
                }
            }
            else
            {
                std::cout << "[Kmbox_b] No response to 'km.version()' at " << baud_rate << " baud (this might be OK for some devices, or indicate an issue)." << std::endl;
            }
            // Fin de la verificación de km.version()

            if (initializeButtonReporting())
            { // Se ejecutará al baud_rate
                std::cout << "[Kmbox_b] Button reporting initialization successful (at " << baud_rate << " baud)." << std::endl;
                startListening();
            }
            else
            {
                std::cerr << "[Kmbox_b] Failed to initialize button reporting (at " << baud_rate << " baud). Listener not started." << std::endl;
            }
        }
        else
        {
            is_open_ = false;
            std::cerr << "[Kmbox_b] Unable to connect to the port: " << port << " (serial_.open() failed at " << baud_rate << " baud)." << std::endl;
        }
    }
    catch (const serial::IOException &e)
    {
        std::cerr << "[Kmbox_b] Serial IO Error on connect: " << e.what() << std::endl;
        is_open_ = false;
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "[Kmbox_b] Invalid argument on connect: " << e.what() << std::endl;
        is_open_ = false;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Kmbox_b] Generic Error on connect: " << e.what() << std::endl;
        is_open_ = false;
    }
}

Kmbox_b_Connection::~Kmbox_b_Connection()
{
    listening_ = false;
    if (listening_thread_.joinable())
    {
        try
        {
            listening_thread_.join();
        }
        catch (const std::system_error &e)
        {
            std::cerr << "[Kmbox_b] Error joining listener thread: " << e.what() << std::endl;
        }
    }
    if (serial_.isOpen())
    {
        try
        {
            serial_.close();
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Kmbox_b] Exception closing serial port: " << e.what() << std::endl;
        }
    }
    is_open_ = false;
}

bool Kmbox_b_Connection::isOpen() const
{
    return is_open_.load();
}

bool Kmbox_b_Connection::isListening() const
{
    return listening_.load();
}

void Kmbox_b_Connection::write(const std::string &data)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_open_.load())
    {
        try
        {
            serial_.write(data);
            serial_.flush();
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Kmbox_b] Exception during write: " << e.what() << std::endl;
            is_open_ = false;
        }
    }
}

std::string Kmbox_b_Connection::read()
{
    if (!is_open_.load())
        return std::string();
    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
        std::cout << result << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Kmbox_b] Exception during read: " << e.what() << std::endl;
        is_open_ = false;
    }
    return result;
}

bool Kmbox_b_Connection::initializeButtonReporting()
{
    if (!is_open_.load())
        return false;

    std::string cmd_enable_events = "km.buttons(1)";
    // Obtener el baud rate actual para el log
    unsigned int current_br = 0;
    try
    {
        if (serial_.isOpen())
            current_br = serial_.getBaudrate();
    }
    catch (...)
    {
    }

    std::cout << "[Kmbox_b] Sending command to enable button events: \"" << cmd_enable_events
              << "\" (NOT expecting 'OK' response) at " << current_br << " baud." << std::endl;
    sendCommand(cmd_enable_events);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    try
    {
        if (serial_.available() > 0)
        {
            size_t available_count = serial_.available();
            std::string flushed_data = serial_.read(available_count);
            std::ostringstream oss_flush;
            for (char c : flushed_data)
            {
                if (isprint(c))
                    oss_flush << c;
                else
                    oss_flush << "[" << std::hex << std::setw(2) << std::setfill('0') << (0xFF & static_cast<unsigned char>(c)) << "]";
            }
            std::cout << "[Kmbox_b] Data flushed after init command (" << available_count << " bytes): " << oss_flush.str() << std::endl;
        }
        else
        {
            std::cout << "[Kmbox_b] No data in buffer to flush after init command." << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Kmbox_b] Exception during (optional) flush in initializeButtonReporting: " << e.what() << std::endl;
    }

    std::cout << "[Kmbox_b] Assumed button reporting initialization successful." << std::endl;
    return true;
}

// --- Métodos de acción del ratón (sin cambios respecto a la última versión) ---
void Kmbox_b_Connection::move(int x, int y)
{
    if (!is_open_.load())
        return;
    sendCommand("km.move(" + std::to_string(x) + "," + std::to_string(y) + ")");
}

void Kmbox_b_Connection::click(int button = 0)
{
    std::string cmd = "km.click(" + std::to_string(button) + ")\r\n";
    sendCommand(cmd);
}

void Kmbox_b_Connection::press(int button)
{
    std::string cmd;
    if (button == 0)
        cmd = "km.left(1)";
    else if (button == 1)
        cmd = "km.right(1)";
    else if (button == 2)
        cmd = "km.middle(1)";
    else
    {
        std::cerr << "[Kmbox_b] Press: Unknown button_id: " << button << std::endl;
        return;
    }
    sendCommand(cmd);
}

void Kmbox_b_Connection::release(int button)
{
    std::string cmd;
    if (button == 0)
        cmd = "km.left(0)";
    else if (button == 1)
        cmd = "km.right(0)";
    else if (button == 2)
        cmd = "km.middle(0)";
    else
    {
        std::cerr << "[Kmbox_b] Release: Unknown button_id: " << button << std::endl;
        return;
    }
    sendCommand(cmd);
}

void Kmbox_b_Connection::start_boot()
{
    write("\x03\x03");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write("exec(open('boot.py').read(),globals())\r\n");
}

void Kmbox_b_Connection::reboot()
{
    write("\x03\x03");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write("km.reboot()");
}

void Kmbox_b_Connection::send_stop()
{
    write("\x03\x03");
}

void Kmbox_b_Connection::sendCommand(const std::string &command)
{
    write(command + "\r\n");
}

std::vector<int> Kmbox_b_Connection::splitValue(int value)
{
    std::vector<int> values;
    return values;
}

void Kmbox_b_Connection::startListening()
{
    if (!is_open_.load())
    {
        std::cerr << "[Kmbox_b] Cannot start listener, port not open." << std::endl;
        return;
    }
    if (listening_.load())
    {
        std::cout << "[Kmbox_b] Listener already running." << std::endl;
        return;
    }
    listening_ = true;
    if (listening_thread_.joinable())
    {
        try
        {
            listening_thread_.join();
        }
        catch (...)
        {
        }
    }
    listening_thread_ = std::thread(&Kmbox_b_Connection::listeningThreadFunc, this);
}

void Kmbox_b_Connection::listeningThreadFunc()
{
    std::vector<uint8_t> header_search_buffer;
    header_search_buffer.reserve(BinaryPacketHeader.size());
    enum class ParserState
    {
        SearchingHeader,
        ReadingMask,
        ReadingTailCR,
        ReadingTailLF
    };
    ParserState currentState = ParserState::SearchingHeader;
    uint8_t received_mask_byte = 0;

    unsigned int loop_counter = 0;
    const unsigned int no_data_log_interval = 400; // Log "no data" cada ~400ms si el sleep es de 1ms

    unsigned int current_br_listener = 0;
    try
    {
        if (serial_.isOpen())
            current_br_listener = serial_.getBaudrate();
    }
    catch (...)
    {
    }

    while (listening_.load() && is_open_.load())
    {
        loop_counter++;
        try
        {
            if (!serial_.isOpen())
            {
                if (loop_counter % no_data_log_interval == 1)
                    std::cout << "[Kmbox_b] BIN Listener: Port seems closed." << std::endl;
                is_open_ = false;
                break;
            }

            if (serial_.available() > 0)
            {
                uint8_t byte_read;
                if (serial_.read(&byte_read, 1) == 0)
                    continue;

                if (currentState == ParserState::SearchingHeader && byte_read == '>')
                {
                    try
                    {
                        if (serial_.available() >= 3)
                        {
                            uint8_t prompt_buf[3];
                            serial_.read(prompt_buf, 3);
                            if (prompt_buf[0] == '>' && prompt_buf[1] == '>' && prompt_buf[2] == ' ')
                            {
                                continue;
                            }
                        }
                    }
                    catch (...)
                    {
                    }
                }

                switch (currentState)
                {
                case ParserState::SearchingHeader:
                    if (byte_read == BinaryPacketHeader[header_search_buffer.size()])
                    {
                        header_search_buffer.push_back(byte_read);
                        if (header_search_buffer.size() == BinaryPacketHeader.size())
                        {
                            currentState = ParserState::ReadingMask;
                            header_search_buffer.clear();
                        }
                    }
                    else
                    {
                        header_search_buffer.clear();
                        if (byte_read == BinaryPacketHeader[0])
                        {
                            header_search_buffer.push_back(byte_read);
                        }
                    }
                    break;
                case ParserState::ReadingMask:
                    received_mask_byte = byte_read;
                    currentState = ParserState::ReadingTailCR;
                    break;
                case ParserState::ReadingTailCR:
                    if (byte_read == 0x0D)
                    {
                        currentState = ParserState::ReadingTailLF;
                    }
                    else
                    {
                        currentState = ParserState::SearchingHeader;
                        header_search_buffer.clear();
                        if (byte_read == BinaryPacketHeader[0])
                            header_search_buffer.push_back(byte_read);
                    }
                    break;
                case ParserState::ReadingTailLF:
                    if (byte_read == 0x0A)
                    {
                        processButtonMask(received_mask_byte);
                    }
                    currentState = ParserState::SearchingHeader;
                    header_search_buffer.clear();
                    break;
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        catch (const serial::SerialException &e)
        {
            is_open_ = false;
            break;
        }
        catch (const serial::IOException &e)
        {
            is_open_ = false;
            break;
        }
        catch (const std::exception &e)
        {
            currentState = ParserState::SearchingHeader;
            header_search_buffer.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    listening_ = false;
    std::cout << "[Kmbox_b] BIN Listener thread has stopped." << std::endl;
}

void Kmbox_b_Connection::processButtonMask(uint8_t current_mask)
{
    if (current_mask > 0b00011111)
    {
        return;
    }
    if (current_mask != last_button_mask_)
    {
        uint8_t changed_bits = current_mask ^ last_button_mask_;

        if ((changed_bits & (1 << 0)) != 0)
        {
            bool is_pressed = (current_mask & (1 << 0)) != 0;
            shooting.store(is_pressed);
            left_button_ = is_pressed;
        }
        if ((changed_bits & (1 << 1)) != 0)
        {
            bool is_pressed = (current_mask & (1 << 1)) != 0;
            right_button_ = is_pressed;
            zooming.store(is_pressed);
        }
        if ((changed_bits & (1 << 2)) != 0)
        {
            bool is_pressed = (current_mask & (1 << 2)) != 0;
            zooming_active = is_pressed;
        }
        last_button_mask_ = current_mask;
    }
}

int Kmbox_b_Connection::monitorMouseLeft() const
{
    if (!isOpen())
        return -1; // Opcional: chequear si está abierto
    return left_button_.load() ? 1 : 0;
}

int Kmbox_b_Connection::monitorMouseRight() const
{
    if (!isOpen())
        return -1;
    return right_button_.load() ? 1 : 0;
}
// Kmbox_b.cpp
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
#include <cstring> // Para memcpy
#include <immintrin.h> // Para _mm_pause

#include "Kmbox_b.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

// --- Constantes para comandos de texto. Cero alocaciones en tiempo de ejecución. ---
const char CMD_LEFT_PRESS[] = "km.left(1)";
const char CMD_LEFT_RELEASE[] = "km.left(0)";
const char CMD_RIGHT_PRESS[] = "km.right(1)";
const char CMD_RIGHT_RELEASE[] = "km.right(0)";
const char CMD_MIDDLE_PRESS[] = "km.middle(1)";
const char CMD_MIDDLE_RELEASE[] = "km.middle(0)";
const char CMD_CLICK_LEFT[] = "km.click(0)";
const char CMD_CLICK_RIGHT[] = "km.click(1)";
const char CMD_CLICK_MIDDLE[] = "km.click(2)";

const size_t COMMAND_BUFFER_SIZE = 128; 

const std::vector<uint8_t> Kmbox_b_Connection::BinaryPacketHeader = {0x6B, 0x6D, 0x2E}; // "km."

// Implementación de itoa de alto rendimiento para evitar snprintf/stringstream
char* Kmbox_b_Connection::fast_itoa(int value, char* buffer)
{
    if (value == 0) {
        *buffer++ = '0';
        return buffer;
    }

    char* start = buffer;
    if (value < 0) {
        *start++ = '-';
        value = -value;
    }

    char* p = start;
    while (value > 0) {
        *p++ = (value % 10) + '0';
        value /= 10;
    }
    char* end = p;
    *p-- = '\0'; // Null terminator not strictly needed for our use case, but good practice

    // Reverse the string
    while (start < p) {
        char temp = *start;
        *start++ = *p;
        *p-- = temp;
    }

    return end; // Return pointer to the end of the number
}


Kmbox_b_Connection::Kmbox_b_Connection(const std::string &port, unsigned int baud_rate)
    : is_open_(false), listening_(false), aiming_active(false), shooting_active(false), zooming_active(false), last_button_mask_(0x00), left_button_(false), right_button_(false)
{
    // El constructor no es parte del hot-path, se mantiene la lógica original por robustez en la inicialización.
    try
    {
        serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate); 
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
            serial_.close();
            serial_.setPort(port);
            serial_.setBaudrate(4000000);
            serial_.setTimeout(timeout);
            std::cout << "[Kmbox_b] Port " << port << " super opened successfully at " << 4000000 << " baud." << std::endl;

            serial_.open();
            
            // Verificación de comunicación (se mantiene por robustez)
            const char version_cmd[] = "km.version()";
            sendCommand(version_cmd, sizeof(version_cmd) - 1);
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
                    std::cerr << "[Kmbox_b] Warning: 'km.version()' response at " << 4000000 << " baud seems invalid or missing. Communication might be problematic." << std::endl;
                }
            }
            else
            {
                std::cout << "[Kmbox_b] No response to 'km.version()' at " << 4000000 << " baud (this might be OK for some devices, or indicate an issue)." << std::endl;
            }

            if (initializeButtonReporting())
            {
                std::cout << "[Kmbox_b] Button reporting initialization successful." << std::endl;
                startListening();
            }
            else
            {
                std::cerr << "[Kmbox_b] Failed to initialize button reporting. Listener not started." << std::endl;
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
    return is_open_.load(std::memory_order_relaxed);
}

bool Kmbox_b_Connection::isListening() const
{
    return listening_.load(std::memory_order_relaxed);
}

void Kmbox_b_Connection::write(const std::string &data)
{
    // Esta función no está en el hot-path, se mantiene por compatibilidad.
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_open_.load(std::memory_order_relaxed)) {
        try {
            serial_.write(data);
            serial_.flush();
        } catch (const std::exception& e) {
            std::cerr << "[Kmbox_b] Exception during write: " << e.what() << std::endl;
            is_open_ = false;
        }
    }
}

std::string Kmbox_b_Connection::read()
{
    // No está en el hot-path
    if (!is_open_.load(std::memory_order_relaxed))
        return {};
    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
        // std::cout << result << std::endl; // Evitar I/O en funciones de bajo nivel
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
    if (!is_open_.load(std::memory_order_relaxed))
        return false;

    const char cmd_enable_events[] = "km.buttons(1)";
    sendCommand(cmd_enable_events, sizeof(cmd_enable_events) - 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Sleep necesario para que el dispositivo procese el comando

    try
    {
        if (serial_.available() > 0)
        {
            serial_.read(serial_.available()); // Flush buffer
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Kmbox_b] Exception during flush in initializeButtonReporting: " << e.what() << std::endl;
    }

    return true;
}

void Kmbox_b_Connection::move(int x, int y)
{
    if (!is_open_.load(std::memory_order_relaxed)) return;

    // *** OPTIMIZACIÓN: Construcción de comando sin alocación y con itoa rápido ***
    char* p = command_buffer_;
    memcpy(p, "km.move(", 8); 
    p += 8;
    p = fast_itoa(x, p);
    *p++ = ',';
    p = fast_itoa(y, p);
    *p++ = ')';

    sendCommand(command_buffer_, p - command_buffer_);
}

void Kmbox_b_Connection::click(int button)
{
    // *** OPTIMIZACIÓN: Usar comandos precalculados ***
    switch(button) {
        case 0: sendCommand(CMD_CLICK_LEFT, sizeof(CMD_CLICK_LEFT) - 1); break;
        case 1: sendCommand(CMD_CLICK_RIGHT, sizeof(CMD_CLICK_RIGHT) - 1); break;
        case 2: sendCommand(CMD_CLICK_MIDDLE, sizeof(CMD_CLICK_MIDDLE) - 1); break;
    }
}

void Kmbox_b_Connection::press(int button)
{
    // *** OPTIMIZACIÓN: Usar comandos precalculados ***
    switch(button) {
        case 0: sendCommand(CMD_LEFT_PRESS, sizeof(CMD_LEFT_PRESS) - 1); break;
        case 1: sendCommand(CMD_RIGHT_PRESS, sizeof(CMD_RIGHT_PRESS) - 1); break;
        case 2: sendCommand(CMD_MIDDLE_PRESS, sizeof(CMD_MIDDLE_PRESS) - 1); break;
        default:
            std::cerr << "[Kmbox_b] Press: Unknown button_id: " << button << std::endl;
    }
}

void Kmbox_b_Connection::release(int button)
{
    // *** OPTIMIZACIÓN: Usar comandos precalculados ***
    switch(button) {
        case 0: sendCommand(CMD_LEFT_RELEASE, sizeof(CMD_LEFT_RELEASE) - 1); break;
        case 1: sendCommand(CMD_RIGHT_RELEASE, sizeof(CMD_RIGHT_RELEASE) - 1); break;
        case 2: sendCommand(CMD_MIDDLE_RELEASE, sizeof(CMD_MIDDLE_RELEASE) - 1); break;
        default:
            std::cerr << "[Kmbox_b] Release: Unknown button_id: " << button << std::endl;
    }
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

void Kmbox_b_Connection::sendCommand(const char* command, size_t length)
{
    if (!is_open_.load(std::memory_order_relaxed) || length + 2 > COMMAND_BUFFER_SIZE)
    {
        return;
    }

    // Usar un buffer temporal en el stack para no interferir con command_buffer_ si es el mismo
    char write_buf[COMMAND_BUFFER_SIZE];
    memcpy(write_buf, command, length);
    write_buf[length] = '\r';
    write_buf[length + 1] = '\n';

    std::lock_guard<std::mutex> lock(write_mutex_); // El mutex se mantiene por seguridad en escrituras concurrentes (aunque poco probable)
    try
    {
        serial_.write(reinterpret_cast<const uint8_t *>(write_buf), length + 2);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Kmbox_b] Exception during sendCommand: " << e.what() << std::endl;
        is_open_ = false;
    }
}

void Kmbox_b_Connection::startListening()
{
    if (!is_open_.load(std::memory_order_relaxed))
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
        try { listening_thread_.join(); } catch (...) {}
    }
    listening_thread_ = std::thread(&Kmbox_b_Connection::listeningThreadFunc, this);
}

void Kmbox_b_Connection::listeningThreadFunc()
{
    std::vector<uint8_t> header_search_buffer;
    header_search_buffer.reserve(BinaryPacketHeader.size());
    enum class ParserState { SearchingHeader, ReadingMask, ReadingTailCR, ReadingTailLF };
    ParserState currentState = ParserState::SearchingHeader;
    uint8_t received_mask_byte = 0;

    while (listening_.load(std::memory_order_relaxed))
    {
        try
        {
            if (!serial_.isOpen()) {
                is_open_ = false;
                break;
            }

            if (serial_.available() > 0)
            {
                uint8_t byte_read;
                if (serial_.read(&byte_read, 1) == 0) continue;

                // (La lógica del parser se mantiene, es eficiente)
                switch (currentState)
                {
                case ParserState::SearchingHeader:
                    if (byte_read == BinaryPacketHeader[header_search_buffer.size()]) {
                        header_search_buffer.push_back(byte_read);
                        if (header_search_buffer.size() == BinaryPacketHeader.size()) {
                            currentState = ParserState::ReadingMask;
                            header_search_buffer.clear();
                        }
                    } else {
                        header_search_buffer.clear();
                        if (byte_read == BinaryPacketHeader[0]) {
                            header_search_buffer.push_back(byte_read);
                        }
                    }
                    break;
                case ParserState::ReadingMask:
                    received_mask_byte = byte_read;
                    currentState = ParserState::ReadingTailCR;
                    break;
                case ParserState::ReadingTailCR:
                    if (byte_read == 0x0D) {
                        currentState = ParserState::ReadingTailLF;
                    } else {
                        currentState = ParserState::SearchingHeader;
                        header_search_buffer.clear();
                        if (byte_read == BinaryPacketHeader[0]) header_search_buffer.push_back(byte_read);
                    }
                    break;
                case ParserState::ReadingTailLF:
                    if (byte_read == 0x0A) {
                        processButtonMask(received_mask_byte);
                    }
                    currentState = ParserState::SearchingHeader;
                    header_search_buffer.clear();
                    break;
                }
            }
            else
            {
                // *** OPTIMIZACIÓN: Reemplazar sleep con busy-wait de baja sobrecarga ***
                _mm_pause(); // Le da una pista a la CPU de que estamos en un spin-loop. Mucho mejor que un bucle vacío.
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
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep en caso de error inesperado para no quemar la CPU
        }
    }
    listening_ = false;
    std::cout << "[Kmbox_b] BIN Listener thread has stopped." << std::endl;
}

void Kmbox_b_Connection::processButtonMask(uint8_t current_mask)
{
    if (current_mask > 0b00011111) return;
    
    // Usar memory_order_relaxed es seguro aquí porque los atomics solo necesitan ser consistentes
    // dentro de este hilo y leídos por otros, no requieren una sincronización estricta con otras variables.
    if (current_mask != last_button_mask_.load(std::memory_order_relaxed))
    {
        uint8_t changed_bits = current_mask ^ last_button_mask_.load(std::memory_order_relaxed);

        if ((changed_bits & 1)) { // Bit 0
            bool is_pressed = (current_mask & 1);
            shooting.store(is_pressed, std::memory_order_relaxed);
            left_button_.store(is_pressed, std::memory_order_relaxed);
        }
        if ((changed_bits & 2)) { // Bit 1
            bool is_pressed = (current_mask & 2);
            right_button_.store(is_pressed, std::memory_order_relaxed);
            zooming.store(is_pressed, std::memory_order_relaxed);
        }
        if ((changed_bits & 4)) { // Bit 2
            bool is_pressed = (current_mask & 4);
            zooming_active.store(is_pressed, std::memory_order_relaxed);
        }
        last_button_mask_.store(current_mask, std::memory_order_relaxed);
    }
}

int Kmbox_b_Connection::monitorMouseLeft() const
{
    return left_button_.load(std::memory_order_relaxed) ? 1 : 0;
}

int Kmbox_b_Connection::monitorMouseRight() const
{
    return right_button_.load(std::memory_order_relaxed) ? 1 : 0;
}
// Kmbox_b.cpp
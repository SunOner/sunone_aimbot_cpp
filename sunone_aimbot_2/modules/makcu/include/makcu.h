#pragma once

// Export macros for shared library support across platforms
#ifdef _WIN32
    #ifdef MAKCU_EXPORTS
        #define MAKCU_API __declspec(dllexport)
    #elif defined(MAKCU_SHARED)
        #define MAKCU_API __declspec(dllimport)
    #else
        #define MAKCU_API
    #endif
#else
    #ifdef __GNUC__
        #define MAKCU_API __attribute__((visibility("default")))
    #else
        #define MAKCU_API
    #endif
#endif

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <exception>
#include <unordered_map>
#include <atomic>
#include <future>
#include <chrono>

namespace makcu {

    // Forward declaration
    class SerialPort;

    // Enums
    enum class MouseButton : uint8_t {
        LEFT = 0,
        RIGHT = 1,
        MIDDLE = 2,
        SIDE1 = 3,
        SIDE2 = 4
    };

    enum class ConnectionStatus {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        CONNECTION_ERROR,
    };

    // Simple structs
    struct DeviceInfo {
        std::string port;
        std::string description;
        uint16_t vid;
        uint16_t pid;
        bool isConnected;
    };

    struct MouseButtonStates {
        bool left;
        bool right;
        bool middle;
        bool side1;
        bool side2;

        MouseButtonStates() : left(false), right(false), middle(false), side1(false), side2(false) {}

        bool operator[](MouseButton button) const {
            switch (button) {
            case MouseButton::LEFT: return left;
            case MouseButton::RIGHT: return right;
            case MouseButton::MIDDLE: return middle;
            case MouseButton::SIDE1: return side1;
            case MouseButton::SIDE2: return side2;
            }
            return false;
        }

        void set(MouseButton button, bool state) {
            switch (button) {
            case MouseButton::LEFT: left = state; break;
            case MouseButton::RIGHT: right = state; break;
            case MouseButton::MIDDLE: middle = state; break;
            case MouseButton::SIDE1: side1 = state; break;
            case MouseButton::SIDE2: side2 = state; break;
            }
        }
    };

    // Exception classes
    class MAKCU_API MakcuException : public std::exception {
    public:
        explicit MakcuException(const std::string& message) : m_message(message) {}
        const char* what() const noexcept override { return m_message.c_str(); }
    private:
        std::string m_message;
    };

    class MAKCU_API ConnectionException : public MakcuException {
    public:
        explicit ConnectionException(const std::string& message)
            : MakcuException("Connection error: " + message) {
        }
    };

    class MAKCU_API CommandException : public MakcuException {
    public:
        explicit CommandException(const std::string& message)
            : MakcuException("Command error: " + message) {
        }
    };

    class MAKCU_API TimeoutException : public MakcuException {
    public:
        explicit TimeoutException(const std::string& message)
            : MakcuException("Timeout error: " + message) {
        }
    };

    // Main Device class - High Performance MAKCU Mouse Controller
    class MAKCU_API Device {
    public:
        // Callback types
        using MouseButtonCallback = std::function<void(MouseButton, bool)>;
        using ConnectionCallback = std::function<void(bool)>;

        // Constructor and destructor
        Device();
        ~Device();

        // Static methods
        static std::vector<DeviceInfo> findDevices();
        static std::string findFirstDevice();

        // Connection with async support
        bool connect(const std::string& port = "");
        void disconnect();
        bool isConnected() const;
        ConnectionStatus getStatus() const;

        // Async connection methods
        std::future<bool> connectAsync(const std::string& port = "");

        // Device info
        DeviceInfo getDeviceInfo() const;
        std::string getVersion() const;

        // High-performance mouse button control (fire-and-forget)
        bool mouseDown(MouseButton button);
        bool mouseUp(MouseButton button);
        bool click(MouseButton button);  // Combined press+release


        // Mouse button state queries (with caching)
        bool mouseButtonState(MouseButton button);

        // High-performance movement (fire-and-forget for gaming)
        bool mouseMove(int32_t x, int32_t y);
        bool mouseMoveSmooth(int32_t x, int32_t y, uint32_t segments);
        bool mouseMoveBezier(int32_t x, int32_t y, uint32_t segments,
            int32_t ctrl_x, int32_t ctrl_y);

        // High-performance drag operations
        bool mouseDrag(MouseButton button, int32_t x, int32_t y);
        bool mouseDragSmooth(MouseButton button, int32_t x, int32_t y, uint32_t segments = 10);
        bool mouseDragBezier(MouseButton button, int32_t x, int32_t y, uint32_t segments = 20,
            int32_t ctrl_x = 0, int32_t ctrl_y = 0);


        // Mouse wheel
        bool mouseWheel(int32_t delta);

        // Mouse locking with state caching
        bool lockMouseX(bool lock = true);
        bool lockMouseY(bool lock = true);
        bool lockMouseLeft(bool lock = true);
        bool lockMouseMiddle(bool lock = true);
        bool lockMouseRight(bool lock = true);
        bool lockMouseSide1(bool lock = true);
        bool lockMouseSide2(bool lock = true);

        // Fast lock state queries (cached)
        bool isMouseXLocked() const;
        bool isMouseYLocked() const;
        bool isMouseLeftLocked() const;
        bool isMouseMiddleLocked() const;
        bool isMouseRightLocked() const;
        bool isMouseSide1Locked() const;
        bool isMouseSide2Locked() const;

        // Batch lock state query
        std::unordered_map<std::string, bool> getAllLockStates() const;

        // Mouse input catching
        uint8_t catchMouseLeft();
        uint8_t catchMouseMiddle();
        uint8_t catchMouseRight();
        uint8_t catchMouseSide1();
        uint8_t catchMouseSide2();

        // Button monitoring with optimized processing
        bool enableButtonMonitoring(bool enable = true);
        bool isButtonMonitoringEnabled() const;
        uint8_t getButtonMask() const;

        // Serial spoofing
        std::string getMouseSerial();
        bool setMouseSerial(const std::string& serial);
        bool resetMouseSerial();


        // Device control
        /**
         * @brief Changes the MAKCU device baud rate using the proper protocol
         * 
         * Sends the MAKCU-specific baud rate change command and reconnects at the new rate.
         * Valid range: 115200 - 4000000 bps. Note: This change is NOT permanent and will
         * reset to 115200 when the MAKCU device is power cycled.
         * 
         * @param baudRate Target baud rate (will be clamped to valid range)
         * @param validateCommunication Whether to test communication after baud rate change (default: true)
         * @return true if baud rate was successfully changed and verified, false otherwise
         */
        bool setBaudRate(uint32_t baudRate, bool validateCommunication = true);

        // Callbacks
        void setMouseButtonCallback(MouseButtonCallback callback);
        void setConnectionCallback(ConnectionCallback callback);

        // High-level automation
        bool clickSequence(const std::vector<MouseButton>& buttons,
            std::chrono::milliseconds delay = std::chrono::milliseconds(50));
        bool movePattern(const std::vector<std::pair<int32_t, int32_t>>& points,
            bool smooth = true, uint32_t segments = 10);

        // Performance utilities
        void enableHighPerformanceMode(bool enable = true);
        bool isHighPerformanceModeEnabled() const;

        // Command batching for maximum performance
        class MAKCU_API BatchCommandBuilder {
        public:
            BatchCommandBuilder& move(int32_t x, int32_t y);
            BatchCommandBuilder& moveSmooth(int32_t x, int32_t y, uint32_t segments = 10);
            BatchCommandBuilder& moveBezier(int32_t x, int32_t y, uint32_t segments = 20,
                int32_t ctrl_x = 0, int32_t ctrl_y = 0);
            BatchCommandBuilder& click(MouseButton button);
            BatchCommandBuilder& press(MouseButton button);
            BatchCommandBuilder& release(MouseButton button);
            BatchCommandBuilder& scroll(int32_t delta);
            BatchCommandBuilder& drag(MouseButton button, int32_t x, int32_t y);
            BatchCommandBuilder& dragSmooth(MouseButton button, int32_t x, int32_t y, uint32_t segments = 10);
            BatchCommandBuilder& dragBezier(MouseButton button, int32_t x, int32_t y, uint32_t segments = 20,
                int32_t ctrl_x = 0, int32_t ctrl_y = 0);
            bool execute();

        private:
            friend class Device;
            BatchCommandBuilder(Device* device) : m_device(device) {}
            Device* m_device;
            std::vector<std::string> m_commands;
        };

        BatchCommandBuilder createBatch();

        // Legacy raw command interface (not recommended for performance)
        bool sendRawCommand(const std::string& command) const;
        std::string receiveRawResponse() const;

    private:
        // Implementation details with caching and optimization
        class Impl;
        std::unique_ptr<Impl> m_impl;

        // Disable copy
        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;
    };

    // Utility functions
    MAKCU_API std::string mouseButtonToString(MouseButton button);
    MAKCU_API MouseButton stringToMouseButton(const std::string& buttonName);

    // Performance profiling utilities
    class MAKCU_API PerformanceProfiler {
    private:
        static std::atomic<bool> s_enabled;
        static std::mutex s_mutex;
        static std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> s_stats;

    public:
        static void enableProfiling(bool enable = true) {
            s_enabled.store(enable);
        }

        static void logCommandTiming(const std::string& command, std::chrono::microseconds duration) {
            if (!s_enabled.load()) return;

            std::lock_guard<std::mutex> lock(s_mutex);
            auto& [count, total_us] = s_stats[command];
            count++;
            total_us += duration.count();
        }

        static std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> getStats() {
            std::lock_guard<std::mutex> lock(s_mutex);
            return s_stats;
        }

        static void resetStats() {
            std::lock_guard<std::mutex> lock(s_mutex);
            s_stats.clear();
        }
    };

} // namespace makcu
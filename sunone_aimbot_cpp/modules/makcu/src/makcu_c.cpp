#include "../modules/makcu/include/makcu_c.h"
#include "makcu.h"
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <functional>

extern "C" {

// Internal wrapper structures
struct makcu_device {
    std::unique_ptr<makcu::Device> cpp_device;
    makcu_mouse_button_callback_t mouse_callback;
    void* mouse_callback_user_data;
    makcu_connection_callback_t connection_callback;
    void* connection_callback_user_data;
    
    makcu_device() : 
        cpp_device(std::make_unique<makcu::Device>()),
        mouse_callback(nullptr), 
        mouse_callback_user_data(nullptr),
        connection_callback(nullptr),
        connection_callback_user_data(nullptr) {}
};

struct makcu_batch_builder {
    std::unique_ptr<makcu::Device::BatchCommandBuilder> cpp_batch;
    
    makcu_batch_builder(makcu::Device::BatchCommandBuilder&& batch) :
        cpp_batch(std::make_unique<makcu::Device::BatchCommandBuilder>(std::move(batch))) {}
};

// Helper functions
static makcu::MouseButton convert_mouse_button(makcu_mouse_button_t button) {
    switch (button) {
        case MAKCU_MOUSE_LEFT: return makcu::MouseButton::LEFT;
        case MAKCU_MOUSE_RIGHT: return makcu::MouseButton::RIGHT;
        case MAKCU_MOUSE_MIDDLE: return makcu::MouseButton::MIDDLE;
        case MAKCU_MOUSE_SIDE1: return makcu::MouseButton::SIDE1;
        case MAKCU_MOUSE_SIDE2: return makcu::MouseButton::SIDE2;
    }
    return makcu::MouseButton::LEFT;
}

static makcu_mouse_button_t convert_mouse_button_to_c(makcu::MouseButton button) {
    switch (button) {
        case makcu::MouseButton::LEFT: return MAKCU_MOUSE_LEFT;
        case makcu::MouseButton::RIGHT: return MAKCU_MOUSE_RIGHT;
        case makcu::MouseButton::MIDDLE: return MAKCU_MOUSE_MIDDLE;
        case makcu::MouseButton::SIDE1: return MAKCU_MOUSE_SIDE1;
        case makcu::MouseButton::SIDE2: return MAKCU_MOUSE_SIDE2;
    }
    return MAKCU_MOUSE_LEFT;
}

static makcu_connection_status_t convert_connection_status(makcu::ConnectionStatus status) {
    switch (status) {
        case makcu::ConnectionStatus::DISCONNECTED: return MAKCU_STATUS_DISCONNECTED;
        case makcu::ConnectionStatus::CONNECTING: return MAKCU_STATUS_CONNECTING;
        case makcu::ConnectionStatus::CONNECTED: return MAKCU_STATUS_CONNECTED;
        case makcu::ConnectionStatus::CONNECTION_ERROR: return MAKCU_STATUS_CONNECTION_ERROR;
    }
    return MAKCU_STATUS_DISCONNECTED;
}

static makcu_error_t handle_exception() {
    try {
        throw;
    } catch (const makcu::ConnectionException&) {
        return MAKCU_ERROR_CONNECTION_FAILED;
    } catch (const makcu::CommandException&) {
        return MAKCU_ERROR_COMMAND_FAILED;
    } catch (const makcu::TimeoutException&) {
        return MAKCU_ERROR_TIMEOUT;
    } catch (const makcu::MakcuException&) {
        return MAKCU_ERROR_COMMAND_FAILED;
    } catch (const std::bad_alloc&) {
        return MAKCU_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return MAKCU_ERROR_COMMAND_FAILED;
    }
}

static void safe_copy_string(char* dest, size_t dest_size, const std::string& src)
{
    if (!dest || dest_size == 0) return;
    strncpy_s(dest, dest_size, src.c_str(), _TRUNCATE);
}

// Error handling
const char* makcu_error_string(makcu_error_t error) {
    switch (error) {
        case MAKCU_SUCCESS: return "Success";
        case MAKCU_ERROR_INVALID_DEVICE: return "Invalid device";
        case MAKCU_ERROR_CONNECTION_FAILED: return "Connection failed";
        case MAKCU_ERROR_COMMAND_FAILED: return "Command failed";
        case MAKCU_ERROR_TIMEOUT: return "Timeout";
        case MAKCU_ERROR_INVALID_PARAMETER: return "Invalid parameter";
        case MAKCU_ERROR_OUT_OF_MEMORY: return "Out of memory";
    }
    return "Unknown error";
}

// Device management
makcu_device_t* makcu_device_create(void) {
    try {
        return new makcu_device();
    } catch (...) {
        return nullptr;
    }
}

void makcu_device_destroy(makcu_device_t* device) {
    delete device;
}

// Static device discovery
int makcu_find_devices(makcu_device_info_t* devices, int max_devices) {
    if (!devices || max_devices <= 0) {
        return 0;
    }
    
    try {
        auto cpp_devices = makcu::Device::findDevices();
        int count = std::min(max_devices, static_cast<int>(cpp_devices.size()));
        
        for (int i = 0; i < count; i++) {
            safe_copy_string(devices[i].port, sizeof(devices[i].port), cpp_devices[i].port);
            safe_copy_string(devices[i].description, sizeof(devices[i].description), cpp_devices[i].description);
            devices[i].vid = cpp_devices[i].vid;
            devices[i].pid = cpp_devices[i].pid;
            devices[i].is_connected = cpp_devices[i].isConnected;
        }
        
        return count;
    } catch (...) {
        return 0;
    }
}

makcu_error_t makcu_find_first_device(char* port, size_t port_size) {
    if (!port || port_size == 0) {
        return MAKCU_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto first_port = makcu::Device::findFirstDevice();
        safe_copy_string(port, port_size, first_port);
        return MAKCU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

// Connection management
makcu_error_t makcu_connect(makcu_device_t* device, const char* port) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        std::string port_str = port ? port : "";
        bool success = device->cpp_device->connect(port_str);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_CONNECTION_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

void makcu_disconnect(makcu_device_t* device) {
    if (device) {
        try {
            device->cpp_device->disconnect();
        } catch (...) {
            // Ignore exceptions on disconnect
        }
    }
}

bool makcu_is_connected(makcu_device_t* device) {
    if (!device) {
        return false;
    }
    
    try {
        return device->cpp_device->isConnected();
    } catch (...) {
        return false;
    }
}

makcu_connection_status_t makcu_get_status(makcu_device_t* device) {
    if (!device) {
        return MAKCU_STATUS_DISCONNECTED;
    }
    
    try {
        return convert_connection_status(device->cpp_device->getStatus());
    } catch (...) {
        return MAKCU_STATUS_CONNECTION_ERROR;
    }
}

// Device information
makcu_error_t makcu_get_device_info(makcu_device_t* device, makcu_device_info_t* info) {
    if (!device || !info) {
        return MAKCU_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto cpp_info = device->cpp_device->getDeviceInfo();
        safe_copy_string(info->port, sizeof(info->port), cpp_info.port);
        safe_copy_string(info->description, sizeof(info->description), cpp_info.description);
        info->vid = cpp_info.vid;
        info->pid = cpp_info.pid;
        info->is_connected = cpp_info.isConnected;
        return MAKCU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_get_version(makcu_device_t* device, char* version, size_t version_size) {
    if (!device || !version || version_size == 0) {
        return MAKCU_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto cpp_version = device->cpp_device->getVersion();
        safe_copy_string(version, version_size, cpp_version);
        return MAKCU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

// Mouse button control
makcu_error_t makcu_mouse_down(makcu_device_t* device, makcu_mouse_button_t button) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseDown(convert_mouse_button(button));
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_mouse_up(makcu_device_t* device, makcu_mouse_button_t button) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseUp(convert_mouse_button(button));
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_mouse_click(makcu_device_t* device, makcu_mouse_button_t button) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->click(convert_mouse_button(button));
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

// Mouse button state queries
makcu_error_t makcu_mouse_button_state(makcu_device_t* device, makcu_mouse_button_t button, bool* state) {
    if (!device || !state) {
        return MAKCU_ERROR_INVALID_PARAMETER;
    }
    
    try {
        *state = device->cpp_device->mouseButtonState(convert_mouse_button(button));
        return MAKCU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

// Mouse movement
makcu_error_t makcu_mouse_move(makcu_device_t* device, int32_t x, int32_t y) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseMove(x, y);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_mouse_move_smooth(makcu_device_t* device, int32_t x, int32_t y, uint32_t segments) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseMoveSmooth(x, y, segments);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_mouse_move_bezier(makcu_device_t* device, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseMoveBezier(x, y, segments, ctrl_x, ctrl_y);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

// Mouse drag operations
makcu_error_t makcu_mouse_drag(makcu_device_t* device, makcu_mouse_button_t button, int32_t x, int32_t y) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseDrag(convert_mouse_button(button), x, y);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_mouse_drag_smooth(makcu_device_t* device, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseDragSmooth(convert_mouse_button(button), x, y, segments);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_mouse_drag_bezier(makcu_device_t* device, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseDragBezier(convert_mouse_button(button), x, y, segments, ctrl_x, ctrl_y);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

// Mouse wheel
makcu_error_t makcu_mouse_wheel(makcu_device_t* device, int32_t delta) {
    if (!device) {
        return MAKCU_ERROR_INVALID_DEVICE;
    }
    
    try {
        bool success = device->cpp_device->mouseWheel(delta);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

// Mouse locking functions
makcu_error_t makcu_lock_mouse_x(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseX(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_lock_mouse_y(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseY(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_lock_mouse_left(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseLeft(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_lock_mouse_middle(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseMiddle(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_lock_mouse_right(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseRight(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_lock_mouse_side1(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseSide1(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_lock_mouse_side2(makcu_device_t* device, bool lock) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->lockMouseSide2(lock);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

// Lock state queries
makcu_error_t makcu_is_mouse_x_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseXLocked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_mouse_y_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseYLocked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_mouse_left_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseLeftLocked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_mouse_middle_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseMiddleLocked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_mouse_right_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseRightLocked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_mouse_side1_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseSide1Locked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_mouse_side2_locked(makcu_device_t* device, bool* locked) {
    if (!device || !locked) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *locked = device->cpp_device->isMouseSide2Locked();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

// Mouse input catching
makcu_error_t makcu_catch_mouse_left(makcu_device_t* device, uint8_t* result) {
    if (!device || !result) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *result = device->cpp_device->catchMouseLeft();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_catch_mouse_middle(makcu_device_t* device, uint8_t* result) {
    if (!device || !result) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *result = device->cpp_device->catchMouseMiddle();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_catch_mouse_right(makcu_device_t* device, uint8_t* result) {
    if (!device || !result) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *result = device->cpp_device->catchMouseRight();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_catch_mouse_side1(makcu_device_t* device, uint8_t* result) {
    if (!device || !result) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *result = device->cpp_device->catchMouseSide1();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_catch_mouse_side2(makcu_device_t* device, uint8_t* result) {
    if (!device || !result) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *result = device->cpp_device->catchMouseSide2();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

// Button monitoring
makcu_error_t makcu_enable_button_monitoring(makcu_device_t* device, bool enable) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->enableButtonMonitoring(enable);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_button_monitoring_enabled(makcu_device_t* device, bool* enabled) {
    if (!device || !enabled) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *enabled = device->cpp_device->isButtonMonitoringEnabled();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_get_button_mask(makcu_device_t* device, uint8_t* mask) {
    if (!device || !mask) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *mask = device->cpp_device->getButtonMask();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

// Serial spoofing
makcu_error_t makcu_get_mouse_serial(makcu_device_t* device, char* serial, size_t serial_size) {
    if (!device || !serial || serial_size == 0) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        auto cpp_serial = device->cpp_device->getMouseSerial();
        safe_copy_string(serial, serial_size, cpp_serial);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_set_mouse_serial(makcu_device_t* device, const char* serial) {
    if (!device || !serial) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        bool success = device->cpp_device->setMouseSerial(serial);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_reset_mouse_serial(makcu_device_t* device) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->resetMouseSerial();
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

// Device control
makcu_error_t makcu_set_baud_rate(makcu_device_t* device, uint32_t baud_rate) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        bool success = device->cpp_device->setBaudRate(baud_rate);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

// Callbacks
makcu_error_t makcu_set_mouse_button_callback(makcu_device_t* device, makcu_mouse_button_callback_t callback, void* user_data) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    
    try {
        device->mouse_callback = callback;
        device->mouse_callback_user_data = user_data;
        
        if (callback) {
            device->cpp_device->setMouseButtonCallback([device](makcu::MouseButton button, bool pressed) {
                if (device->mouse_callback) {
                    device->mouse_callback(convert_mouse_button_to_c(button), pressed, device->mouse_callback_user_data);
                }
            });
        } else {
            device->cpp_device->setMouseButtonCallback(nullptr);
        }
        
        return MAKCU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_set_connection_callback(makcu_device_t* device, makcu_connection_callback_t callback, void* user_data) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    
    try {
        device->connection_callback = callback;
        device->connection_callback_user_data = user_data;
        
        if (callback) {
            device->cpp_device->setConnectionCallback([device](bool connected) {
                if (device->connection_callback) {
                    device->connection_callback(connected, device->connection_callback_user_data);
                }
            });
        } else {
            device->cpp_device->setConnectionCallback(nullptr);
        }
        
        return MAKCU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

// High-level automation
makcu_error_t makcu_click_sequence(makcu_device_t* device, const makcu_mouse_button_t* buttons, size_t count, uint32_t delay_ms) {
    if (!device || !buttons) return MAKCU_ERROR_INVALID_PARAMETER;
    
    try {
        std::vector<makcu::MouseButton> cpp_buttons;
        cpp_buttons.reserve(count);
        
        for (size_t i = 0; i < count; i++) {
            cpp_buttons.push_back(convert_mouse_button(buttons[i]));
        }
        
        bool success = device->cpp_device->clickSequence(cpp_buttons, std::chrono::milliseconds(delay_ms));
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

makcu_error_t makcu_move_pattern(makcu_device_t* device, const makcu_point_t* points, size_t count, bool smooth, uint32_t segments) {
    if (!device || !points) return MAKCU_ERROR_INVALID_PARAMETER;
    
    try {
        std::vector<std::pair<int32_t, int32_t>> cpp_points;
        cpp_points.reserve(count);
        
        for (size_t i = 0; i < count; i++) {
            cpp_points.emplace_back(points[i].x, points[i].y);
        }
        
        bool success = device->cpp_device->movePattern(cpp_points, smooth, segments);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) {
        return handle_exception();
    }
}

// Performance mode
makcu_error_t makcu_enable_high_performance_mode(makcu_device_t* device, bool enable) {
    if (!device) return MAKCU_ERROR_INVALID_DEVICE;
    try {
        device->cpp_device->enableHighPerformanceMode(enable);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_is_high_performance_mode_enabled(makcu_device_t* device, bool* enabled) {
    if (!device || !enabled) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        *enabled = device->cpp_device->isHighPerformanceModeEnabled();
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

// Batch operations
makcu_batch_builder_t* makcu_create_batch(makcu_device_t* device) {
    if (!device) return nullptr;
    
    try {
        auto cpp_batch = device->cpp_device->createBatch();
        return new makcu_batch_builder(std::move(cpp_batch));
    } catch (...) {
        return nullptr;
    }
}

void makcu_batch_destroy(makcu_batch_builder_t* batch) {
    delete batch;
}

makcu_error_t makcu_batch_move(makcu_batch_builder_t* batch, int32_t x, int32_t y) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->move(x, y);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_move_smooth(makcu_batch_builder_t* batch, int32_t x, int32_t y, uint32_t segments) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->moveSmooth(x, y, segments);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_move_bezier(makcu_batch_builder_t* batch, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->moveBezier(x, y, segments, ctrl_x, ctrl_y);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_click(makcu_batch_builder_t* batch, makcu_mouse_button_t button) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->click(convert_mouse_button(button));
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_press(makcu_batch_builder_t* batch, makcu_mouse_button_t button) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->press(convert_mouse_button(button));
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_release(makcu_batch_builder_t* batch, makcu_mouse_button_t button) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->release(convert_mouse_button(button));
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_scroll(makcu_batch_builder_t* batch, int32_t delta) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->scroll(delta);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_drag(makcu_batch_builder_t* batch, makcu_mouse_button_t button, int32_t x, int32_t y) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->drag(convert_mouse_button(button), x, y);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_drag_smooth(makcu_batch_builder_t* batch, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->dragSmooth(convert_mouse_button(button), x, y, segments);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_drag_bezier(makcu_batch_builder_t* batch, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        batch->cpp_batch->dragBezier(convert_mouse_button(button), x, y, segments, ctrl_x, ctrl_y);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_batch_execute(makcu_batch_builder_t* batch) {
    if (!batch) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        bool success = batch->cpp_batch->execute();
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

// Raw command interface
makcu_error_t makcu_send_raw_command(makcu_device_t* device, const char* command) {
    if (!device || !command) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        bool success = device->cpp_device->sendRawCommand(command);
        return success ? MAKCU_SUCCESS : MAKCU_ERROR_COMMAND_FAILED;
    } catch (...) { return handle_exception(); }
}

makcu_error_t makcu_receive_raw_response(makcu_device_t* device, char* response, size_t response_size) {
    if (!device || !response || response_size == 0) return MAKCU_ERROR_INVALID_PARAMETER;
    try {
        auto cpp_response = device->cpp_device->receiveRawResponse();
        safe_copy_string(response, response_size, cpp_response);
        return MAKCU_SUCCESS;
    } catch (...) { return handle_exception(); }
}

// Utility functions
const char* makcu_mouse_button_to_string(makcu_mouse_button_t button) {
    switch (button) {
        case MAKCU_MOUSE_LEFT: return "LEFT";
        case MAKCU_MOUSE_RIGHT: return "RIGHT";
        case MAKCU_MOUSE_MIDDLE: return "MIDDLE";
        case MAKCU_MOUSE_SIDE1: return "SIDE1";
        case MAKCU_MOUSE_SIDE2: return "SIDE2";
        default: return "Unknown";
    }
}

makcu_mouse_button_t makcu_string_to_mouse_button(const char* button_name) {
    if (!button_name) return MAKCU_MOUSE_LEFT;
    if (strcmp(button_name, "LEFT") == 0) return MAKCU_MOUSE_LEFT;
    if (strcmp(button_name, "RIGHT") == 0) return MAKCU_MOUSE_RIGHT;
    if (strcmp(button_name, "MIDDLE") == 0) return MAKCU_MOUSE_MIDDLE;
    if (strcmp(button_name, "SIDE1") == 0) return MAKCU_MOUSE_SIDE1;
    if (strcmp(button_name, "SIDE2") == 0) return MAKCU_MOUSE_SIDE2;
    return MAKCU_MOUSE_LEFT;
}

// Performance profiling
void makcu_profiler_enable(bool enable) {
    makcu::PerformanceProfiler::enableProfiling(enable);
}

void makcu_profiler_reset_stats(void) {
    makcu::PerformanceProfiler::resetStats();
}

int makcu_profiler_get_stats(makcu_perf_stat_t* stats, int max_stats) {
    if (!stats || max_stats <= 0) return 0;
    
    try {
        auto cpp_stats = makcu::PerformanceProfiler::getStats();
        int count = std::min(max_stats, static_cast<int>(cpp_stats.size()));
        
        int i = 0;
        for (const auto& [command, data] : cpp_stats) {
            if (i >= count) break;
            
            safe_copy_string(stats[i].command_name, sizeof(stats[i].command_name), command);
            stats[i].call_count = data.first;
            stats[i].total_microseconds = data.second;
            i++;
        }
        
        return count;
    } catch (...) {
        return 0;
    }
}

} // extern "C"
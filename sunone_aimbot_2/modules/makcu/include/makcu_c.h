#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Export macros for C API
#ifdef _WIN32
    #ifdef MAKCU_EXPORTS
        // Building shared library - export symbols
        #define MAKCU_C_API __declspec(dllexport)
    #elif defined(MAKCU_SHARED)
        // Using shared library - import symbols
        #define MAKCU_C_API __declspec(dllimport)
    #else
        // Using static library - no decoration needed
        #define MAKCU_C_API
    #endif
#else
    // Non-Windows platforms
    #ifdef __GNUC__
        #define MAKCU_C_API __attribute__((visibility("default")))
    #else
        #define MAKCU_C_API
    #endif
#endif

// Forward declarations - opaque types
typedef struct makcu_device makcu_device_t;
typedef struct makcu_batch_builder makcu_batch_builder_t;

// Enums (C-compatible)
typedef enum {
    MAKCU_MOUSE_LEFT = 0,
    MAKCU_MOUSE_RIGHT = 1,
    MAKCU_MOUSE_MIDDLE = 2,
    MAKCU_MOUSE_SIDE1 = 3,
    MAKCU_MOUSE_SIDE2 = 4
} makcu_mouse_button_t;

typedef enum {
    MAKCU_STATUS_DISCONNECTED = 0,
    MAKCU_STATUS_CONNECTING = 1,
    MAKCU_STATUS_CONNECTED = 2,
    MAKCU_STATUS_CONNECTION_ERROR = 3
} makcu_connection_status_t;

// Simple structs (C-compatible)
typedef struct {
    char port[256];
    char description[256];
    uint16_t vid;
    uint16_t pid;
    bool is_connected;
} makcu_device_info_t;

typedef struct {
    bool left;
    bool right;
    bool middle;
    bool side1;
    bool side2;
} makcu_mouse_button_states_t;

// Callback function pointers
typedef void (*makcu_mouse_button_callback_t)(makcu_mouse_button_t button, bool pressed, void* user_data);
typedef void (*makcu_connection_callback_t)(bool connected, void* user_data);

// Error handling
typedef enum {
    MAKCU_SUCCESS = 0,
    MAKCU_ERROR_INVALID_DEVICE = 1,
    MAKCU_ERROR_CONNECTION_FAILED = 2,
    MAKCU_ERROR_COMMAND_FAILED = 3,
    MAKCU_ERROR_TIMEOUT = 4,
    MAKCU_ERROR_INVALID_PARAMETER = 5,
    MAKCU_ERROR_OUT_OF_MEMORY = 6
} makcu_error_t;

// Get error message string
MAKCU_C_API const char* makcu_error_string(makcu_error_t error);

// Device management
MAKCU_C_API makcu_device_t* makcu_device_create(void);
MAKCU_C_API void makcu_device_destroy(makcu_device_t* device);

// Static device discovery
MAKCU_C_API int makcu_find_devices(makcu_device_info_t* devices, int max_devices);
MAKCU_C_API makcu_error_t makcu_find_first_device(char* port, size_t port_size);

// Connection management
MAKCU_C_API makcu_error_t makcu_connect(makcu_device_t* device, const char* port);
MAKCU_C_API void makcu_disconnect(makcu_device_t* device);
MAKCU_C_API bool makcu_is_connected(makcu_device_t* device);
MAKCU_C_API makcu_connection_status_t makcu_get_status(makcu_device_t* device);

// Device information
MAKCU_C_API makcu_error_t makcu_get_device_info(makcu_device_t* device, makcu_device_info_t* info);
MAKCU_C_API makcu_error_t makcu_get_version(makcu_device_t* device, char* version, size_t version_size);

// Mouse button control
MAKCU_C_API makcu_error_t makcu_mouse_down(makcu_device_t* device, makcu_mouse_button_t button);
MAKCU_C_API makcu_error_t makcu_mouse_up(makcu_device_t* device, makcu_mouse_button_t button);
MAKCU_C_API makcu_error_t makcu_mouse_click(makcu_device_t* device, makcu_mouse_button_t button);

// Mouse button state queries
MAKCU_C_API makcu_error_t makcu_mouse_button_state(makcu_device_t* device, makcu_mouse_button_t button, bool* state);

// Mouse movement
MAKCU_C_API makcu_error_t makcu_mouse_move(makcu_device_t* device, int32_t x, int32_t y);
MAKCU_C_API makcu_error_t makcu_mouse_move_smooth(makcu_device_t* device, int32_t x, int32_t y, uint32_t segments);
MAKCU_C_API makcu_error_t makcu_mouse_move_bezier(makcu_device_t* device, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y);

// Mouse drag operations
MAKCU_C_API makcu_error_t makcu_mouse_drag(makcu_device_t* device, makcu_mouse_button_t button, int32_t x, int32_t y);
MAKCU_C_API makcu_error_t makcu_mouse_drag_smooth(makcu_device_t* device, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments);
MAKCU_C_API makcu_error_t makcu_mouse_drag_bezier(makcu_device_t* device, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y);

// Mouse wheel
MAKCU_C_API makcu_error_t makcu_mouse_wheel(makcu_device_t* device, int32_t delta);

// Mouse locking
MAKCU_C_API makcu_error_t makcu_lock_mouse_x(makcu_device_t* device, bool lock);
MAKCU_C_API makcu_error_t makcu_lock_mouse_y(makcu_device_t* device, bool lock);
MAKCU_C_API makcu_error_t makcu_lock_mouse_left(makcu_device_t* device, bool lock);
MAKCU_C_API makcu_error_t makcu_lock_mouse_middle(makcu_device_t* device, bool lock);
MAKCU_C_API makcu_error_t makcu_lock_mouse_right(makcu_device_t* device, bool lock);
MAKCU_C_API makcu_error_t makcu_lock_mouse_side1(makcu_device_t* device, bool lock);
MAKCU_C_API makcu_error_t makcu_lock_mouse_side2(makcu_device_t* device, bool lock);

// Lock state queries
MAKCU_C_API makcu_error_t makcu_is_mouse_x_locked(makcu_device_t* device, bool* locked);
MAKCU_C_API makcu_error_t makcu_is_mouse_y_locked(makcu_device_t* device, bool* locked);
MAKCU_C_API makcu_error_t makcu_is_mouse_left_locked(makcu_device_t* device, bool* locked);
MAKCU_C_API makcu_error_t makcu_is_mouse_middle_locked(makcu_device_t* device, bool* locked);
MAKCU_C_API makcu_error_t makcu_is_mouse_right_locked(makcu_device_t* device, bool* locked);
MAKCU_C_API makcu_error_t makcu_is_mouse_side1_locked(makcu_device_t* device, bool* locked);
MAKCU_C_API makcu_error_t makcu_is_mouse_side2_locked(makcu_device_t* device, bool* locked);

// Mouse input catching
MAKCU_C_API makcu_error_t makcu_catch_mouse_left(makcu_device_t* device, uint8_t* result);
MAKCU_C_API makcu_error_t makcu_catch_mouse_middle(makcu_device_t* device, uint8_t* result);
MAKCU_C_API makcu_error_t makcu_catch_mouse_right(makcu_device_t* device, uint8_t* result);
MAKCU_C_API makcu_error_t makcu_catch_mouse_side1(makcu_device_t* device, uint8_t* result);
MAKCU_C_API makcu_error_t makcu_catch_mouse_side2(makcu_device_t* device, uint8_t* result);

// Button monitoring
MAKCU_C_API makcu_error_t makcu_enable_button_monitoring(makcu_device_t* device, bool enable);
MAKCU_C_API makcu_error_t makcu_is_button_monitoring_enabled(makcu_device_t* device, bool* enabled);
MAKCU_C_API makcu_error_t makcu_get_button_mask(makcu_device_t* device, uint8_t* mask);

// Serial spoofing
MAKCU_C_API makcu_error_t makcu_get_mouse_serial(makcu_device_t* device, char* serial, size_t serial_size);
MAKCU_C_API makcu_error_t makcu_set_mouse_serial(makcu_device_t* device, const char* serial);
MAKCU_C_API makcu_error_t makcu_reset_mouse_serial(makcu_device_t* device);

// Device control
MAKCU_C_API makcu_error_t makcu_set_baud_rate(makcu_device_t* device, uint32_t baud_rate);

// Callbacks
MAKCU_C_API makcu_error_t makcu_set_mouse_button_callback(makcu_device_t* device, makcu_mouse_button_callback_t callback, void* user_data);
MAKCU_C_API makcu_error_t makcu_set_connection_callback(makcu_device_t* device, makcu_connection_callback_t callback, void* user_data);

// High-level automation
MAKCU_C_API makcu_error_t makcu_click_sequence(makcu_device_t* device, const makcu_mouse_button_t* buttons, size_t count, uint32_t delay_ms);

// Move pattern - simplified version
typedef struct {
    int32_t x;
    int32_t y;
} makcu_point_t;

MAKCU_C_API makcu_error_t makcu_move_pattern(makcu_device_t* device, const makcu_point_t* points, size_t count, bool smooth, uint32_t segments);

// Performance mode
MAKCU_C_API makcu_error_t makcu_enable_high_performance_mode(makcu_device_t* device, bool enable);
MAKCU_C_API makcu_error_t makcu_is_high_performance_mode_enabled(makcu_device_t* device, bool* enabled);

// Batch operations
MAKCU_C_API makcu_batch_builder_t* makcu_create_batch(makcu_device_t* device);
MAKCU_C_API void makcu_batch_destroy(makcu_batch_builder_t* batch);
MAKCU_C_API makcu_error_t makcu_batch_move(makcu_batch_builder_t* batch, int32_t x, int32_t y);
MAKCU_C_API makcu_error_t makcu_batch_move_smooth(makcu_batch_builder_t* batch, int32_t x, int32_t y, uint32_t segments);
MAKCU_C_API makcu_error_t makcu_batch_move_bezier(makcu_batch_builder_t* batch, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y);
MAKCU_C_API makcu_error_t makcu_batch_click(makcu_batch_builder_t* batch, makcu_mouse_button_t button);
MAKCU_C_API makcu_error_t makcu_batch_press(makcu_batch_builder_t* batch, makcu_mouse_button_t button);
MAKCU_C_API makcu_error_t makcu_batch_release(makcu_batch_builder_t* batch, makcu_mouse_button_t button);
MAKCU_C_API makcu_error_t makcu_batch_scroll(makcu_batch_builder_t* batch, int32_t delta);
MAKCU_C_API makcu_error_t makcu_batch_drag(makcu_batch_builder_t* batch, makcu_mouse_button_t button, int32_t x, int32_t y);
MAKCU_C_API makcu_error_t makcu_batch_drag_smooth(makcu_batch_builder_t* batch, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments);
MAKCU_C_API makcu_error_t makcu_batch_drag_bezier(makcu_batch_builder_t* batch, makcu_mouse_button_t button, int32_t x, int32_t y, uint32_t segments, int32_t ctrl_x, int32_t ctrl_y);
MAKCU_C_API makcu_error_t makcu_batch_execute(makcu_batch_builder_t* batch);

// Raw command interface
MAKCU_C_API makcu_error_t makcu_send_raw_command(makcu_device_t* device, const char* command);
MAKCU_C_API makcu_error_t makcu_receive_raw_response(makcu_device_t* device, char* response, size_t response_size);

// Utility functions
MAKCU_C_API const char* makcu_mouse_button_to_string(makcu_mouse_button_t button);
MAKCU_C_API makcu_mouse_button_t makcu_string_to_mouse_button(const char* button_name);

// Performance profiling
MAKCU_C_API void makcu_profiler_enable(bool enable);
MAKCU_C_API void makcu_profiler_reset_stats(void);

// Performance stats result structure
typedef struct {
    char command_name[64];
    uint64_t call_count;
    uint64_t total_microseconds;
} makcu_perf_stat_t;

MAKCU_C_API int makcu_profiler_get_stats(makcu_perf_stat_t* stats, int max_stats);

#ifdef __cplusplus
}
#endif
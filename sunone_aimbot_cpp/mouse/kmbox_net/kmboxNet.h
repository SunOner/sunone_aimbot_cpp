#ifndef KMBOX_NET_H
#define KMBOX_NET_H
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <stdio.h>

#include <cmath>
#pragma warning(disable : 4996)

// Command codes
#define     cmd_connect         0xaf3c2828 // ok: Connect to the box
#define     cmd_mouse_move      0xaede7345 // ok: Mouse move
#define     cmd_mouse_left      0x9823AE8D // ok: Mouse left button control
#define     cmd_mouse_middle    0x97a3AE8D // ok: Mouse middle button control
#define     cmd_mouse_right     0x238d8212 // ok: Mouse right button control
#define     cmd_mouse_wheel     0xffeead38 // ok: Mouse wheel control
#define     cmd_mouse_automove  0xaede7346 // ok: Simulated human mouse movement control
#define     cmd_keyboard_all    0x123c2c2f // ok: Control all keyboard parameters
#define     cmd_reboot          0xaa8855aa // ok: Box reboot
#define     cmd_bazerMove       0xa238455a // ok: Mouse Bezier movement
#define     cmd_monitor         0x27388020 // ok: Monitor physical mouse/keyboard data on the box
#define     cmd_debug           0x27382021 // ok: Enable debug info
#define     cmd_mask_mouse      0x23234343 // ok: Block physical mouse/keyboard
#define     cmd_unmask_all      0x23344343 // ok: Unblock physical mouse/keyboard
#define     cmd_setconfig       0x1d3d3323 // ok: Set IP configuration info
#define     cmd_showpic         0x12334883 // ok: Display image

extern SOCKET sockClientfd; // Socket communication handle
typedef struct
{
	unsigned int  mac;        // MAC address of the box (required)
	unsigned int  rand;       // Random value
	unsigned int  indexpts;   // Timestamp
	unsigned int  cmd;        // Command code
} cmd_head_t;


typedef struct
{
	unsigned char buff[1024];	//
}cmd_data_t;
typedef struct
{
	unsigned short buff[512];	//
}cmd_u16_t;

// Mouse data structure
typedef struct
{
    int button;
    int x;
    int y;
    int wheel;
    int point[10]; // For Bezier curve control (reserved for up to 5th order derivative)
} soft_mouse_t;

// Keyboard data structure
typedef struct
{
    char ctrl;
    char resvel;
    char button[10];
} soft_keyboard_t;

// Union structure
typedef struct
{
    cmd_head_t head;
    union {
        cmd_data_t      u8buff;         // Buffer
        cmd_u16_t       u16buff;        // U16
        soft_mouse_t    cmd_mouse;      // Mouse command to send
        soft_keyboard_t cmd_keyboard;   // Keyboard command to send
    };
} client_tx;

enum
{
    err_creat_socket = -9000,   // Failed to create socket
    err_net_version,            // Socket version error
    err_net_tx,                 // Socket send error
    err_net_rx_timeout,         // Socket receive timeout
    err_net_cmd,                // Command error
    err_net_pts,                // Timestamp error
    success = 0,                // Success
    usb_dev_tx_timeout,         // USB device send failed
};


/*
Connect to the kmboxNet box. Input parameters:
ip   : IP address of the box (displayed on the screen)
port : Communication port number (displayed on the screen)
mac  : MAC address of the box (displayed on the screen)
Return value: 0 if connection is successful, see error codes for other values
*/
int kmNet_init(char* ip, char* port, char* mac); // ok
int kmNet_mouse_move(short x, short y);          // ok
int kmNet_mouse_left(int isdown);                // ok
int kmNet_mouse_right(int isdown);               // ok
int kmNet_mouse_middle(int isdown);              // ok
int kmNet_mouse_wheel(int wheel);                // ok
int kmNet_mouse_all(int button, int x, int y, int wheel); // ok
int kmNet_mouse_move_auto(int x, int y, int time_ms);     // ok
int kmNet_mouse_move_beizer(int x, int y, int ms, int x1, int y1, int x2, int y2); // Second-order curve

// Keyboard functions
int kmNet_keydown(int vkey); // ok
int kmNet_keyup(int vkey);   // ok

// Monitoring series
int kmNet_monitor(short port);            // Enable/disable physical mouse/keyboard monitoring
int kmNet_monitor_mouse_left();           // Query physical mouse left button state
int kmNet_monitor_mouse_middle();         // Query mouse middle button state
int kmNet_monitor_mouse_right();          // Query mouse right button state
int kmNet_monitor_mouse_side1();          // Query mouse side button 1 state
int kmNet_monitor_mouse_side2();          // Query mouse side button 2 state
int kmNet_monitor_keyboard(short vk_key); // Query the state of the specified keyboard key

// Physical mouse/keyboard masking series
int kmNet_mask_mouse_left(int enable);    // Block mouse left button
int kmNet_mask_mouse_right(int enable);   // Block mouse right button
int kmNet_mask_mouse_middle(int enable);  // Block mouse middle button
int kmNet_mask_mouse_side1(int enable);   // Block mouse side button 1
int kmNet_mask_mouse_side2(int enable);   // Block mouse side button 2
int kmNet_mask_mouse_x(int enable);       // Block mouse X-axis
int kmNet_mask_mouse_y(int enable);       // Block mouse Y-axis
int kmNet_mask_mouse_wheel(int enable);   // Block mouse wheel
int kmNet_mask_keyboard(short vkey);      // Block the specified keyboard key
int kmNet_unmask_keyboard(short vkey);    // Unblock the specified keyboard key
int kmNet_unmask_all();                   // Unblock all previously set physical masks

// Configuration functions
int kmNet_reboot(void);
int kmNet_setconfig(char* ip, unsigned short port);        // Configure the box IP address
int kmNet_debug(short port, char enable);                  // Enable debugging
int kmNet_lcd_color(unsigned short rgb565);                // Fill the entire LCD screen with the specified color. Use black to clear the screen
int kmNet_lcd_picture_bottom(unsigned char* buff_128_80);  // Display 128x80 image at the bottom
int kmNet_lcd_picture(unsigned char* buff_128_160);        // Display 128x160 image on the whole screen
#endif // KMBOX_NET_H
#include <time.h>

#include "kmbox_net/kmboxNet.h"
#include "kmbox_net/HidTable.h"

#define monitor_ok    2
#define monitor_exit  0
SOCKET sockClientfd = 0;              // Mouse and keyboard network communication handle
SOCKET sockMonitorfd = 0;             // Monitor network communication handle
client_tx tx;                         // Data to send
client_tx rx;                         // Data to receive
SOCKADDR_IN addrSrv;
soft_mouse_t    softmouse;            // Software mouse data
soft_keyboard_t softkeyboard;         // Software keyboard data
static int monitor_run = 0;           // Whether physical mouse/keyboard monitoring is running
static int mask_keyboard_mouse_flag = 0; // Mouse/keyboard block status
static short monitor_port = 0;


#pragma pack(1)
typedef struct {
	unsigned char report_id;
	unsigned char buttons;		// 8 buttons available
	short x;					// -32767 to 32767
	short y;					// -32767 to 32767
	short wheel;				// -32767 to 32767
}standard_mouse_report_t;

typedef struct {
	unsigned char report_id;
	unsigned char buttons;      // 8 buttons control keys
	unsigned char data[10];     // Regular keys
} standard_keyboard_report_t;
#pragma pack()

standard_mouse_report_t     hw_mouse;     // Hardware mouse message
standard_keyboard_report_t  hw_keyboard;  // Hardware keyboard message

// Generate a random number between A and B
int myrand(int a, int b)
{
	int min = a < b ? a : b;
	int max = a > b ? a : b;
	return ((rand() % (max - min)) + min);
}

unsigned int StrToHex(char* pbSrc, int nLen)
{
	char h1, h2;
	unsigned char s1, s2;
	int i;
	unsigned int pbDest[16] = { 0 };
	for (i = 0; i < nLen; i++) {
		h1 = pbSrc[2 * i];
		h2 = pbSrc[2 * i + 1];
		s1 = toupper(h1) - 0x30;
		if (s1 > 9)
			s1 -= 7;
		s2 = toupper(h2) - 0x30;
		if (s2 > 9)
			s2 -= 7;
		pbDest[i] = s1 * 16 + s2;
	}
	return pbDest[0] << 24 | pbDest[1] << 16 | pbDest[2] << 8 | pbDest[3];
}

int NetRxReturnHandle(client_tx* rx, client_tx* tx)      // Received content
{
	if (rx->head.cmd != tx->head.cmd)
		return  err_net_cmd;    // Command code error
	if (rx->head.indexpts != tx->head.indexpts)
		return  err_net_pts;    // Timestamp error
	return 0;                   // No error, return 0
	//return  rx->head.rand;    // Actual return value
}


/*
Connect to kmboxNet box. The input parameters are:
ip   : The IP address of the box (displayed on the screen, e.g., 192.168.2.88)
port : Communication port number (displayed on the screen, e.g., 6234)
mac  : The MAC address of the box (displayed on the screen, e.g., 12345)
Return value: 0 means success, non-zero values refer to error codes
*/
int kmNet_init(char* ip, char* port, char* mac)
{
	WORD wVersionRequested; WSADATA wsaData; int err;
	wVersionRequested = MAKEWORD(1, 1);
	err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0)        return err_creat_socket;
	if (LOBYTE(wsaData.wVersion) != 1 || HIBYTE(wsaData.wVersion) != 1) {
		WSACleanup(); sockClientfd = -1;
		return err_net_version;
	}
	srand((unsigned)time(NULL));
	sockClientfd = socket(AF_INET, SOCK_DGRAM, 0);
	addrSrv.sin_addr.S_un.S_addr = inet_addr(ip);
	addrSrv.sin_family = AF_INET;
	addrSrv.sin_port = htons(atoi(port)); // Port UUID[1] >> 16 high 16 bits
	tx.head.mac = StrToHex(mac, 4);         // Box MAC, fixed UUID[1]
	tx.head.rand = rand();                  // Random value. Can be used later for packet encryption. Reserved for now.
	tx.head.indexpts = 0;                   // Command statistics value
	tx.head.cmd = cmd_connect;              // Command
	memset(&softmouse, 0, sizeof(softmouse));       // Clear software mouse data
	memset(&softkeyboard, 0, sizeof(softkeyboard)); // Clear software keyboard data
	err = sendto(sockClientfd, (const char*)&tx, sizeof(cmd_head_t), 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	Sleep(20); // The first connection may take longer
	int clen = sizeof(addrSrv);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&addrSrv, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

/*
Move the mouse by x, y units. One-time move, no trajectory simulation, fastest speed.
Use this function when implementing your own trajectory movement.
Return value: 0 if successful, nonzero means error.
*/
int kmNet_mouse_move(short x, short y)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_mouse_move;    // Command
	tx.head.rand = rand();           // Random obfuscation value
	softmouse.x = x;
	softmouse.y = y;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	softmouse.x = 0;
	softmouse.y = 0;
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}



/*
Mouse left button control
isdown : 0 = release, 1 = press
Return value: 0 if successful, nonzero means error.
*/
int kmNet_mouse_left(int isdown)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_mouse_left;    // Command
	tx.head.rand = rand();           // Random obfuscation value
	softmouse.button = (isdown ? (softmouse.button | 0x01) : (softmouse.button & (~0x01)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

/*
Mouse middle button control
isdown : 0 = release, 1 = press
Return value: 0 if successful, nonzero means error.
*/
int kmNet_mouse_middle(int isdown)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_mouse_middle;  // Command
	tx.head.rand = rand();           // Random obfuscation value
	softmouse.button = (isdown ? (softmouse.button | 0x04) : (softmouse.button & (~0x04)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

/*
Mouse right button control
isdown : 0 = release, 1 = press
Return value: 0 if successful, nonzero means error.
*/
int kmNet_mouse_right(int isdown)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_mouse_right;   // Command
	tx.head.rand = rand();           // Random obfuscation value
	softmouse.button = (isdown ? (softmouse.button | 0x02) : (softmouse.button & (~0x02)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Mouse wheel control
int kmNet_mouse_wheel(int wheel)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_mouse_wheel;   // Command
	tx.head.rand = rand();           // Random obfuscation value
	softmouse.wheel = wheel;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.wheel = 0;
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


/*
Mouse full report control function
*/
int kmNet_mouse_all(int button, int x, int y, int wheel)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_mouse_wheel;   // Command
	tx.head.rand = rand();           // Random obfuscation value
	softmouse.button = button;
	softmouse.x = x;
	softmouse.y = y;
	softmouse.wheel = wheel;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.x = 0;
	softmouse.y = 0;
	softmouse.wheel = 0;
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

/*
Move the mouse by x, y units. Simulate human-like movement of x, y units.
This avoids detection of abnormal mouse and keyboard behavior.
If you do not implement a movement curve, it is recommended to use this function.
This function will not cause jumps; it approaches the target using minimal steps.
It takes more time than kmNet_mouse_move.
'ms' specifies how many milliseconds the movement should take.
Note: do not set 'ms' too low, otherwise abnormal data may still be detected.
Try to imitate human operation. Actual time may be less than 'ms'.
*/
int kmNet_mouse_move_auto(int x, int y, int ms)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;                  // Command statistics value
	tx.head.cmd = cmd_mouse_automove;    // Command
	tx.head.rand = ms;                   // Random obfuscation value (here: movement time in ms)
	softmouse.x = x;
	softmouse.y = y;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.x = 0;                     // Clear
	softmouse.y = 0;                     // Clear
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


/*
Second-order Bezier curve control
x, y   : Target point coordinates
ms     : Time to fit this process (in milliseconds)
x1, y1 : Control point p1 coordinates
x2, y2 : Control point p2 coordinates
*/
int kmNet_mouse_move_beizer(int x, int y, int ms, int x1, int y1, int x2, int y2)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;               // Command statistics value
	tx.head.cmd = cmd_bazerMove;      // Command
	tx.head.rand = ms;                // Random obfuscation value
	softmouse.x = x;
	softmouse.y = y;
	softmouse.point[0] = x1;
	softmouse.point[1] = y1;
	softmouse.point[2] = x2;
	softmouse.point[3] = y2;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.x = 0;
	softmouse.y = 0;
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


/*
Key down event.
If vk_key is between KEY_LEFTCONTROL and KEY_RIGHT_GUI, it's a control key.
Otherwise, it's a regular key.
For regular keys, the function tries to add vk_key to the queue. If the queue is full, the oldest key is removed.
*/
int kmNet_keydown(int vk_key)
{
	int i;
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI) // Control key
	{
		switch (vk_key)
		{
		case KEY_LEFTCONTROL: softkeyboard.ctrl |= BIT0; break;
		case KEY_LEFTSHIFT:   softkeyboard.ctrl |= BIT1; break;
		case KEY_LEFTALT:     softkeyboard.ctrl |= BIT2; break;
		case KEY_LEFT_GUI:    softkeyboard.ctrl |= BIT3; break;
		case KEY_RIGHTCONTROL:softkeyboard.ctrl |= BIT4; break;
		case KEY_RIGHTSHIFT:  softkeyboard.ctrl |= BIT5; break;
		case KEY_RIGHTALT:    softkeyboard.ctrl |= BIT6; break;
		case KEY_RIGHT_GUI:   softkeyboard.ctrl |= BIT7; break;
		}
	}
	else
	{   // Regular key
		for (i = 0; i < 10; i++) // First, check if vk_key already exists in the queue
		{
			if (softkeyboard.button[i] == vk_key)
				goto KM_down_send; // vk_key already in the queue, just send
		}
		// vk_key not in the queue
		for (i = 0; i < 10; i++) // Traverse all data, add vk_key to the queue
		{
			if (softkeyboard.button[i] == 0)
			{   // vk_key already in the queue, just send
				softkeyboard.button[i] = vk_key;
				goto KM_down_send;
			}
		}
		// Queue is full, remove the first one
		memcpy(&softkeyboard.button[0], &softkeyboard.button[1], 10);
		softkeyboard.button[9] = vk_key;
	}
KM_down_send:
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;               // Command statistics value
	tx.head.cmd = cmd_keyboard_all;   // Command
	tx.head.rand = rand();            // Random obfuscation value
	memcpy(&tx.cmd_keyboard, &softkeyboard, sizeof(soft_keyboard_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_keyboard_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


int kmNet_keyup(int vk_key)
{
	int i;
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI) // Control key
	{
		switch (vk_key)
		{
		case KEY_LEFTCONTROL: softkeyboard.ctrl &= ~BIT0; break;
		case KEY_LEFTSHIFT:   softkeyboard.ctrl &= ~BIT1; break;
		case KEY_LEFTALT:     softkeyboard.ctrl &= ~BIT2; break;
		case KEY_LEFT_GUI:    softkeyboard.ctrl &= ~BIT3; break;
		case KEY_RIGHTCONTROL:softkeyboard.ctrl &= ~BIT4; break;
		case KEY_RIGHTSHIFT:  softkeyboard.ctrl &= ~BIT5; break;
		case KEY_RIGHTALT:    softkeyboard.ctrl &= ~BIT6; break;
		case KEY_RIGHT_GUI:   softkeyboard.ctrl &= ~BIT7; break;
		}
	}
	else
	{   // Regular key
		for (i = 0; i < 10; i++) // First, check if vk_key is in the queue
		{
			if (softkeyboard.button[i] == vk_key) // vk_key found in the queue
			{
				memcpy(&softkeyboard.button[i], &softkeyboard.button[i + 1], 10 - i);
				softkeyboard.button[9] = 0;
				goto KM_up_send;
			}
		}
	}
KM_up_send:
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;               // Command statistics value
	tx.head.cmd = cmd_keyboard_all;   // Command
	tx.head.rand = rand();            // Random obfuscation value
	memcpy(&tx.cmd_keyboard, &softkeyboard, sizeof(soft_keyboard_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_keyboard_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Reboot the box
int kmNet_reboot(void)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;               // Command statistics value
	tx.head.cmd = cmd_reboot;         // Command
	tx.head.rand = rand();            // Random obfuscation value
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	WSACleanup();
	sockClientfd = -1;
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);

}


// Listen to physical mouse and keyboard
//static HANDLE handle_listen = NULL;
DWORD WINAPI ThreadListenProcess(LPVOID lpParameter)
{
	WSADATA wsaData; int ret;
	WSAStartup(MAKEWORD(1, 1), &wsaData);            // Create socket, SOCK_DGRAM specifies UDP protocol
	sockMonitorfd = socket(AF_INET, SOCK_DGRAM, 0);  // Bind socket
	sockaddr_in servAddr;
	memset(&servAddr, 0, sizeof(servAddr));          // Fill every byte with 0
	servAddr.sin_family = PF_INET;                   // Use IPv4 address
	servAddr.sin_addr.s_addr = INADDR_ANY;           // Automatically obtain IP address
	servAddr.sin_port = htons(monitor_port);         // Listening port
	ret = bind(sockMonitorfd, (SOCKADDR*)&servAddr, sizeof(SOCKADDR));
	SOCKADDR cliAddr;  // Client address info
	int nSize = sizeof(SOCKADDR);
	char buff[1024];   // Buffer
	monitor_run = monitor_ok;
	while (1) {
		int ret = recvfrom(sockMonitorfd, buff, 1024, 0, &cliAddr, &nSize); // Blocking read
		if (ret > 0)
		{
			memcpy(&hw_mouse, buff, sizeof(hw_mouse));                          // Physical mouse state
			memcpy(&hw_keyboard, &buff[sizeof(hw_mouse)], sizeof(hw_keyboard)); // Physical keyboard state
		}
		else
		{
			break;
		}
	}
	monitor_run = 0;
	sockMonitorfd = 0;
	return 0;
}

// Enable mouse and keyboard monitoring. Port number must be in range 1024–49151
int kmNet_monitor(short port)
{
	int err;
	if (sockClientfd <= 0)       return err_creat_socket;
	tx.head.indexpts++;              // Command statistics value
	tx.head.cmd = cmd_monitor;       // Command
	if (port) {
		monitor_port = port;                 // The port used to listen for physical mouse and keyboard data
		tx.head.rand = port | 0xaa55 << 16;  // Random obfuscation value
	}
	else
		tx.head.rand = 0;    // Random obfuscation value
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (sockMonitorfd > 0)   // Close listener
	{
		closesocket(sockMonitorfd);
		sockMonitorfd = 0;
	}
	if (port)
	{
		CreateThread(NULL, 0, ThreadListenProcess, NULL, 0, NULL);
	}
	Sleep(10); // Give some time for the thread to start running
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


/*
Monitor the physical mouse left button state
Return value:
-1: Monitoring not enabled yet. You need to call kmNet_monitor(1) first.
 0: Physical mouse left button is released
 1: Physical mouse left button is pressed
*/
int kmNet_monitor_mouse_left()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x01) ? 1 : 0;
}

/*
Monitor the physical mouse middle button state
Return value:
-1: Monitoring not enabled yet. You need to call kmNet_monitor(1) first.
 0: Physical mouse middle button is released
 1: Physical mouse middle button is pressed
*/
int kmNet_monitor_mouse_middle()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x04) ? 1 : 0;
}

/*
Monitor the physical mouse right button state
Return value:
-1: Monitoring not enabled yet. You need to call kmNet_monitor(1) first.
 0: Physical mouse right button is released
 1: Physical mouse right button is pressed
*/
int kmNet_monitor_mouse_right()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x02) ? 1 : 0;
}

/*
Monitor the physical mouse side button 1 state
Return value:
-1: Monitoring not enabled yet. You need to call kmNet_monitor(1) first.
 0: Physical mouse side button 1 is released
 1: Physical mouse side button 1 is pressed
*/
int kmNet_monitor_mouse_side1()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x08) ? 1 : 0;
}


/*
Monitor the physical mouse side button 2 state
Return value:
-1: Monitoring not enabled yet. You need to call kmNet_monitor(1) first.
 0: Physical mouse side button 2 is released
 1: Physical mouse side button 2 is pressed
*/
int kmNet_monitor_mouse_side2()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x10) ? 1 : 0;
}


// Monitor the specified keyboard key state
int kmNet_monitor_keyboard(short  vkey)
{
	unsigned char vk_key = vkey & 0xff;
	if (monitor_run != monitor_ok) return -1;
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI) // Control key
	{
		switch (vk_key)
		{
		case KEY_LEFTCONTROL: return  hw_keyboard.buttons & BIT0 ? 1 : 0;
		case KEY_LEFTSHIFT:   return  hw_keyboard.buttons & BIT1 ? 1 : 0;
		case KEY_LEFTALT:     return  hw_keyboard.buttons & BIT2 ? 1 : 0;
		case KEY_LEFT_GUI:    return  hw_keyboard.buttons & BIT3 ? 1 : 0;
		case KEY_RIGHTCONTROL:return  hw_keyboard.buttons & BIT4 ? 1 : 0;
		case KEY_RIGHTSHIFT:  return  hw_keyboard.buttons & BIT5 ? 1 : 0;
		case KEY_RIGHTALT:    return  hw_keyboard.buttons & BIT6 ? 1 : 0;
		case KEY_RIGHT_GUI:   return  hw_keyboard.buttons & BIT7 ? 1 : 0;
		}
	}
	else // Regular key
	{
		for (int i = 0; i < 10; i++)
		{
			if (hw_keyboard.data[i] == vk_key)
			{
				return 1;
			}
		}
	}
	return 0;
}


/*
Enable internal box debug printing and send to the specified port (for debugging)
*/
int kmNet_debug(short port, char enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_debug;              // Command
	tx.head.rand = port | enable << 16;   // Random obfuscation value
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);

}


// Block (mask) mouse left button
int kmNet_mask_mouse_left(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT0) : (mask_keyboard_mouse_flag &= ~BIT0); // Block mouse left button
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Block (mask) mouse right button
int kmNet_mask_mouse_right(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT1) : (mask_keyboard_mouse_flag &= ~BIT1); // Block mouse right button
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Block (mask) mouse middle button
int kmNet_mask_mouse_middle(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT2) : (mask_keyboard_mouse_flag &= ~BIT2); // Block mouse middle button
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Block (mask) mouse side button 1
int kmNet_mask_mouse_side1(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT3) : (mask_keyboard_mouse_flag &= ~BIT3); // Block mouse side button 1
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Block (mask) mouse side button 2
int kmNet_mask_mouse_side2(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT4) : (mask_keyboard_mouse_flag &= ~BIT4); // Block mouse side button 2
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Block (mask) mouse X-axis
int kmNet_mask_mouse_x(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT5) : (mask_keyboard_mouse_flag &= ~BIT5); // Block mouse X-axis
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Block (mask) mouse Y-axis
int kmNet_mask_mouse_y(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT6) : (mask_keyboard_mouse_flag &= ~BIT6); // Block mouse Y-axis
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}

// Block (mask) mouse wheel
int kmNet_mask_mouse_wheel(int enable)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT7) : (mask_keyboard_mouse_flag &= ~BIT7); // Block mouse wheel
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Block (mask) the specified keyboard key
int kmNet_mask_keyboard(short vkey)
{
	int err;
	BYTE v_key = vkey & 0xff;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_mask_mouse;         // Command
	tx.head.rand = (mask_keyboard_mouse_flag & 0xff) | (v_key << 8); // Mask keyboard vkey
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Unblock the specified keyboard key
int kmNet_unmask_keyboard(short vkey)
{
	int err;
	BYTE v_key = vkey & 0xff;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_unmask_all;         // Command
	tx.head.rand = (mask_keyboard_mouse_flag & 0xff) | (v_key << 8); // Unmask keyboard vkey
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Unblock all previously set physical blocks
int kmNet_unmask_all()
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_unmask_all;         // Command
	mask_keyboard_mouse_flag = 0;
	tx.head.rand = mask_keyboard_mouse_flag;
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Set configuration info (change IP and port)
int kmNet_setconfig(char* ip, unsigned short port)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	tx.head.indexpts++;                   // Command statistics value
	tx.head.cmd = cmd_setconfig;          // Command
	tx.head.rand = inet_addr(ip);
	tx.u8buff.buff[0] = port >> 8;
	tx.u8buff.buff[1] = port >> 0;
	int length = sizeof(cmd_head_t) + 2;
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (err < 0)
		return err_net_rx_timeout;
	return NetRxReturnHandle(&rx, &tx);
}


// Fill the entire LCD screen with the specified color. Use black for clearing the screen.
int kmNet_lcd_color(unsigned short rgb565)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	for (int y = 0; y < 40; y++)
	{
		tx.head.indexpts++;           // Command statistics value
		tx.head.cmd = cmd_showpic;    // Command
		tx.head.rand = 0 | y * 4;
		for (int c = 0; c < 512; c++)
			tx.u16buff.buff[c] = rgb565;
		int length = sizeof(cmd_head_t) + 1024;
		sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
		SOCKADDR_IN sclient;
		int clen = sizeof(sclient);
		err = recvfrom(sockClientfd, (char*)&rx, length, 0, (struct sockaddr*)&sclient, &clen);
		if (err < 0)
			return err_net_rx_timeout;
	}
	return NetRxReturnHandle(&rx, &tx);

}

// Display a 128x80 image at the bottom
int kmNet_lcd_picture_bottom(unsigned char* buff_128_80)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	for (int y = 0; y < 20; y++)
	{
		tx.head.indexpts++;           // Command statistics value
		tx.head.cmd = cmd_showpic;    // Command
		tx.head.rand = 80 + y * 4;
		memcpy(tx.u8buff.buff, &buff_128_80[y * 1024], 1024);
		int length = sizeof(cmd_head_t) + 1024;
		sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
		SOCKADDR_IN sclient;
		int clen = sizeof(sclient);
		err = recvfrom(sockClientfd, (char*)&rx, length, 0, (struct sockaddr*)&sclient, &clen);
		if (err < 0)
			return err_net_rx_timeout;
	}
	return NetRxReturnHandle(&rx, &tx);
}

// Display a 128x160 image at the bottom
int kmNet_lcd_picture(unsigned char* buff_128_160)
{
	int err;
	if (sockClientfd <= 0)        return err_creat_socket;
	for (int y = 0; y < 40; y++)
	{
		tx.head.indexpts++;           // Command statistics value
		tx.head.cmd = cmd_showpic;    // Command
		tx.head.rand = y * 4;
		memcpy(tx.u8buff.buff, &buff_128_160[y * 1024], 1024);
		int length = sizeof(cmd_head_t) + 1024;
		sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
		SOCKADDR_IN sclient;
		int clen = sizeof(sclient);
		err = recvfrom(sockClientfd, (char*)&rx, length, 0, (struct sockaddr*)&sclient, &clen);
		if (err < 0)
			return err_net_rx_timeout;
	}
	return NetRxReturnHandle(&rx, &tx);
}

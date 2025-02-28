
#include "kmboxNet.h"
#include "HidTable.h"
#include <time.h>
#include "my_enc.h"
#define monitor_ok    2
#define monitor_exit  0
unsigned int xbox_mac = 0;  // how to call this globally?
SOCKET sockClientfd  = 0;				//键鼠网络通信句柄
SOCKET sockMonitorfd = 0;				//监听网络通信句柄
client_tx tx;							//发送的内容
client_tx rx;							//接收的内容
SOCKADDR_IN addrSrv;
soft_mouse_t    softmouse;				//软件鼠标数据
soft_keyboard_t softkeyboard;			//软件键盘数据
static int monitor_run = 0;				//物理键鼠监控是否运行
static int mask_keyboard_mouse_flag = 0;//键鼠屏蔽状态
static short monitor_port = 0;
static unsigned char key[16] = { 0 };	//加密密钥
static HANDLE m_hMutex_lock = NULL;	    //多线程互斥锁
extern unsigned int xbox_mac;			//xboxAPI也需要mac
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
	unsigned char buttons;      // 8 buttons控制键
	unsigned char data[10];     //常规按键
}standard_keyboard_report_t;
#pragma pack()

standard_mouse_report_t		hw_mouse;   //硬件鼠标消息
standard_keyboard_report_t	hw_keyboard;//硬件键盘消息



HANDLE hMutex_busy()
{
	return m_hMutex_lock;
}

SOCKADDR_IN *GetSocketAddress()
{
	return &addrSrv;
}
SOCKET GetSocketFd()
{
	return sockClientfd;
}


//生成一个A到B之间的随机数
int myrand(int a, int b)
{
	int min = a < b ? a : b;
	int max = a > b ? a : b;
	if (a == b) return a;
	return ((rand() % (max - min)) + min);
}

unsigned int StrToHex(char* pbSrc, int nLen)
{	char h1, h2;
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

int NetRxReturnHandle(client_tx* rx, client_tx* tx)		 //接收的内容
{
	int ret = 0;
	if (rx->head.cmd != tx->head.cmd)
		ret=  err_net_cmd;//命令码错误
	if (rx->head.indexpts != tx->head.indexpts)
		ret=  err_net_pts;//时间戳错误
	ReleaseMutex(m_hMutex_lock);
	return ret;				//没有错误返回0
}


/*
连接kmboxNet盒子输入参数分别是盒子
ip  ：盒子的IP地址 （显示屏上会有显示,例如：192.168.2.88）
port: 通信端口号   （显示屏上会有显示，例如：6234）
mac : 盒子的mac地址（显示屏幕上有显示，例如：12345）
返回值:0正常，非零值请看错误代码
*/
int kmNet_init(char* ip, char* port, char* mac)
{
	WORD wVersionRequested;WSADATA wsaData;	int err;
	wVersionRequested = MAKEWORD(1, 1);
	err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) 		return err_creat_socket;
	if (LOBYTE(wsaData.wVersion) != 1 || HIBYTE(wsaData.wVersion) != 1) {
		WSACleanup(); sockClientfd = -1;
		return err_net_version;
	}

	if (m_hMutex_lock == NULL)
	{
#if __UNICODE
		m_hMutex_lock = CreateMutex(NULL, TRUE, (LPCSTR)"busy");//非UNICODE 字符集
#else 
		m_hMutex_lock = CreateMutex(NULL, TRUE, (LPCWSTR)"busy");//UNICODE 字符集
#endif 
	}
	ReleaseMutex(m_hMutex_lock);
	memset(key, 0, 16);
	srand((unsigned)time(NULL));
	sockClientfd = socket(AF_INET, SOCK_DGRAM, 0);
	addrSrv.sin_addr.S_un.S_addr = inet_addr(ip);
	addrSrv.sin_family = AF_INET;
	addrSrv.sin_port = htons(atoi(port));//端口UUID[1]>>16高16位
	tx.head.mac = StrToHex(mac, 4);		 //盒子的mac 固定 UUID
	xbox_mac = tx.head.mac;				 //记录mac值
	tx.head.rand = rand();				 //随机值。后续可用于网络数据包加密。避免特征。先预留
	tx.head.indexpts = 0;				 //指令统计值
	tx.head.cmd = cmd_connect;			 //指令
	memset(&softmouse, 0, sizeof(softmouse));	//软件鼠标数据清零
	memset(&softkeyboard, 0, sizeof(softkeyboard));//软件鼠标数据清零
	key[0] = tx.head.mac >> 24; key[1] = tx.head.mac >> 16;//设置加密密钥
	key[2] = tx.head.mac >> 8; key[3] = tx.head.mac  >> 0; //设置加密密钥
	err = sendto(sockClientfd, (const char*)&tx, sizeof(cmd_head_t), 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	Sleep(20);//第一次连接可能时间比较久
	int clen = sizeof(addrSrv);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&addrSrv, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

/*
鼠标移动x,y个单位。一次性移动。无轨迹模拟，速度最快.
自己写轨迹移动时使用此函数。
返回值：0正常执行，其他值异常。
*/
int kmNet_mouse_move(short x, short y)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_move;	//指令
	tx.head.rand = rand();			//随机混淆值
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
	return NetRxReturnHandle(&rx, &tx);
}

/*
带加密功能的控制
鼠标移动x,y个单位。一次性移动。无轨迹模拟，速度最快.
自己写轨迹移动时使用此函数。
返回值：0正常执行，其他值异常。
此函数是带加密功能的，可以保证同一个移动指令网络数据包内容都不一样。无法通过网络发码抓捕来特征盒子。
*/
int kmNet_enc_mouse_move(short x, short y)
{	int err; client_tx  tx_enc = {0};
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_move;	//
	tx.head.rand = rand();			//随机混淆值
	softmouse.x = x;
	softmouse.y = y;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);														 //加密发送数据
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv)); //
	softmouse.x = 0;
	softmouse.y = 0;
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;//只要收到回码就认为发送OK ，减少盒子的加密过程。 节约时间
}





/*
鼠标左键控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_mouse_left(int isdown)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_left;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x01) : (softmouse.button & (~0x01)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

/*
鼠标左键控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_enc_mouse_left(int isdown)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_left;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x01) : (softmouse.button & (~0x01)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}


/*
鼠标中键控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_mouse_middle(int isdown)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_middle;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x04) : (softmouse.button & (~0x04)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


/*
鼠标中键控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_enc_mouse_middle(int isdown)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_middle;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x04) : (softmouse.button & (~0x04)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}

/*
鼠标右键控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_mouse_right(int isdown)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_right;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x02) : (softmouse.button & (~0x02)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


/*
鼠标右键控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_enc_mouse_right(int isdown)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_right;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x02) : (softmouse.button & (~0x02)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}




/*
鼠标侧键1控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_mouse_side1(int isdown)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_right;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x08) : (softmouse.button & (~0x08)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

/*
鼠标侧键1控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_enc_mouse_side1(int isdown)
{
	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_right;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x08) : (softmouse.button & (~0x08)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}



/*
鼠标侧键2控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_mouse_side2(int isdown)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_right;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x10) : (softmouse.button & (~0x10)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

/*
鼠标侧键2控制
isdown :0松开 ，1按下
返回值：0正常执行，其他值异常。
*/
int kmNet_enc_mouse_side2(int isdown)
{
	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_right;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = (isdown ? (softmouse.button | 0x10) : (softmouse.button & (~0x10)));
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}



//鼠标滚轮控制
int kmNet_mouse_wheel(int wheel)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_wheel;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.wheel = wheel;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.wheel = 0;
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}



//鼠标滚轮控制
int kmNet_enc_mouse_wheel(int wheel)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_wheel;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.wheel = wheel;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.wheel = 0;
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}


/*
鼠标全报告控制函数
*/
int kmNet_mouse_all(int button, int x, int y, int wheel)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_wheel;	//指令
	tx.head.rand = rand();			//随机混淆值
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
	return NetRxReturnHandle(&rx, &tx);
}


/*
鼠标全报告控制函数
*/
int kmNet_enc_mouse_all(int button, int x, int y, int wheel)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_mouse_wheel;	//指令
	tx.head.rand = rand();			//随机混淆值
	softmouse.button = button;
	softmouse.x = x;
	softmouse.y = y;
	softmouse.wheel = wheel;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.x = 0;
	softmouse.y = 0;
	softmouse.wheel = 0;
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}



/*
鼠标移动x,y个单位。模拟人为移动x,y个单位。不会出现键鼠异常的检测.
没有写移动曲线的推荐用此函数。此函数不会出现跳跃现象，按照最小步进逼近
目标点。耗时比kmNet_mouse_move高。
ms是设置移动需要多少毫秒.注意ms给的值不要太小，太小一样会出现键鼠数据异常。
尽量像人操作。实际用时会比ms小。
*/
int kmNet_mouse_move_auto(int x, int y, int ms)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				 //指令统计值
	tx.head.cmd = cmd_mouse_automove;//指令
	tx.head.rand = ms;			     //随机混淆值
	softmouse.x = x;
	softmouse.y = y;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.x = 0;				//清零
	softmouse.y = 0;				//清零
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}




/*
鼠标移动x,y个单位。模拟人为移动x,y个单位。不会出现键鼠异常的检测.
没有写移动曲线的推荐用此函数。此函数不会出现跳跃现象，按照最小步进逼近
目标点。耗时比kmNet_mouse_move高。
ms是设置移动需要多少毫秒.注意ms给的值不要太小，太小一样会出现键鼠数据异常。
尽量像人操作。实际用时会比ms小。
*/
int kmNet_enc_mouse_move_auto(int x, int y, int ms)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				 //指令统计值
	tx.head.cmd = cmd_mouse_automove;//指令
	tx.head.rand = ms;			     //随机混淆值
	softmouse.x = x;
	softmouse.y = y;
	memcpy(&tx.cmd_mouse, &softmouse, sizeof(soft_mouse_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_mouse_t);
	softmouse.x = 0;				//清零
	softmouse.y = 0;				//清零
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}





/*
二阶贝塞尔曲线控制
x,y 	:目标点坐标
ms		:拟合此过程用时（单位ms）
x1,y1	:控制点p1点坐标
x2,y2	:控制点p2点坐标
*/
int kmNet_mouse_move_beizer(int x, int y, int ms, int x1, int y1, int x2, int y2)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;			 //指令统计值
	tx.head.cmd = cmd_bazerMove; //指令
	tx.head.rand = ms;			 //随机混淆值
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
	return NetRxReturnHandle(&rx, &tx);
}




/*
二阶贝塞尔曲线控制
x,y 	:目标点坐标
ms		:拟合此过程用时（单位ms）
x1,y1	:控制点p1点坐标
x2,y2	:控制点p2点坐标
*/
int kmNet_enc_mouse_move_beizer(int x, int y, int ms, int x1, int y1, int x2, int y2)
{	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;			 //指令统计值
	tx.head.cmd = cmd_bazerMove; //指令
	tx.head.rand = ms;			 //随机混淆值
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
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}






int kmNet_keydown(int vk_key)
{	int i;
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI)//控制键
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
	{//常规键  
		for (i = 0; i < 10; i++)//首先检查队列中是否存在vk_key
		{
			if (softkeyboard.button[i] == vk_key)
				goto KM_down_send;// 队列里面已经有vk_key 直接发送就行
		}
		//队列里面没有vk_key 
		for (i = 0; i < 10; i++)//遍历所有的数据，将vk_key添加到队列里
		{
			if (softkeyboard.button[i] == 0)
			{// 队列里面已经有vk_key 直接发送就行
				softkeyboard.button[i] = vk_key;
				goto KM_down_send;
			}
		}
		//队列已经满了 那么就剔除最开始的那个
		memcpy(&softkeyboard.button[0], &softkeyboard.button[1], 9);
		softkeyboard.button[9] = vk_key;
	}
KM_down_send:
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_keyboard_all;	//指令
	tx.head.rand = rand();			// 随机混淆值
	memcpy(&tx.cmd_keyboard, &softkeyboard, sizeof(soft_keyboard_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_keyboard_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}




int kmNet_enc_keydown(int vk_key)
{	int i; client_tx  tx_enc = { 0 };
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI)//控制键
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
	{//常规键  
		for (i = 0; i < 10; i++)//首先检查队列中是否存在vk_key
		{
			if (softkeyboard.button[i] == vk_key)
				goto KM_enc_down_send;// 队列里面已经有vk_key 直接发送就行
		}
		//队列里面没有vk_key 
		for (i = 0; i < 10; i++)//遍历所有的数据，将vk_key添加到队列里
		{
			if (softkeyboard.button[i] == 0)
			{// 队列里面已经有vk_key 直接发送就行
				softkeyboard.button[i] = vk_key;
				goto KM_enc_down_send;
			}
		}
		//队列已经满了 那么就剔除最开始的那个
		memcpy(&softkeyboard.button[0], &softkeyboard.button[1], 9);
		softkeyboard.button[9] = vk_key;
	}
KM_enc_down_send:

	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_keyboard_all;	//指令
	tx.head.rand = rand();			// 随机混淆值
	memcpy(&tx.cmd_keyboard, &softkeyboard, sizeof(soft_keyboard_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_keyboard_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}


//键盘按键松开
int kmNet_keyup(int vk_key)
{	int i;
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI)//控制键
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
	{//常规键  
		for (i = 0; i < 10; i++)//首先检查队列中是否存在vk_key
		{
			if (softkeyboard.button[i] == vk_key)// 队列里面已经有vk_key 
			{
				memcpy(&softkeyboard.button[i], &softkeyboard.button[i + 1], 9 - i);
				softkeyboard.button[9] = 0;
				goto KM_up_send;
			}
		}
	}
KM_up_send:
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_keyboard_all;	//指令
	tx.head.rand = rand();			// 随机混淆值
	memcpy(&tx.cmd_keyboard, &softkeyboard, sizeof(soft_keyboard_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_keyboard_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}




//键盘案件松开
int kmNet_enc_keyup(int vk_key)
{	int i; client_tx  tx_enc = { 0 };
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI)//控制键
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
	{//常规键  
		for (i = 0; i < 10; i++)//首先检查队列中是否存在vk_key
		{
			if (softkeyboard.button[i] == vk_key)// 队列里面已经有vk_key 
			{
				memcpy(&softkeyboard.button[i], &softkeyboard.button[i + 1], 9 - i);
				softkeyboard.button[9] = 0;
				goto KM_enc_up_send;
			}
		}
	}
KM_enc_up_send:

	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_keyboard_all;	//指令
	tx.head.rand = rand();			// 随机混淆值
	memcpy(&tx.cmd_keyboard, &softkeyboard, sizeof(soft_keyboard_t));
	int length = sizeof(cmd_head_t) + sizeof(soft_keyboard_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	ReleaseMutex(m_hMutex_lock);
	return 0;
}


//单击指定按键
int kmNet_keypress(int vk_key, int ms)
{
	kmNet_keydown(vk_key);
	Sleep(ms/2);
	kmNet_keyup(vk_key);
	Sleep(ms / 2);
	return 0;
}

//单击指定按键
int kmNet_enc_keypress(int vk_key, int ms)
{
	kmNet_enc_keydown(vk_key);
	Sleep(ms/2);
	kmNet_enc_keyup(vk_key);
	Sleep(ms/2);
	return 0;
}



//重启盒子
int kmNet_reboot(void)
{	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_reboot;		//指令
	tx.head.rand = rand();			// 随机混淆值
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	WSACleanup();
	sockClientfd = -1;
	ReleaseMutex(m_hMutex_lock);
	return 0;
}


int kmNet_enc_reboot(void)
{
	int err; client_tx  tx_enc = { 0 };
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_reboot;		//指令
	tx.head.rand = rand();			// 随机混淆值
	int length = sizeof(cmd_head_t);
	memcpy(&tx_enc, &tx, length);
	my_encrypt((unsigned char*)&tx_enc, key);
	sendto(sockClientfd, (const char*)&tx_enc, 128, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	WSACleanup();
	sockClientfd = -1;
	ReleaseMutex(m_hMutex_lock);
	return 0;
}


//监听物理键鼠 如果监听不到物理键鼠，请关闭防火墙
DWORD WINAPI ThreadListenProcess(LPVOID lpParameter)
{
	WSADATA wsaData; int ret;
	WSAStartup(MAKEWORD(1, 1), &wsaData);			//创建套接字，SOCK_DGRAM指明使用 UDP 协议
	sockMonitorfd = socket(AF_INET, SOCK_DGRAM, 0);	//绑定套接字
	sockaddr_in servAddr;
	memset(&servAddr, 0, sizeof(servAddr));			//每个字节都用0填充
	servAddr.sin_family		 = PF_INET;				//使用IPv4地址
	servAddr.sin_addr.s_addr = INADDR_ANY;	        //自动获取IP地址
	servAddr.sin_port = htons(monitor_port);		//监听端口
	ret=bind(sockMonitorfd, (SOCKADDR*)&servAddr, sizeof(SOCKADDR));
	SOCKADDR cliAddr;  //客户端地址信息
	int nSize = sizeof(SOCKADDR);
	char buff[1024];  //缓冲区
	monitor_run = monitor_ok;
	while (1) 
	{
		int ret=recvfrom(sockMonitorfd, buff, 1024, 0, &cliAddr, &nSize);	//阻塞读数据
		if (ret > 0)
		{
			memcpy(&hw_mouse, buff, sizeof(hw_mouse));							//物理鼠标状态
			memcpy(&hw_keyboard, &buff[sizeof(hw_mouse)], sizeof(hw_keyboard));	//物理键盘状态
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

//使能键鼠监控  端口号取值范围是1024到49151
int kmNet_monitor(short port)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_monitor;		//指令
	if (port){
		monitor_port = port;				//那个端口用来监听物理键鼠数据
		tx.head.rand = port | 0xaa55 << 16;	// 随机混淆值
	}
	else
		tx.head.rand = 0;	// 随机混淆值
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	if (sockMonitorfd > 0)	//关闭监听
	{	closesocket(sockMonitorfd);
		sockMonitorfd = 0;
		Sleep(100);//给点时间让线程先运行
	}
	if (port)
	{
		CreateThread(NULL, 0, ThreadListenProcess, NULL, 0, NULL);
	}
	return NetRxReturnHandle(&rx, &tx);
}


/*
监听物理鼠标左键状态
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :物理鼠标左键松开
1：物理鼠标左键按下
*/
int kmNet_monitor_mouse_left()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x01) ? 1 : 0;
}


/*//监听物理鼠标中键状态
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :物理鼠标中键松开
1：物理鼠标中键按下
*/
int kmNet_monitor_mouse_middle()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x04) ? 1 : 0;
}

/*//监听物理鼠标右键状态
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :物理鼠标右键松开
1：物理鼠标右键按下
*/
int kmNet_monitor_mouse_right()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x02) ? 1 : 0;
}


/*//监听物理鼠标侧键1状态
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :物理鼠标侧键1松开
1：物理鼠标侧键1按下
*/
int kmNet_monitor_mouse_side1()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x08) ? 1 : 0;
}

/*//监听物理鼠标侧键2状态
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :物理鼠标侧键2松开
1：物理鼠标侧键2按下
*/
int kmNet_monitor_mouse_side2()
{
	if (monitor_run != monitor_ok) return -1;
	return (hw_mouse.buttons & 0x10) ? 1 : 0;
}

/*监听物理鼠标XY坐标值
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :xy坐标没有发生变化
1：xy坐标发生了变化
*/
int kmNet_monitor_mouse_xy(int *x,int *y)
{
	static int lastx, lasty;
	if (monitor_run != monitor_ok) return -1;
	*x = hw_mouse.x;
	*y = hw_mouse.y;
	if (*x != lastx || *y != lasty)
	{
		lastx = *x;
		lasty = *y;
		return 1;
	}
	return 0;
}



/*监听物理鼠标滚轮坐标值
返回值
-1：还未开启监听功能 需要先调用kmNet_monitor（1）
0 :xy坐标没有发生变化
1：xy坐标发生了变化
*/
int kmNet_monitor_mouse_wheel(int* wheel)
{
	static int lastwheel;
	if (monitor_run != monitor_ok) return -1;
	*wheel = hw_mouse.wheel;
	if (*wheel != lastwheel )
	{
		lastwheel = *wheel;
		return 1;
	}
	return 0;
}



//监听键盘指定按键状态
int kmNet_monitor_keyboard(short  vkey)
{
	unsigned char vk_key = vkey & 0xff;
	if (monitor_run != monitor_ok) return -1;
	if (vk_key >= KEY_LEFTCONTROL && vk_key <= KEY_RIGHT_GUI)//控制键
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
	else//常规键
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




//屏蔽鼠标左键 
int kmNet_mask_mouse_left(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT0) : (mask_keyboard_mouse_flag &= ~BIT0);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

//屏蔽鼠标右键 
int kmNet_mask_mouse_right(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT1) : (mask_keyboard_mouse_flag &= ~BIT1);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//屏蔽鼠标中键 
int kmNet_mask_mouse_middle(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT2) : (mask_keyboard_mouse_flag &= ~BIT2);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//屏蔽鼠标侧键键1 
int kmNet_mask_mouse_side1(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT3) : (mask_keyboard_mouse_flag &= ~BIT3);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}



//屏蔽鼠标侧键键2
int kmNet_mask_mouse_side2(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT4) : (mask_keyboard_mouse_flag &= ~BIT4);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//屏蔽鼠标X轴坐标
int kmNet_mask_mouse_x(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT5) : (mask_keyboard_mouse_flag &= ~BIT5);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//屏蔽鼠标y轴坐标
int kmNet_mask_mouse_y(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT6) : (mask_keyboard_mouse_flag &= ~BIT6);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

//屏蔽鼠标滚轮
int kmNet_mask_mouse_wheel(int enable)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = enable ? (mask_keyboard_mouse_flag |= BIT7) : (mask_keyboard_mouse_flag &= ~BIT7);	// 屏蔽鼠标左键
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//屏蔽键盘指定按键
int kmNet_mask_keyboard(short vkey)
{
	int err;
	BYTE v_key = vkey & 0xff;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_mask_mouse;		//指令
	tx.head.rand = (mask_keyboard_mouse_flag & 0xff) | (v_key << 8);	// 屏蔽键盘vkey
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//解除屏蔽键盘指定按键
int kmNet_unmask_keyboard(short vkey)
{
	int err;
	BYTE v_key = vkey & 0xff;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_unmask_all;		//指令
	tx.head.rand = (mask_keyboard_mouse_flag & 0xff) | (v_key << 8);	// 屏蔽键盘vkey
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//解除屏蔽所有已经设置的物理屏蔽
int kmNet_unmask_all()
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_unmask_all;		//指令
	mask_keyboard_mouse_flag = 0;
	tx.head.rand = mask_keyboard_mouse_flag;
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}



//设置配置信息  改IP与端口号
int kmNet_setconfig(char* ip, unsigned short port)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_setconfig;		//指令
	tx.head.rand = inet_addr(ip); ;
	tx.u8buff.buff[0] = port >> 8;
	tx.u8buff.buff[1] = port >> 0;
	int length = sizeof(cmd_head_t) + 2;
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

//设置盒子device端的VIDPID
int kmNet_setvidpid(unsigned short vid,unsigned short pid)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;					//指令统计值
	tx.head.cmd = cmd_setvidpid;		//指令
	tx.head.rand = vid| pid<<16;
	int length = sizeof(cmd_head_t);
	sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}


//将整个LCD屏幕用指定颜色填充。 清屏可以用黑色
int kmNet_lcd_color(unsigned short rgb565)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	for (int y = 0; y < 40; y++)
	{
		tx.head.indexpts++;		    //指令统计值
		tx.head.cmd = cmd_showpic;	//指令
		tx.head.rand = 0 | y * 4;
		for (int c = 0;c < 512;c++)
			tx.u16buff.buff[c] = rgb565;
		int length = sizeof(cmd_head_t) + 1024;
		sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
		SOCKADDR_IN sclient;
		int clen = sizeof(sclient);
		err = recvfrom(sockClientfd, (char*)&rx, length, 0, (struct sockaddr*)&sclient, &clen);
	}
	return NetRxReturnHandle(&rx, &tx);

}

//在底部显示一张128x80的图片
int kmNet_lcd_picture_bottom(unsigned char* buff_128_80)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	for (int y = 0; y < 20; y++)
	{
		tx.head.indexpts++;		    //指令统计值
		tx.head.cmd = cmd_showpic;	//指令
		tx.head.rand = 80 + y * 4;
		memcpy(tx.u8buff.buff, &buff_128_80[y * 1024], 1024);
		int length = sizeof(cmd_head_t) + 1024;
		sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
		SOCKADDR_IN sclient;
		int clen = sizeof(sclient);
		err = recvfrom(sockClientfd, (char*)&rx, length, 0, (struct sockaddr*)&sclient, &clen);
	}
	return NetRxReturnHandle(&rx, &tx);
}

//在底部显示一张128x160的图片
int kmNet_lcd_picture(unsigned char* buff_128_160)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	for (int y = 0; y < 40; y++)
	{
		tx.head.indexpts++;		    //指令统计值
		tx.head.cmd = cmd_showpic;	//指令
		tx.head.rand = y * 4;
		memcpy(tx.u8buff.buff, &buff_128_160[y * 1024], 1024);
		int length = sizeof(cmd_head_t) + 1024;
		sendto(sockClientfd, (const char*)&tx, length, 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
		SOCKADDR_IN sclient;
		int clen = sizeof(sclient);
		err = recvfrom(sockClientfd, (char*)&rx, length, 0, (struct sockaddr*)&sclient, &clen);
	}
	return NetRxReturnHandle(&rx, &tx);
}

//使能盒子的硬件修正功能
/*
type  :0:贝塞尔曲线  1：导弹追踪曲线,2:贝塞尔实时 ，3:RM-RT
value ：小于或者等与0表示关闭硬件曲线修正功能。大于0表示开启硬件曲线修正功能。数值越大则曲线越平滑。但耗时越高。推荐取值范围16到50之间。最大可以到100.
*/
int kmNet_Trace(int type, int value)
{
	int err;
	if (sockClientfd <= 0)		return err_creat_socket;
	WaitForSingleObject(m_hMutex_lock, INFINITE);
	tx.head.indexpts++;				//指令统计值
	tx.head.cmd = cmd_trace_enable;	//指令
	tx.head.rand = type << 24 | value;			//跨度值
	sendto(sockClientfd, (const char*)&tx, sizeof(cmd_head_t), 0, (struct sockaddr*)&addrSrv, sizeof(addrSrv));
	SOCKADDR_IN sclient;
	int clen = sizeof(sclient);
	err = recvfrom(sockClientfd, (char*)&rx, 1024, 0, (struct sockaddr*)&sclient, &clen);
	return NetRxReturnHandle(&rx, &tx);
}

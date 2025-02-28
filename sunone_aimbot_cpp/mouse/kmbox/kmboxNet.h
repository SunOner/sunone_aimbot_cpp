#include <stdio.h>
#include <Winsock2.h>
#include "math.h"
#pragma warning(disable : 4996)
//命令码
#define 	cmd_connect			0xaf3c2828 //ok连接盒子
#define     cmd_mouse_move		0xaede7345 //ok鼠标移动
#define		cmd_mouse_left		0x9823AE8D //ok鼠标左键控制
#define		cmd_mouse_middle	0x97a3AE8D //ok鼠标中键控制
#define		cmd_mouse_right		0x238d8212 //ok鼠标右键控制
#define		cmd_mouse_wheel		0xffeead38 //ok鼠标滚轮控制
#define     cmd_mouse_automove	0xaede7346 //ok鼠标自动模拟人工移动控制
#define     cmd_keyboard_all    0x123c2c2f //ok键盘所有参数控制
#define		cmd_reboot			0xaa8855aa //ok盒子重启
#define     cmd_bazerMove       0xa238455a //ok鼠标贝塞尔移动
#define     cmd_monitor         0x27388020 //ok监控盒子上的物理键鼠数据
#define     cmd_debug           0x27382021 //ok开启调试信息
#define     cmd_mask_mouse      0x23234343 //ok 屏蔽物理键鼠
#define     cmd_unmask_all      0x23344343 //ok 解除屏蔽物理键鼠
#define     cmd_setconfig       0x1d3d3323 //ok 设置IP配置信息
#define     cmd_setvidpid       0xffed3232 //ok 设置device端的vidpid
#define     cmd_showpic         0x12334883 //显示图片
#define     cmd_trace_enable    0xbbcdddac //使能硬件修正功能



extern SOCKET sockClientfd; //socket通信句柄
typedef struct
{	
	unsigned int  mac;			//盒子的mac地址（必须）
	unsigned int  rand;			//随机值
	unsigned int  indexpts;		//时间戳
	unsigned int  cmd;			//指令码
}cmd_head_t;

typedef struct
{
	unsigned char buff[1024];	//
}cmd_data_t;

typedef struct
{
	unsigned short buff[512];	//
}cmd_u16_t;

//鼠标数据结构体
typedef struct
{
	int button;
	int x;
	int y;
	int wheel;
	int point[10];//用于贝塞尔曲线控制(预留5阶导)
}soft_mouse_t;

//键盘数据结构体
typedef struct
{
	char ctrl;
	char resvel;
	char button[10];
}soft_keyboard_t;

//联合体
typedef struct
{
	cmd_head_t head;
	union {
		cmd_data_t      u8buff;		  //buff
		cmd_u16_t       u16buff;	  //U16
		soft_mouse_t    cmd_mouse;    //鼠标发送指令
		soft_keyboard_t cmd_keyboard; //键盘发送指令
	};
}client_tx;

enum
{
	err_creat_socket = -9000,	//创建socket失败
	err_net_version,			//socket版本错误
	err_net_tx,					//socket发送错误
	err_net_rx_timeout,			//socket接收超时
	err_net_cmd,				//命令错误
	err_net_pts,				//时间戳错误
	success = 0,				//正常执行
	usb_dev_tx_timeout,			//USB devic发送失败
};



/*
连接kmboxNet盒子输入参数分别是盒子
ip  ：盒子的IP地址 （显示屏上会有显示）
port: 通信端口号   （显示屏上会有显示）
mac : 盒子的mac地址（显示屏幕上有显示）
返回值：连接成功返回0 ，其他值参见错误代码
*/
int kmNet_init(char* ip, char* port, char* mac);//ok
int kmNet_mouse_move(short x, short y);			//ok
int kmNet_mouse_left(int isdown);				//ok
int kmNet_mouse_right(int isdown);				//ok
int kmNet_mouse_middle(int isdown);				//ok
int kmNet_mouse_wheel(int wheel);				//ok
int kmNet_mouse_side1(int isdown);				//ok
int kmNet_mouse_side2(int isdown);				//ok
int kmNet_mouse_all(int button, int x, int y, int wheel);//ok
int kmNet_mouse_move_auto(int x, int y, int time_ms);	//ok
int kmNet_mouse_move_beizer(int x, int y, int ms, int x1, int y1, int x2, int y2);//二阶曲线
//键盘函数
int kmNet_keydown(int vkey);// ok
int kmNet_keyup(int vkey);  // ok
int kmNet_keypress(int vk_key, int ms =10);	//ok
int kmNet_enc_keydown(int vkey);// ok
int kmNet_enc_keyup(int vkey);  // ok
int kmNet_enc_keypress(int vk_key, int ms = 10);//ok


//带加密功能的函数。单机推荐使用，防止网络数据包抓包特征
int kmNet_enc_mouse_move(short x, short y);	    //加密类型的鼠标移动 ok
int kmNet_enc_mouse_left(int isdown);				//ok
int kmNet_enc_mouse_right(int isdown);				//ok
int kmNet_enc_mouse_middle(int isdown);				//ok
int kmNet_enc_mouse_wheel(int wheel);				//ok
int kmNet_enc_mouse_side1(int isdown);				//ok
int kmNet_enc_mouse_side2(int isdown);				//ok
int kmNet_enc_mouse_all(int button, int x, int y, int wheel);//ok
int kmNet_enc_mouse_move_auto(int x, int y, int time_ms);	//ok
int kmNet_enc_mouse_move_beizer(int x, int y, int ms, int x1, int y1, int x2, int y2);//二阶曲线
int kmNet_enc_keydown(int vkey);// ok
int kmNet_enc_keyup(int vkey);  // ok


//监控系列
int kmNet_monitor(short port);			//开启关闭物理键鼠监控
int kmNet_monitor_mouse_left();			//查询物理鼠标左键状态
int kmNet_monitor_mouse_middle();		//查询鼠标中键状态
int kmNet_monitor_mouse_right();		//查询鼠标右键状态
int kmNet_monitor_mouse_side1();		//查询鼠标侧键1状态
int kmNet_monitor_mouse_side2();		//查询鼠标侧键2状态 
int kmNet_monitor_mouse_xy(int* x, int* y);//查询鼠标xy坐标值
int kmNet_monitor_mouse_wheel(int* wheel);//查询鼠标滚轮值
int kmNet_monitor_keyboard(short  vk_key);//查询键盘指定按键状态
//物理键鼠屏蔽系列
int kmNet_mask_mouse_left(int enable);	//屏蔽鼠标左键 
int kmNet_mask_mouse_right(int enable);	//屏蔽鼠标右键 
int kmNet_mask_mouse_middle(int enable);//屏蔽鼠标中键 
int kmNet_mask_mouse_side1(int enable);	//屏蔽鼠标侧键键1 
int kmNet_mask_mouse_side2(int enable);	//屏蔽鼠标侧键键2
int kmNet_mask_mouse_x(int enable);		//屏蔽鼠标X轴坐标
int kmNet_mask_mouse_y(int enable);		//屏蔽鼠标y轴坐标
int kmNet_mask_mouse_wheel(int enable);	//屏蔽鼠标滚轮
int kmNet_mask_keyboard(short vkey);	//屏蔽键盘指定按键
int kmNet_unmask_keyboard(short vkey);	//解除键盘指定按键
int kmNet_unmask_all();					//解除屏蔽所有已经设置的物理屏蔽


//配置类函数
int kmNet_reboot(void);									  //重启盒子
int kmNet_setconfig(char* ip, unsigned short port);		  //配置盒子IP地址
int kmNet_setvidpid(unsigned short vid,unsigned short pid);//设置盒子VIDPID 设置后需要重启盒子。下次上电才能生效
int kmNet_debug(short port, char enable);				  //调试使能
int kmNet_lcd_color(unsigned short rgb565);				  //将整个LCD屏幕用指定颜色填充。 清屏可以用黑色
int kmNet_lcd_picture_bottom(unsigned char* buff_128_80); //下半部分显示128x80图片
int kmNet_lcd_picture(unsigned char* buff_128_160);		  //整屏显示128x160图片

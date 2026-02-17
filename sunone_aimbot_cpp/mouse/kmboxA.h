#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned int Onoff;
    unsigned int StartAddr;
    unsigned int Length;
    unsigned int RunCnt;
    int Exist;
    char Name[12];
} script_detail_t;

typedef struct {
    unsigned int NewBoardFlag;
    unsigned int defaultVID;
    unsigned int defaultPID;
    unsigned int TotalSize;
    unsigned int UsedSize;
    unsigned int NowIndex;
    script_detail_t script[5];
} kmbox_t;

int KM_init(unsigned short vid, unsigned short pid);
int KM_close(void);

int KM_press(unsigned char vk_key);
int KM_down(unsigned char vk_key);
int KM_up(unsigned char vk_key);
int KM_keyboard(unsigned char ctrButton, unsigned char* key);

int KM_mouse(unsigned char lmr_side, short x, short y, unsigned char wheel);
int KM_left(unsigned char vk_key);
int KM_middle(unsigned char vk_key);
int KM_right(unsigned char vk_key);
int KM_move(short x, short y);
int KM_wheel(unsigned char w);
int KM_side1(unsigned char w);
int KM_side2(unsigned char w);

int KM_GetRegcode(unsigned char* outMac);
int KM_SetRegcode(char* skey);

int KM_LCDstr(int mode, char* str, int x, int y);
int KM_LCDpic(unsigned char* bmp);

int KM_ERASE(void);
int KM_UserData(int rw, unsigned char* buff);
int KM_Readscript(kmbox_t* km);
int KM_SetVIDPID(int VID, int PID);

int KM_WriteScript(char* name, int index, int trigger, int doneNext, int Switch, char* str);
int KM_HostVidpid(int rw, unsigned int* vidpid, unsigned int* hiddid, unsigned int* mtype);

int MakeKey(char* mac, char* key);

#ifdef __cplusplus
}
#endif

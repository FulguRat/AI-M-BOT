// pch.cpp: source file corresponding to the pre-compiled header
// When you are using pre-compiled headers, this source file is necessary for compilation to succeed.

#include "pch.h"
#include "win_sdipmk.h"


struct ScrnWH { long width, height; };
typedef struct ScrnWH Struct;


Struct getscrnwh()
{
	Struct s;
	DEVMODEA dm;
	ZeroMemory(&dm, sizeof(dm));
	dm.dmSize = sizeof(DEVMODE);
	EnumDisplaySettingsA(NULL, ENUM_CURRENT_SETTINGS, &dm);
	s.width = dm.dmPelsWidth;
	s.height = dm.dmPelsHeight;
	return s;
}


BOOL Agulll() { return 1; }  // mouse_open


void Shwaji() {}  // mouse_close


void Mach_Move(int x, int y, bool abs)  // moveR
{
	INPUT ip;
	ZeroMemory(&ip, sizeof(INPUT));
	Struct screen = getscrnwh();
	ip.type = INPUT_MOUSE;
	ip.mi.mouseData = 0;
	ip.mi.time = 0;
	ip.mi.dx = (abs) ? x *= (65536 / screen.width) : x;
	ip.mi.dy = (abs) ? y *= (65536 / screen.height) : y;
	ip.mi.dwFlags = (abs) ? MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK : MOUSEEVENTF_MOVE;
	SendInput(1, &ip, sizeof(INPUT));
}


void Leo_Kick(char button)  // press
{
	INPUT ip;
	ZeroMemory(&ip, sizeof(INPUT));
	ip.type = INPUT_MOUSE;
	ip.mi.mouseData = 0;
	ip.mi.time = 0;
	ip.mi.dx = 0;
	ip.mi.dy = 0;
	ip.mi.dwFlags = (button == 1) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
	SendInput(1, &ip, sizeof(INPUT));
}


void Niman_Years(char button)  // release
{
	INPUT ip;
	ZeroMemory(&ip, sizeof(INPUT));
	ip.type = INPUT_MOUSE;
	ip.mi.mouseData = 0;
	ip.mi.time = 0;
	ip.mi.dx = 0;
	ip.mi.dy = 0;
	ip.mi.dwFlags = (button == 1) ? MOUSEEVENTF_LEFTUP : MOUSEEVENTF_RIGHTUP;
	SendInput(1, &ip, sizeof(INPUT));
}


void Mebiuspin(char wheel)  // scroll
{
	INPUT ip;
	ZeroMemory(&ip, sizeof(INPUT));
	ip.type = INPUT_MOUSE;
	ip.mi.mouseData = (DWORD)-wheel*120;
	ip.mi.time = 0;
	ip.mi.dx = 0;
	ip.mi.dy = 0;
	ip.mi.dwFlags = MOUSEEVENTF_WHEEL;
	SendInput(1, &ip, sizeof(INPUT));
}


void Orb_Ground(char key)  // key_down
{
	INPUT ip;
	ZeroMemory(&ip, sizeof(INPUT));
	ip.type = INPUT_KEYBOARD;
	ip.ki.wScan = 0;
	ip.ki.time = 0;
	ip.ki.dwExtraInfo = 0;
	ip.ki.wVk = key;
	ip.ki.dwFlags = 0;  //0为按下
	SendInput(1, &ip, sizeof(INPUT));
}


void Zestium_Upper(char key)  // key_up
{
	INPUT ip;
	ZeroMemory(&ip, sizeof(INPUT));
	ip.type = INPUT_KEYBOARD;
	ip.ki.wScan = 0;
	ip.ki.time = 0;
	ip.ki.dwExtraInfo = 0;
	ip.ki.wVk = key;
	ip.ki.dwFlags = KEYEVENTF_KEYUP;
	SendInput(1, &ip, sizeof(INPUT));
}

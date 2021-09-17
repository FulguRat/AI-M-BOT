#include "pch.h"
#include "load_msdk.h"
#include "msdk.h"
#pragma comment (lib,"msdk.lib")
#include <iostream>

HANDLE m_hdl;


BOOL Check()
{
	unsigned char *userdata = (unsigned char*)"something";
	int verified = M_VerifyUserData(m_hdl, strlen((const char*)userdata), (unsigned char*)userdata);
	return verified == 0;
}


BOOL Agulll()  // mouse_open
{
	m_hdl = M_ScanAndOpen();
	int pid = M_GetVidPid(m_hdl, 2);
	if (not Check())
	{
		std::cout << "飞易来测试验证失败\n";
		pid = 0
	}
	return pid > 0;
}


void Shwaji()
{
	M_Close(m_hdl);
}


void WIN_API Mach_Move(int x, int y, bool abs)  // moveR
{
	if (abs)
	{
		M_ResetMousePos(m_hdl);
		M_MoveTo(m_hdl, x, y);
	}
	else
	{
		M_MoveR(m_hdl, x, y);
	}
}


void WIN_API Leo_Kick(char button)  // press
{
	if (button == 1) { M_LeftDown(m_hdl); }
	else if (button == 2) { M_RightDown(m_hdl); }
}


void WIN_API Niman_Years(char button)  // release
{
	if (button == 1) { M_LeftUp(m_hdl); }
	else if (button == 2) { M_RightUp(m_hdl); }
}


void WIN_API Mebiuspin(char wheel)  // scroll
{
	M_MouseWheel(m_hdl, -wheel);  // 正为下,负为上
}


void WIN_API Orb_Ground(char key)  // key_down
{
	M_KeyDown2(m_hdl, key);
}


void WIN_API Zestium_Upper(char key)  // key_up
{
	M_KeyUp2(m_hdl, key);
}

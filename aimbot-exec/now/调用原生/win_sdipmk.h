#ifndef MOUSEKEYBOARD_H
#define MOUSEKEYBOARD_H


#define WIN_API __declspec(dllexport)
typedef int BOOL;

#ifdef __cplusplus
extern "C" {
#endif

	BOOL WIN_API Agulll(void);  // mouse_open
	void WIN_API Shwaji(void);  // mouse_close
	void WIN_API Mach_Move(int x, int y, bool abs);  // moveR
	void WIN_API Leo_Kick(char button);  // press
	void WIN_API Niman_Years(char button);  // release
	void WIN_API Mebiuspin(char wheel);  // scroll
	void WIN_API Orb_Ground(char key);  // key_down
	void WIN_API Zestium_Upper(char key);  // key_up

#ifdef __cplusplus
}
#endif

#endif

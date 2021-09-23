//fork from https://github.com/ekknod/logitech-cve
#include "ghub_mouse.h"
#include "pch.h"
#include <math.h>
#include <windows.h>
#include <winternl.h>
#include <iostream>
#include <string>
#pragma comment(lib, "ntdll.lib")

typedef struct {
    char button;
    char x;
    char y;
    char wheel;
    char unk1;
} MOUSE_IO;


static HANDLE g_input;
static IO_STATUS_BLOCK g_io;

BOOL g_found_mouse;


static BOOL callmouse(MOUSE_IO* buffer)
{
    IO_STATUS_BLOCK block;
    return NtDeviceIoControlFile(g_input, 0, 0, 0, &block, 0x2a2010, buffer, sizeof(MOUSE_IO), 0, 0) == 0L;
}


static NTSTATUS device_initialize(PCWSTR device_name)
{
    UNICODE_STRING name;
    OBJECT_ATTRIBUTES attr;

    RtlInitUnicodeString(&name, device_name);
    InitializeObjectAttributes(&attr, &name, 0, NULL, NULL);

    NTSTATUS status = NtCreateFile(&g_input, GENERIC_WRITE | SYNCHRONIZE, &attr, &g_io, 0,
        FILE_ATTRIBUTE_NORMAL, 0, 3, FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, 0, 0);

    return status;
}


BOOL Agulll()  // mouse_open
{
    NTSTATUS status = 0;

    if (g_input == 0) {
        std::wstring front = L"\\??\\ROOT#SYSTEM#000";
        std::wstring rear = L"#{1abc05c0-c378-41b9-9cef-df1aba82b015}";

        for (int i = 1; i < 10; i++)
        {
            std::wstring buffer = front + std::to_wstring(i) + rear;
            status = device_initialize(buffer.c_str());
            if (NT_SUCCESS(status))
            {
                g_found_mouse = 1;
                break;
            }
        }
    }
    return status == 0;
}


void Shwaji()  // mouse_close
{
    if (g_input != 0) {
        NtClose(g_input);  //ZwClose
        g_input = 0;
    }
}


void Check(char button, char x, char y, char wheel)  // mouse_move
{
    MOUSE_IO io;
    io.unk1 = 0;
    io.button = button;
    io.x = x;
    io.y = y;
    io.wheel = wheel;

    if (!callmouse(&io)) {
        Shwaji();  // mouse_close
        Agulll();  // mouse_open
    }
}


void Mach_Move(int x, int y, bool abs_move = false)  // moveR
{
    while (abs(x) > 127 || abs(y) > 127) {
        if (abs(x) > 127) {
            Check(0, int(x / abs(x)) * 127, 0, 0);
            x -= int(x / abs(x)) * 127;
        }

        if (abs(y) > 127) {
            Check(0, 0, int(y / abs(y)) * 127, 0);
            y -= int(y / abs(y)) * 127;
        }
    }
    Check(0, x, y, 0);
}


void Leo_Kick(char button)  // press
{
    Check(button, 0, 0, 0);
}


void Niman_Years(char button)  // release
{
    Check(0, 0, 0, 0);
}


void Mebiuspin(char wheel)  // scroll
{
    Check(0, 0, 0, -wheel);  //向下为正
}

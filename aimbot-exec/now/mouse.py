from ctypes import windll
from ctypes import CDLL
from time import sleep
from math import pow
from os import path


basedir = path.dirname(path.abspath(__file__))
ghubdlldir = path.join(basedir, 'ghub_mouse.dll')
windlldir = path.join(basedir, 'win_sdipmk.dll')
msdkdlldir = path.join(basedir, 'load_msdk.dll')
DDdlldir = path.join(basedir, 'DDHID64.dll')

# ↓↓↓↓↓↓↓↓↓ 调用ghub/键鼠驱动 ↓↓↓↓↓↓↓↓↓

try:
    gm = CDLL(ghubdlldir)
    gmok = gm.Agulll()
except FileNotFoundError:
    gmok = 0

try:
    msdk = CDLL(msdkdlldir)
    msdkok = msdk.Agulll()
except FileNotFoundError:
    msdkok = 0

try:
    dd = CDLL(DDdlldir)
    ddok = dd.DD_btn(0)  # 初始化
    sleep(0.1)
    dd.DD_key(100, 1)  # 关闭windows菜单
    dd.DD_key(100, 2)
except FileNotFoundError:
    ddok = 0

sdip = CDLL(windlldir)
sdipok = sdip.Agulll()

Mach_Move = gm.Mach_Move if gmok else msdk.Mach_Move if msdkok else sdip.Mach_Move
Leo_Kick = gm.Leo_Kick if gmok else msdk.Leo_Kick if msdkok else sdip.Leo_Kick
Niman_Years = gm.Niman_Years if gmok else msdk.Niman_Years if msdkok else sdip.Niman_Years
Mebiuspin = gm.Mebiuspin if gmok else msdk.Mebiuspin if msdkok else sdip.Mebiuspin
Shwaji = gm.Shwaji if gmok else msdk.Shwaji if msdkok else sdip.Shwaji

Orb_Ground = sdip.Orb_Ground if sdipok else msdk.Orb_Ground
Zestium_Upper = sdip.Orb_Ground if sdipok else msdk.Zestium_Upper


def mouse_xy(x, y, abs_move = False):
    if ddok and not(gmok or msdkok):
        if not abs_move:
            while (abs(x) > 127 or abs(y) > 127):
                if abs(x) > 127:
                    dd.DD_movR(int(abs(x)/x*127), 0)
                    x -= abs(x)/x*127
                if abs(y) > 127:
                    dd.DD_movR(0, int(abs(y)/y*127))
                    y -= abs(y)/y*127
        return dd.DD_mov(int(x), int(y)) if abs_move else dd.DD_movR(int(x), int(y))
    return Mach_Move(int(x), int(y), abs_move)


def mouse_down(key = 1):
    if ddok and not(gmok or msdkok):
        return dd.DD_btn(int(pow(key, 2)))
    return Leo_Kick(int(key))


def mouse_up(key = 1):
    if ddok and not(gmok or msdkok):
        return dd.DD_btn(int(pow(key, 2)*2))
    return Niman_Years(key)


def scroll(num = 1):
    if ddok and not(gmok or msdkok):
        for i in range(abs(num)):
            dd.DD_whl(1 if num < 0 else 2)
        return
    return Mebiuspin(int(num))


def mouse_close():
    try:
        return Shwaji()
    except OSError:
        pass


def key_down(key):
    if ddok and not(gmok or msdkok):
        return dd.DD_key(DD_keycode(key), 1)
    if type(key) == str and len(key) == 1:  # 如果不是str就不会检查第二个条件
        return Orb_Ground(char2vk(key))
    elif isinstance(key, int):
        return Orb_Ground(key)


def key_up(key):
    if ddok and not(gmok or msdkok):
        return dd.DD_key(DD_keycode(key), 2)
    if type(key) == str and len(key) == 1:  # 如果不是str就不会检查第二个条件
        return Zestium_Upper(char2vk(key))
    elif isinstance(key, int):
        return Zestium_Upper(key)

# ↑↑↑↑↑↑↑↑↑ 调用ghub/键鼠驱动 ↑↑↑↑↑↑↑↑↑


# 将部分按键转换为虚拟键值
def char2vk(c):
    try:
        return windll.user32.VkKeyScanW(ord(c)) & 0xFF
    except TypeError:
        return 0


def DD_keycode(c):
    if len(c) > 1:
        return 0
    key_code = {
        '0': 210,
        '1': 201,
        '2': 202,
        '3': 203,
        '4': 204,
        '5': 205,
        '6': 206,
        '7': 207,
        '8': 208,
        '9': 209,
        'a': 401,
        'b': 505,
        'c': 503,
        'd': 403,
        'e': 303,
        'f': 404,
        'g': 405,
        'h': 406,
        'i': 308,
        'j': 407,
        'k': 408,
        'l': 409,
        'm': 507,
        'n': 506,
        'o': 309,
        'p': 310,
        'q': 301,
        'r': 304,
        's': 402,
        't': 305,
        'u': 307,
        'v': 504,
        'w': 302,
        'x': 502,
        'y': 306,
        'z': 501,
    }.get(c.lower(), 0)
    return key_code


"""
键盘按键和键盘对应代码表:
数字键盘 1 <--------> 96 数字键盘 2 <--------> 97 数字键盘 3 <--------> 98
数字键盘 4 <--------> 99 数字键盘 5 <--------> 100 数字键盘 6 <--------> 101
数字键盘 7 <--------> 102 数字键盘 8 <--------> 103 数字键盘 9 <--------> 104
数字键盘 0 <--------> 105
乘号 <--------> 106 加号 <--------> 107 Enter <--------> 108 减号 <--------> 109
小数点 <--------> 110 除号 <--------> 111
F1 <--------> 112 F2 <--------> 113 F3 <--------> 114 F4 <--------> 115
F5 <--------> 116 F6 <--------> 117 F7 <--------> 118 F8 <--------> 119
F9 <--------> 120 F10 <--------> 121 F11 <--------> 122 F12 <--------> 123
F13 <--------> 124 F14 <--------> 125 F15 <--------> 126
Backspace <--------> 8
Tab <--------> 9
Clear <--------> 12
Enter <--------> 13
Shift <--------> 16
Control <--------> 17
Alt <--------> 18
Caps Lock <--------> 20
Esc <--------> 27
空格键 <--------> 32
Page Up <--------> 33
Page Down <--------> 34
End <--------> 35
Home <--------> 36
左箭头 <--------> 37
向上箭头 <--------> 38
右箭头 <--------> 39
向下箭头 <--------> 40
Insert <--------> 45
Delete <--------> 46
Help <--------> 47
Num Lock <--------> 144
; : <--------> 186
= + <--------> 187
- _ <--------> 189
/ ? <--------> 191
` ~ <--------> 192
[ { <--------> 219
| <--------> 220
] } <--------> 221
'' ' <--------> 222
"""

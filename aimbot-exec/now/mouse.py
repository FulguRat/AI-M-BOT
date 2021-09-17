from ctypes import windll
from ctypes import CDLL
from os import path


basedir = path.dirname(path.abspath(__file__))
ghubdlldir = path.join(basedir, 'ghub_mouse.dll')
windlldir = path.join(basedir, 'win_sdipmk.dll')
msdkdlldir = path.join(basedir, 'load_msdk.dll')

# ↓↓↓↓↓↓↓↓↓ 调用ghub/键鼠驱动 ↓↓↓↓↓↓↓↓↓

gm = CDLL(ghubdlldir)
gmok = gm.Agulll()

sdip = CDLL(windlldir)
sdipok = sdip.Agulll()

msdk = CDLL(msdkdlldir)
msdkok = msdk.Agulll()

Mach_Move = gm.Mach_Move if gmok else msdk.Mach_Move if msdkok else sdip.Mach_Move
Leo_Kick = gm.Leo_Kick if gmok else msdk.Leo_Kick if msdkok else sdip.Leo_Kick
Niman_Years = gm.Niman_Years if gmok else msdk.Niman_Years if msdkok else sdip.Niman_Years
Mebiuspin = gm.Mebiuspin if gmok else msdk.Mebiuspin if msdkok else sdip.Mebiuspin
Shwaji = gm.Shwaji if gmok else msdk.Shwaji if msdkok else sdip.Shwaji

Orb_Ground = sdip.Orb_Ground if sdipok else msdk.Orb_Ground
Zestium_Upper = sdip.Orb_Ground if sdipok else msdk.Zestium_Upper


def mouse_xy(x, y, abs_move = False):
    return Mach_Move(int(x), int(y), abs_move)


def mouse_down(key = 1):
    return Leo_Kick(int(key))


def mouse_up(key = 1):
    return Niman_Years(key)


def scroll(num = 1):
    return Mebiuspin(int(num))


def mouse_close():
    try:
        return Shwaji()
    except OSError:
        pass


def key_down(key):
    if type(key) == str and len(key) == 1:  # 如果不是str就不会检查第二个条件
        return Orb_Ground(char2vk(key))
    elif isinstance(key, int):
        return Orb_Ground(key)


def key_up(key):
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

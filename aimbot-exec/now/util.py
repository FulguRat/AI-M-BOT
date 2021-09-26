from win32api import EnumDisplaySettings
from sys import exit, executable
from platform import release
from math import atan, tan
from ctypes import windll
from os import system
import nvidia_smi
import pywintypes
import win32gui
import pynvml


# 预加载为睡眠函数做准备
TimeBeginPeriod = windll.winmm.timeBeginPeriod
HPSleep = windll.kernel32.Sleep
TimeEndPeriod = windll.winmm.timeEndPeriod


# 简单检查gpu是否够格
def check_gpu():
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认卡1
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        pynvml.nvmlShutdown()
    except FileNotFoundError as e:
        # pynvml.nvml.NVML_ERROR_LIBRARY_NOT_FOUND
        print(str(e))
        nvidia_smi.nvmlInit()
        gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # 默认卡1
        gpu_name = nvidia_smi.nvmlDeviceGetName(gpu_handle)
        memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
        nvidia_smi.nvmlShutdown()
    if b'RTX' in gpu_name:
        return 2
    memory_total = memory_info.total / 1024 / 1024
    if memory_total > 3000:
        return 1
    return 0


# 高DPI感知
def set_dpi():
    if int(release()) >= 7:
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except AttributeError:
            windll.user32.SetProcessDPIAware()
    else:
        exit(0)


# 检测是否全屏
def is_full_screen(hWnd):
    try:
        scrnsetting = EnumDisplaySettings(None, -1)
        full_screen_wh = (scrnsetting.PelsWidth, scrnsetting.PelsHeight)
        window_rect = win32gui.GetWindowRect(hWnd)
        window_wh = (window_rect[2], window_rect[3])
        return window_wh >= full_screen_wh
    except pywintypes.error as e:
        print('全屏检测错误\n' + str(e))
        return False


# 检查是否为管理员权限
def is_admin():
    try:
        return windll.shell32.IsUserAnAdmin()
    except OSError as err:
        print('OS error: {0}'.format(err))
        return False


# 重启脚本
def restart(file_path):
    windll.shell32.ShellExecuteW(None, 'runas', executable, file_path, None, 1)
    exit(0)


# 清空命令指示符输出
def clear():
    _ = system('cls')


# 确认窗口句柄与类名
def get_window_info():
    test_window = 'Notepad3 PX_WINDOW_CLASS Notepad Notepad++'
    class_name, hwnd_var = None, None
    testing_purpose = False
    found_window = False
    while not found_window:  # 等待游戏窗口出现
        millisleep(3000)
        try:
            hwnd_active = win32gui.GetForegroundWindow()
            title_name = win32gui.GetWindowText(hwnd_active)
            class_name = win32gui.GetClassName(hwnd_active)
            if MsgBox('这是你需要的游戏窗口名称吗?(请开启游戏窗口化)', title_name, style=4):
                outer_hwnd = hwnd_var = win32gui.FindWindow(class_name, None)
                inner_hwnd = win32gui.FindWindowEx(hwnd_var, None, None, None)
                if inner_hwnd and win32gui.GetClientRect(inner_hwnd) != (0, 0, 0, 0):
                    hwnd_var = inner_hwnd
                if class_name in test_window:
                    testing_purpose = True

                print('已找到并确认窗口窗口')
                found_window = True
        except pywintypes.error:
            print('您可能正使用沙盒,目前不支持沙盒使用')
            continue

    return class_name, hwnd_var, outer_hwnd, testing_purpose


# 比起python自带sleep稍微精准的睡眠
def millisleep(num):
    TimeBeginPeriod(1)
    HPSleep(int(num))  # 减少报错
    TimeEndPeriod(1)


# 简易FOV计算
def FOV(target_move, base_len):
    actual_move = atan(target_move/base_len) * base_len  # 弧长
    return actual_move


# 简易反FOV计算
def anti_FOV(actual_move, base_len):
    target_move = tan(actual_move/base_len) * base_len
    return target_move


# 用户选择
def use_choice(rangemin, rangemax, askstring):
    selection = -1
    while not (rangemax >= selection >= rangemin):
        user_choice = input(askstring)
        try:
            selection = int(user_choice)
            if not (rangemax >= selection >= rangemin):
                print('请在给定范围选择')
        except ValueError:
            print('呵呵...请重新输入')

    return selection


# 简易消息框
def MsgBox(title, text, style=0):
    result = windll.user32.MessageBoxW(0, text, title, style)
    if style in [0, 1]:
        return result == 1
    elif style in [3, 4]:
        return result == 6

    ##  Styles:
    ##  0 : OK
    ##  1 : OK | Cancel
    ##  2 : Abort | Retry | Ignore
    ##  3 : Yes | No | Cancel
    ##  4 : Yes | No
    ##  5 : Retry | Cancel
    ##  6 : Cancel | Try Again | Continue

    ##  Results:
    ##  1 : IDOK
    ##  2 : IDCANCEL
    ##  3 : IDABORT
    ##  6 : IDYES
    ##  7 : IDNO


# 尝试pid
class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0, ExpValue=0.0):
        self.kp = P
        self.ki = I
        self.kd = D
        self.uPrevious = 0
        self.uCurent = 0
        self.setValue = ExpValue
        self.lastErr = 0
        self.preLastErr = 0
        self.errSum = 0
        self.errSumLimit = 10

    # 位置式PID
    def pidPosition(self, curValue):
        err = self.setValue - curValue
        dErr = err - self.lastErr
        self.preLastErr = self.lastErr
        self.lastErr = err
        self.errSum += err
        outPID = self.kp * err + (self.ki * self.errSum) + (self.kd * dErr)
        return outPID

    # 增量式PID
    def __call__(self, curValue):
        self.uCurent = self.pidPosition(curValue)  # 用位置式记录位置
        outPID = self.uCurent - self.uPrevious
        self.uPrevious = self.uCurent
        return outPID

    # 更新比例P值
    def set_p(self, new_p):
        self.kp = new_p

    # 获取比例P值
    def get_p(self):
        return self.kp

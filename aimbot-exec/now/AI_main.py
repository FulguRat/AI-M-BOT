from util import set_dpi, is_full_screen, is_admin, clear, restart, millisleep, get_window_info, FOV, use_choice, anti_FOV
from win32api import GetAsyncKeyState, GetCurrentProcessId, OpenProcess, GetSystemMetrics
from win32process import SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS
from multiprocessing import Process, shared_memory, Array, Lock
from win32con import VK_END, PROCESS_ALL_ACCESS
from pynput.mouse import Listener, Button
from configparser import ConfigParser
from sys import exit, platform
from collections import deque
from statistics import median
from util import PID, MsgBox
from math import sqrt, pow
from random import uniform
from ctypes import windll
from time import time
import numpy as np
import pywintypes
import win32gui
# import bezier
import cv2
import os


# 检测是否存在配置与权重文件
def check_file(file):
    cfg_file = file + '.cfg'
    weights_file = file + '.weights'
    if not (os.path.isfile(cfg_file) and os.path.isfile(weights_file)):
        print(f'请下载{file}相关文件!!!')
        millisleep(3000)
        exit(0)


# 加锁换值
def change_withlock(arrays, var, target_var, locker):
    with locker:
        arrays[var] = target_var


# 转变状态
def check_status(arr):
    if GetAsyncKeyState(VK_END) < 0:  # End
        change_withlock(arr, 14, 1, lock)
    if GetAsyncKeyState(0x31) < 0:  # 1
        change_withlock(arr, 6, 1, lock)
    if GetAsyncKeyState(0x32) < 0:  # 2
        change_withlock(arr, 6, 2, lock)
    if GetAsyncKeyState(0x33) < 0 or GetAsyncKeyState(0x34) < 0:  # 3,4
        change_withlock(arr, 6, 0, lock)
        change_withlock(arr, 18, 0, lock)
    if GetAsyncKeyState(0x46) < 0:  # F恢复移动
        change_withlock(arr, 8, 1, lock)
    if GetAsyncKeyState(0x4A) < 0:  # J停止移动
        change_withlock(arr, 8, 0, lock)
    if GetAsyncKeyState(0x12) < 0:  # Alt恢复开火
        change_withlock(arr, 9, 1, lock)
    if GetAsyncKeyState(0x30) < 0:  # 0停止开火
        change_withlock(arr, 9, 0, lock)
    if GetAsyncKeyState(0x26) < 0:  # 上方向键↑
        move_factor = arr[21]
        move_factor += 0.002
        change_withlock(arr, 21, move_factor, lock)
    if GetAsyncKeyState(0x28) < 0:  # 下方向键↓
        move_factor = arr[21]
        move_factor -= 0.002
        if move_factor < 0.01:
            move_factor = 0.01
        change_withlock(arr, 21, move_factor, lock)
    if GetAsyncKeyState(0x27) < 0:  # 右方向键→
        lock_range = arr[20]
        lock_range += 0.005
        if lock_range > 1.0:
            lock_range = 1.0
        change_withlock(arr, 20, lock_range, lock)
    if GetAsyncKeyState(0x25) < 0:  # 左方向键←
        lock_range = arr[20]
        lock_range -= 0.005
        if lock_range < 0.25:
            lock_range = 0.25
        change_withlock(arr, 20, lock_range, lock)


# 多线程展示效果
def show_frames(array):
    set_dpi()
    cv2.namedWindow('Show frame', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Show frame', 0, 0)
    # cv2.destroyAllWindows()
    font = cv2.FONT_HERSHEY_SIMPLEX  # 效果展示字体

    while True:  # 等待共享内存加载完毕
        try:
            existing_show_shm = shared_memory.SharedMemory(name='showimg')
            millisleep(1000)
            break
        except FileNotFoundError:
            millisleep(1000)

    while not array[14]:
        show_img = np.ndarray((320, 512, 3), dtype=np.uint8, buffer=existing_show_shm.buf)
        show_color = {
            0: (127, 127, 127),
            1: (255, 255, 0),
            2: (0, 255, 0)
        }.get(array[6])
        try:
            img_ex = np.zeros((1, 1, 3), np.uint8)
            show_str0 = str('{:03.1f}'.format(array[4]))
            show_str1 = 'Move factor is about ' + str(round(array[21], 3))
            show_str2 = 'Lock size is around ' + str(round(array[20], 3))
            show_str3 = 'Fire rate is at ' + str('{:02.0f}'.format((1000 / (array[10] + 30.6)))) + ' RPS'
            show_str4 = 'Please enjoy coding ^_^' if array[8] else 'Please enjoy coding @_@'
            if show_img.any():
                show_img_h, show_img_w = show_img.shape[:2]
                show_img = cv2.resize(show_img, (int(array[3]), int(array[3] / show_img_w * show_img_h)))
                img_ex = cv2.resize(img_ex, (int(array[3]), int(array[3] / 2)))
                cv2.putText(show_img, show_str0, (int(array[3] / 25), int(array[3] / 12)), font, array[3] / 600, (127, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img_ex, show_str1, (10, int(array[3] / 9)), font, array[3] / 450, show_color, 1, cv2.LINE_AA)
                cv2.putText(img_ex, show_str2, (10, int(array[3] / 9) * 2), font, array[3] / 450, show_color, 1, cv2.LINE_AA)
                cv2.putText(img_ex, show_str3, (10, int(array[3] / 9) * 3), font, array[3] / 450, show_color, 1, cv2.LINE_AA)
                cv2.putText(img_ex, show_str4, (10, int(array[3] / 9) * 4), font, array[3] / 450, show_color, 1, cv2.LINE_AA)
                show_image = cv2.vconcat([show_img, img_ex])
                cv2.imshow('Show frame', show_image)
            cv2.waitKey(25)
            check_status(array)
        except AttributeError:
            pass
        except cv2.error:
            pass

    cv2.destroyAllWindows()
    existing_show_shm.close()


# 鼠标检测进程
def mouse_detection(array, lock):
    def on_click(x, y, button, pressed):
        if array[14]:
            return False
        change_withlock(array, 15, 1 if pressed and button == Button.left else 0, lock)
        if pressed and button == Button.left:
            change_withlock(array, 16, time() * 1000, lock)
        elif not pressed and button == Button.left:
            change_withlock(array, 17, time() * 1000, lock)

    with Listener(on_click=on_click) as listener:
        listener.join()  # 阻塞鼠标检测线程


# 主程序
def main():
    if not is_admin():  # 检查管理员权限
        restart(__file__)

    set_dpi()  # 设置高DPI不受影响
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 设置工作路径
    check_file('yolov4-tiny')  # 如果文件不存在则退出

    # 提升进程优先级
    if platform == 'win32':
        pid = GetCurrentProcessId()
        handle = OpenProcess(PROCESS_ALL_ACCESS, True, pid)
        SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)
    else:
        os.nice(1)

    from scrnshot import WindowCapture
    from mouse import mouse_down, mouse_up, mouse_close, scroll, key_down, key_up, move_mouse, gmok, msdkok, ddok
    MsgBox('键鼠检测', f'=============\nDD  驱动加载状态: {ddok}\n罗技驱动加载状态: {gmok}\n盒子驱动准备状态: {msdkok}\n', 0)

    # 鼠标射击
    def click_mouse(win_class, rate, go_fire):
        # 不分敌友射击
        if arr[15]:  # GetAsyncKeyState(VK_LBUTTON) < 0
            if time() * 1000 - arr[16] > 30.6:  # press_moment
                mouse_up()
        elif win_class != 'CrossFire' or arr[13]:
            if arr[13] or go_fire:
                if time() * 1000 - arr[17] > rate:  # release_moment
                    mouse_down()
                    change_withlock(arr, 18, arr[18] + 1, lock)

        if time() * 1000 - arr[17] > 219.4:
            change_withlock(arr, 18, 0, lock)

        if arr[18] > 12:
            change_withlock(arr, 18, 12, lock)


    # 滑稽/选择模型
    print('提示: 您的选择将决定使用的模型')
    Conan = use_choice(0, 2, '柯南能在本程序作者有生之年完结吗?(1:能, 2:能, 0:不能): ')

    show_fps, DPI_Var = [1], [1]

    # 寻找读取游戏窗口类型并确认截取位置
    window_class_name, window_hwnd_name, window_outer_hwnd, test_win = get_window_info()

    mouse_detect_proc = Process(target=mouse_detection, args=(arr, lock,))  # 鼠标检测进程
    show_proc = Process(target=show_frames, args=(arr,))  # 效果展示进程

    # 检查窗口DPI
    DPI_Var[0] = max(windll.user32.GetDpiForWindow(window_outer_hwnd) / 96, windll.user32.GetDpiForWindow(window_hwnd_name) / 96)
    DPI_Var[0] = 1.0 if DPI_Var[0] == 0.0 else DPI_Var[0]

    process_times = deque()
    move_recordx = deque()

    arr[0] = 0  # 截图宽
    arr[1] = 0  # 截图高
    arr[2] = 1  # 截图进程状态
    arr[3] = 0  # 左侧距离
    arr[4] = 0  # FPS值
    arr[5] = 600  # 基础边长
    arr[6] = 0  # 控制鼠标/所持武器
    arr[7] = 0  # 目标数量
    arr[8] = 1  # 移动鼠标与否
    arr[9] = 1  # 按击鼠标与否
    arr[10] = 94.4  # 射击速度
    arr[11] = 0  # 瞄准位置(0中1头2胸)
    arr[12] = 0  # 简易后坐力控制
    arr[13] = 0  # CF下红名
    arr[14] = 0  # 是否退出
    arr[15] = 0  # 鼠标状态
    arr[16] = time()  # 左键按下时刻
    arr[17] = time()  # 左键抬起时刻
    arr[18] = 0  # 连续射击次数
    arr[19] = 1600  # 窗口宽
    arr[20] = 1.0  # 锁定范围
    arr[21] = 1.0  # 鼠标移动系数

    recoil_more = 1
    recoil_control = 2.0

    config = ConfigParser()
    if len(config.read('config.ini')) > 0:
        if 'recoil control' in config:
            if window_class_name in config['recoil control']:
                recoil_control = float(config['recoil control'][window_class_name])
        else:
            config.add_section('recoil control')

        if 'move factor' in config:
            if window_class_name in config['move factor']:
                arr[21] = float(config['move factor'][window_class_name])
        else:
            config.add_section('move factor')
    else:
        config.add_section('recoil control')
        config.add_section('move factor')

    mouse_detect_proc.start()  # 开始鼠标监测进程

    # 如果非全屏则展示效果
    F11_Mode = 1 if is_full_screen(window_hwnd_name) else 0
    if not F11_Mode:
        show_proc.start()
    else:
        print('全屏模式下不会有小窗口...')

    # 等待游戏画面完整出现(拥有大于0的长宽)
    window_ready = 0
    while not window_ready:
        millisleep(1000)
        win_client_rect = win32gui.GetClientRect(window_hwnd_name)
        win_pos = win32gui.ClientToScreen(window_hwnd_name, (0, 0))
        if win_client_rect[2] - win_client_rect[0] > 0 and win_client_rect[3] - win_client_rect[1] > 0:
            window_ready = 1

    print(win_pos[0], win_pos[1], win_client_rect[2], win_client_rect[3])

    # 初始化分析类
    if Conan == 1:
        from darknet_yolo34 import FrameDetection34
        Analysis = FrameDetection34(window_hwnd_name)
        string_model = '您正使用yolov4-tiny模型'
    elif Conan == 2:
        from torch_yolox import FrameDetectionX
        Analysis = FrameDetectionX(window_hwnd_name)
        string_model = '您正使用yolox-tiny模型'
    elif Conan == 3:
        from torch_yolo5 import FrameDetection5
        Analysis = FrameDetection5(window_hwnd_name)
        string_model = '您正使用yolov5-s模型'

    print(string_model)

    # 等待截图类初始化
    while not arr[2]:
        millisleep(1000)

    # clear()  # 清空命令指示符面板

    ini_sct_time = 0  # 初始化计时
    target_count, moveX, moveY, fire0pos, enemy_close, can_fire = 0, 0, 0, 0, 0, 0
    pidx = PID(0.3, 0.875, 0.01, 0)  # 初始化pid
    prev_movex, prev_movey = 0, 0
    small_float = np.finfo(np.float64).eps  # 初始化一个尽可能小却小得不过分的数
    shm_show_img = shared_memory.SharedMemory(create=True, size=GetSystemMetrics(0) * GetSystemMetrics(1) * 3, name='showimg')  # 创建进程间共享内存
    cf_enemy_color = np.array([3487638, 3487639, 3487640, 3487641, 3422105, 3422106, 3422362, 3422363, 3422364, 3356828, 3356829, 3356830, 3356831, 3291295, 3291551, 3291552, 3291553, 3291554, 3226018, 3226019, 3226020, 3226276, 3226277, 3160741, 3160742, 3160743, 3160744, 3095208, 3095209, 3095465, 3095466, 3095467, 3029931, 3029932, 3029933, 3029934, 3030190, 2964654, 2964655, 2964656, 2964657, 2899121, 2899122, 2899123, 2899379, 2899380, 2833844, 2833845, 2833846, 2833847, 2768311, 2768567, 2768568, 2768569, 2768570, 2703034, 2703035, 2703036, 2703292, 2703292, 2703293, 2637757, 2637758, 2637759, 2637760, 2572224, 2572225, 2572481, 2572482, 2572483, 2506948, 2506949, 2506950, 2507206, 2507207, 2441671, 2441672, 2441673, 2441674, 2376138, 2376139, 2376395, 2376396, 2376397, 2310861, 2310862, 2310863, 2310864, 2311120, 2245584, 2245585, 2245586, 2245587, 2180051, 2180052, 2180308, 2180309, 2180310, 2114774, 2114775, 2114776, 2114777, 2049241, 2049497, 2049498, 2049499, 2049500, 1983964, 1983965, 1983966, 1984222, 1984223, 1918687, 1918688, 1918689, 1918690, 1853154, 1853155, 1853411, 1853412, 1853413, 1787877, 1787878, 1787879, 1787880, 1788136, 1722600, 1722601, 1722602, 1722603, 1657067, 1657068, 1657069, 1657325, 1657326, 1591790, 1591791, 1591792, 1591793, 1526514])  # CF敌方红名库

    win_cap = WindowCapture(window_class_name, window_hwnd_name, 4/9, 1.6)  # 初始化截图类
    winw, winh = win_cap.get_window_info()  # 获取窗口宽高
    change_withlock(arr, 19, winw, lock)
    cutw, cuth = win_cap.get_cut_info()  # 获取截屏宽高
    change_withlock(arr, 1, cutw, lock)
    change_withlock(arr, 0, cuth, lock)
    opt_trackingx = 0

    # 计算基础边长
    change_withlock(arr, 5, win_cap.get_side_len(), lock)
    print(f'基础边长 = {arr[5]}')

    while not arr[14]:
        # screenshot = win_cap.grab_screenshot()
        screenshot = win_cap.get_screenshot()
        try:
            frame_height, frame_width = screenshot.shape[:2]
            change_withlock(arr, 0, frame_height, lock)  # 改变共享数组中对应截屏高的数据
            change_withlock(arr, 1, frame_width, lock)  # 改变共享数组中对应截屏宽的数据

            # 穿越火线检测红名
            if window_class_name == 'CrossFire':
                cut_scrn = screenshot[cuth // 2 + winh // 16 : cuth // 2 + winh // 15, cutw // 2 - winw // 40 : cutw // 2 + winw // 40]  # 从截屏中截取红名区域

                # 将红名区域rgb转为十进制数值
                hexcolor = []
                for i in range(cut_scrn.shape[0]):
                    for j in range(cut_scrn.shape[1]):
                        rgbint = cut_scrn[i][j][0]<<16 | cut_scrn[i][j][1]<<8 | cut_scrn[i][j][2]
                        hexcolor.append(rgbint)

                # 与内容中的敌方红名色库比较
                hexcolor = np.array(hexcolor)
                indices = np.intersect1d(cf_enemy_color, hexcolor)
                change_withlock(arr, 13, len(indices), lock)

            # 简易实现改变窗口范围
            lockwinrangew = arr[20] * frame_width // 2 * 2
            lockwinrangeh = arr[20] * frame_height // 2 * 2
            cv2.rectangle(screenshot, (0, 0), (int((frame_width-lockwinrangew) / 2), frame_height), (127, 127, 127), cv2.FILLED)
            cv2.rectangle(screenshot, (int((frame_width+lockwinrangew) / 2), 0), (frame_width, frame_height), (127, 127, 127), cv2.FILLED)
            cv2.rectangle(screenshot, (int((frame_width-lockwinrangew) / 2), 0), (int((frame_width+lockwinrangew) / 2), int((frame_height-lockwinrangeh) / 2)), (127, 127, 127), cv2.FILLED)
            cv2.rectangle(screenshot, (int((frame_width-lockwinrangew) / 2), int((frame_height+lockwinrangeh) / 2)), (int((frame_width+lockwinrangew) / 2), frame_height), (127, 127, 127), cv2.FILLED)

            screenshot = cv2.resize(screenshot, (512, 320), interpolation=cv2.INTER_AREA)
            win_left = (150 if win_cap.get_window_left() - 10 < 150 else win_cap.get_window_left() - 10)
            change_withlock(arr, 3, win_left, lock)

        except (AttributeError, pywintypes.error) as e:
            print('窗口已关闭\n' + str(e))
            millisleep(1)
            continue
        except cv2.error as e:
            print('窗口已最小化\n' + str(e))
            millisleep(1)
            continue

        if Conan:
            try:
                target_count, moveX, moveY, fire0pos, enemy_close, can_fire, screenshot = Analysis.detect(screenshot, arr[12], arr[19], opt_trackingx)
                change_withlock(arr, 7, target_count, lock)
                change_withlock(arr, 11, fire0pos, lock)
            except ValueError as e:
                print('窗口已最小化\n' + str(e))
                break

        if str(win32gui.GetForegroundWindow()) in (str(window_hwnd_name) + str(window_outer_hwnd)) and not test_win:  # 是否需要控制鼠标:
            if arr[6] == 1:  # 主武器
                change_withlock(arr, 10, 94.4 if enemy_close or arr[11] != 1 else 169.4, lock)
            elif arr[6] == 2:  # 副武器
                change_withlock(arr, 10, 69.4 if enemy_close or arr[11] != 1 else 94.4, lock)

            if arr[6] and target_count and (arr[8] or GetAsyncKeyState(0x06) < 0):  # XButton2
                change_withlock(arr, 12, recoil_more * recoil_control * arr[18] / arr[6], lock)
                act_moveX = FOV(moveX, arr[5]) / DPI_Var[0] * arr[21] * arr[0] / 512
                act_moveY = FOV(moveY, arr[5]) / DPI_Var[0] * arr[21] * arr[1] / 320
                pid_moveX = -pidx(act_moveX) - prev_movex
                pid_moveY = act_moveY * (1 / pow(show_fps[0]/3, 1/3)) - prev_movey
                prev_movex, prev_movey = pid_moveX, pid_moveY

                move_recordx.append(pid_moveX)
                opt_trackingx = median(move_recordx)*2.1 if GetAsyncKeyState(0x45) < 0 else 0  # E

                recoil_more = 1.25 if 1000/(arr[10] + 30.6) > 6 else 1
                move_mouse(round(pid_moveX, 3), round(pid_moveY, 3))
            else:
                prev_movex, prev_movey = 0, 0
                move_recordx.append(0)
                opt_trackingx = 0

            if arr[6] and arr[9]:
                click_mouse(window_class_name, arr[10], can_fire)

        with lock:
            show_img = np.ndarray(screenshot.shape, dtype=screenshot.dtype, buffer=shm_show_img.buf)
            show_img[:] = screenshot[:]  # 将截取数据拷贝进分享的内存

        # millisleep(1)  # 降低平均cpu占用
        if F11_Mode:
            check_status(arr)
        time_used = time() - ini_sct_time
        ini_sct_time = time()
        process_times.append(time_used)
        med_time = median(process_times)
        pidx.set_p(1 / pow(show_fps[0]/3, 1/3))
        show_fps[0] = 1 / med_time if med_time > 0 else 1 / (med_time + small_float)
        change_withlock(arr, 4, round(show_fps[0]), lock)
        if len(process_times) > show_fps[0]:
            process_times.popleft()
        if len(move_recordx) > sqrt(show_fps[0]):
            move_recordx.popleft()

    config['move factor'][window_class_name] = str(round(arr[21], 3))
    config['recoil control'][window_class_name] = str(round(recoil_control, 3))
    with open('config.ini', 'w') as cf:
        config.write(cf)

    print('关闭进程中......')
    change_withlock(arr, 14, 1, lock)
    win_cap.release_resource()
    millisleep(1000)  # 为了稳定
    shm_show_img.close()
    shm_show_img.unlink()
    if not F11_Mode:
        show_proc.terminate()
    mouse_detect_proc.terminate()
    mouse_close()
    exit(0)


arr = Array('d', range(30))  # 进程间分享数据
lock = Lock()  # 锁

"""
New Detection method(onnxruntime) modified from Project YOLOX
YOLOX Project Authors: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun
YOLOX Project website: https://github.com/Megvii-BaseDetection/YOLOX
New Detection method(onnxruntime) cooperator: Barry
Detection code modified from project AIMBOT-YOLO
Detection code Author: monokim
Detection project website: https://github.com/monokim/AIMBOT-YOLO
Detection project video: https://www.youtube.com/watch?v=vQlb0tK1DH0
Screenshot method from: https://www.youtube.com/watch?v=WymCpVUPWQ4
Screenshot method code modified from project: opencv_tutorials
Screenshot method code Author: Ben Johnson (learncodebygaming)
Screenshot method website: https://github.com/learncodebygaming/opencv_tutorials
Mouse event method code modified from project logitech-cve
Mouse event method website: https://github.com/ekknod/logitech-cve
Mouse event method project Author: ekknod
PID method Author: peizhuo_liu (RawBottle)
PID method website: https://blog.csdn.net/peizhuo_liu/article/details/112058679
"""

from multiprocessing import freeze_support
from mouse import gmok, msdkok, ddok
from util import MsgBox


# 主程序
if __name__ == '__main__':
    freeze_support()  # 为了顺利编译成exe

    # 选择标准/烧卡模式
    if MsgBox('欢迎使用与交流', f'作者: jiapai12138 (killmatt01)\n学习交流欢迎加群: 212068326\n\n键鼠检测============\nDD驱动加载状态: {ddok}\n罗技驱动加载状态: {gmok}\n飞易来/文盒驱动准备状态: {msdkok}\n\n\n==================\n请使游戏窗口化运行\n请问您的电脑是高配机吗?', 4):
        from AI_main_pow import main
    else:
        from AI_main import main

    main()

from win32file import CreateFile, SetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING
from random import randrange
from time import time, sleep
from tqdm import tqdm
from sys import exit
import numpy as np
import pywintypes
import cv2
import os


# 随机改变文件创建/修改/访问时间
def RandomFileTime(file_path):
    try:
        fh = CreateFile(file_path, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, 0)
        new_c_time = int(time() - randrange(2500000, 30000000))
        new_m_time = new_c_time + randrange(10000, 500000)
        new_a_time = new_m_time + randrange(90000, 2000000)

        createTimes = pywintypes.Time(new_c_time)
        accessTimes = pywintypes.Time(new_a_time)
        modifyTimes = pywintypes.Time(new_m_time)
        SetFileTime(fh, createTimes, accessTimes, modifyTimes)
        CloseHandle(fh)
        return 0
    except pywintypes.error:
        return 1


source_path = './游戏截图/'  # 源文件路径
target_path = './预先处理/'  # 输出目标文件路径

if not os.path.isdir(source_path):
    os.makedirs(source_path)
    exit(0)

if not os.path.isdir(target_path):
    os.makedirs(target_path)

if __name__ == '__main__':
    if not os.listdir(source_path):
        exit(0)

    fail_change_time = 0
    for file in tqdm(os.listdir(source_path)):
        if not os.path.isfile(source_path + file):
            continue
        image_source = cv2.imdecode(np.fromfile(source_path + file, dtype=np.uint8), -1)  # 读取图片
        if RandomFileTime(source_path + file):
            fail_change_time += 1

        image = cv2.resize(image_source, (512, 320), interpolation=cv2.INTER_AREA)  # 修改尺寸变小
        cv2.imencode('.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])[1].tofile(target_path + file[0:file.find('.')] + '.png')  # 重命名并且保存

    print(f'批量处理完成，文件改时失败{fail_change_time}个')
    sleep(3)

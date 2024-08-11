import os
import time

import cv2

from adb.scrcpy_adb import ScrcpyADB

if __name__ == '__main__':
    sadb = ScrcpyADB(1384)
    i = 193
    # 判断images目录是否存在，不存在就创建文件夹
    dir_name = "images"
    # 判断目录是否存在
    if not os.path.exists(dir_name):
        # 如果不存在，则创建目录
        os.makedirs(dir_name)

    while True:
        if sadb.last_screen is None:
            continue
        # 保存屏幕
        cv2.imwrite(f'{dir_name}/jh3_screen_{i}.jpg', sadb.last_screen)
        i += 1
        time.sleep(1)

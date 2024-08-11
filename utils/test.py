import time

from utils.yolov5 import YoloV5s
import cv2 as cv


def plot_one_box(x, img_source, color=None, label=None, line_thickness=None):
    """
    画框
    :param x:
    :param img_source:
    :param color:
    :param label:
    :param line_thickness:
    :return:
    """
    # 线条粗细
    tl = line_thickness or round(0.002 * (img_source.shape[0] + img_source.shape[1]) / 2) + 1
    color = color or [0, 255, 0]  # 默认绿色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 转换坐标为整数

    # 画框
    cv.rectangle(img_source, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)

    if label:
        tf = max(tl - 1, 2)  # 文本字体大小
        text_color = [255, 255, 255]  # 默认白色，可以考虑通过参数传递
        text_color = text_color if color is None else color  # 如果有颜色参数，则使用此颜色作为文字颜色

        # 计算文本位置
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # 在原位置绘制文本，无背景
        cv.putText(img_source, label, (c1[0], c1[1] - 2), 0, tl / 3, text_color, thickness=tf,
                   lineType=cv.LINE_AA)


yolo = YoloV5s(target_size=640,
                            prob_threshold=0.25,
                            nms_threshold=0.45,
                            num_threads=4,
                            use_gpu=True)

frame = cv.imread('/Users/wenzhuangxie/PycharmProjects/dnfm-yolo-tutorial/utils/frame_26.jpg')
frame2 = cv.imread('/Users/wenzhuangxie/PycharmProjects/X-AnyLabeling/mylabeldnf/布万加/hebing/images/frame_162.jpg')
frame3 = cv.imread('/Users/wenzhuangxie/PycharmProjects/X-AnyLabeling/mylabeldnf/布万加/hebing/images/frame_205.jpg')
# frame = cv.imread('/Users/wenzhuangxie/Downloads/导师套布万加数据集/images/14_frame_00097.bmp')
try:
    s = time.time()
    result = yolo(frame2)
    print(f"匹配1耗时：{(time.time() - s) * 1000}ms")
    s = time.time()
    result =yolo(frame3)
    print(f"匹配2耗时：{(time.time() - s) * 1000}ms")
    s = time.time()
    result = yolo(frame)
    print(f"匹配3耗时：{(time.time() - s)*1000}ms")
    s = time.time()
    for obj in result:
        color = (0, 255, 0)
        if obj.label == 1:
            color = (255, 0, 0)
        elif obj.label == 5:
            color = (0, 0, 255)
        text = f"{yolo.class_names[int(obj.label)]}:{obj.prob:.2f}"


        cv.rectangle(frame,
                     (int(obj.rect.x), int(obj.rect.y)),
                     (int(obj.rect.x + obj.rect.w), int(obj.rect.y + + obj.rect.h)),
                     color, 2
                     )
        plot_one_box([obj.rect.x, obj.rect.y, obj.rect.x + obj.rect.w, obj.rect.y + obj.rect.h], frame, color=color, label=text, line_thickness=2)
        print(text)
    print(f"画框耗时：{(time.time() - s)*1000}ms")

except Exception as e:
    print(e)

cv.circle(frame, (561, 853), 5, (0, 255, 0), 5)
cv.imshow('screen', frame)
cv.waitKey(0)
# cv.imshow('frame', frame)
# cv.waitKey(10)
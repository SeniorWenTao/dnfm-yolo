import typing

from ncnn.utils.objects import Rect
from torch import Size


# 走的房间路线顺序路线图,第几行第几列
room_route = [
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 2),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5)
]
print(f"布万加刷图的路线图:{room_route}")
all_room = [
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 2),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (0, 2),
    (0, 1),
    (0, 0)
]

# 获取屏幕每帧的缩放比例
zoom_ratio = 1


def rect_slice_index(rect: Rect, size: Size, point: tuple) -> tuple[int, int]:
    """
    矩形（范围)，切片索引
    ：param rect：范围
    ：param size： 切片大小 x 方块数量 y 轴方块数量
    ：param point：要索引的坐标
    : return: 如 (4, 1) 则表示目标在切片索引的第 4 行，第 1 列
    """
    rect = Rect(rect.x * zoom_ratio, rect.y * zoom_ratio, rect.w * zoom_ratio, rect.h * zoom_ratio)
    x1, y1, x2, y2 = rect.x, rect.y, rect.x + rect.w, rect.y + rect.h
    width = rect.w / size[0]
    height = rect.h / size[1]
    rects = [
        [(x1 + i * width, y1 + j * height, x1 + (i + 1) * width, y1 + (j + 1) * height) for i in range(size[0])] for
        j in range(size[1])]
    # print(f"所有方块对应的坐标范围:{rects}")
    for i, row in enumerate(rects):
        for j, rect in enumerate(row):
            x1, y1, x2, y2 = rect
            if x1 <= point[0] < x2 and y1 <= point[1] < y2:
                return i, j


def get_cur_room_index(point):
    """
    获取房间索引
    ：param point：要索引的坐标
    : return: 如 (4, 1) 则表示目标在切片索引的第 4 行，第 1 列
    """
    # 锁定获取的范围和每个房间方块的大小
    xy = rect_slice_index(Rect(850, 380, 635, 315), Size((6, 3)), point)
    # 遍历room_route
    cur_ind = None

    for ind,room in enumerate(room_route):
        if xy == room_route[ind]:
            cur_ind = ind
            break
    print(f"当前房间索引:{cur_ind},房间行列号：{xy}")
    return cur_ind,xy


def get_recent_room(cur_room):
    next_room_1 = (cur_room[0],cur_room[1] + 1)
    next_room_2 = (cur_room[0] + 1,cur_room[1])
    next_room_3 = (cur_room[0],cur_room[1] - 1)
    next_room_4 = (cur_room[0] - 1,cur_room[1])
    for ind,room in enumerate(room_route):
        if next_room_1 == room or next_room_2 == room or next_room_3 == room or next_room_4 == room:
            return room
    print("从预设的路线中，没有找到下一个房间")
    for ind,room in enumerate(all_room):
        if next_room_1 == room or next_room_2 == room or next_room_3 == room or next_room_4 == room:
            return room


def get_next_room(point, is_succ_sztroom: bool = False):
    ind,cur_room = get_cur_room_index(point)
    # 狮子头房间刷过了，不用去了
    if is_succ_sztroom and cur_room == (1, 2):
        return 7,(1,3)

    # vals转换成连表的形式，用于获取下一个
    if cur_room in room_route:
        try:
            index = room_route.index(cur_room,ind)
        except ValueError:
            index = all_room.index(cur_room,0)

        if index >= len(room_route) - 1:
            return None,None
        next_room = room_route[index + 1]
        return index + 1,next_room


def get_run_direction(cur_room,next_room):
    # (1,0)-(2,0) 走右
    # 计算行走的方向
    if cur_room[0] == next_room[0]:
        if cur_room[1] > next_room[1]:
            return 'left'
        else:
            return 'right'
    else:
        if cur_room[0] > next_room[0]:
            return 'up'
        else:
            return 'down'


def get_tag_by_direction(direction):
    # 计算行走的方向
    if direction == 'up':
        return 'opendoor_u'
    elif direction == 'down':
        return 'opendoor_d'
    elif direction == 'left':
        return 'opendoor_l'
    elif direction == 'right':
        return 'opendoor_r'


if __name__ == '__main__':
    t = rect_slice_index(Rect(100, 100, 100, 100), Size((10, 6)), (101.54, 145.864))
    print(t)
    print(t == (4, 0))
    print(get_cur_room_index((101.54, 145.864)))
    print(get_next_room((101.54, 145.864), False))
    print(get_run_direction((0, 1), (0, 2)))
    print(get_run_direction((0, 2), (1, 2)))
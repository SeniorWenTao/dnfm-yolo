# sift FLANN matcher
import time

import numpy as np
import cv2


def filter_good_point(matches, kp_src, kp_sch, kp_sch_point, kp_src_matches_point):
    """ 筛选最佳点 """
    # 假设第一个点,及distance最小的点,为基准点
    sort_list = [sorted(match, key=lambda x: x is np.nan and float('inf') or x.distance)[0]
                 for match in matches]
    sort_list = [v for v in sort_list if v is not np.nan]

    first_good_point: cv2.DMatch = sorted(sort_list, key=lambda x: x.distance)[0]
    first_good_point_train: cv2.KeyPoint = kp_src[first_good_point.trainIdx]
    first_good_point_query: cv2.KeyPoint = kp_sch[first_good_point.queryIdx]
    first_good_point_angle = first_good_point_train.angle - first_good_point_query.angle

    def get_points_origin_angle(point_x, point_y, offset):
        points_origin_angle = np.arctan2(
            (point_y - offset.pt[1]),
            (point_x - offset.pt[0])
        ) * 180 / np.pi

        points_origin_angle = np.where(
            points_origin_angle == 0,
            points_origin_angle, points_origin_angle - offset.angle
        )
        points_origin_angle = np.where(
            points_origin_angle >= 0,
            points_origin_angle, points_origin_angle + 360
        )
        return points_origin_angle

    # 计算模板图像上,该点与其他特征点的旋转角
    first_good_point_sch_origin_angle = get_points_origin_angle(kp_sch_point[:, 0], kp_sch_point[:, 1],
                                                                first_good_point_query)

    # 计算目标图像中,该点与其他特征点的夹角
    kp_sch_rotate_angle = kp_sch_point[:, 2] + first_good_point_angle
    kp_sch_rotate_angle = np.where(kp_sch_rotate_angle >= 360, kp_sch_rotate_angle - 360, kp_sch_rotate_angle)
    kp_sch_rotate_angle = kp_sch_rotate_angle.reshape(kp_sch_rotate_angle.shape + (1,))

    kp_src_angle = kp_src_matches_point[:, :, 2]
    good_point = np.array([matches[index][array[0]] for index, array in
                           enumerate(np.argsort(np.abs(kp_src_angle - kp_sch_rotate_angle)))])

    # 计算各点以first_good_point为原点的旋转角
    good_point_nan = (np.nan, np.nan)
    good_point_pt = np.array([good_point_nan if dMatch is np.nan else (*kp_src[dMatch.trainIdx].pt,)
                              for dMatch in good_point])
    good_point_origin_angle = get_points_origin_angle(good_point_pt[:, 0], good_point_pt[:, 1],
                                                      first_good_point_train)
    threshold = round(5 / 360, 2) * 100
    point_bool = (np.abs(good_point_origin_angle - first_good_point_sch_origin_angle) / 360) * 100 < threshold
    _, index = np.unique(good_point_pt[point_bool], return_index=True, axis=0)
    good = good_point[point_bool]
    good = good[index]
    return good, int(first_good_point_angle), first_good_point


from utils.cvmatch import (generate_result, get_keypoint_from_matches, keypoint_distance, rectangle_transform)


def _find_homography(sch_pts, src_pts):
    """
    多组特征点对时，求取单向性矩阵
    """
    try:
        # M, mask = cv2.findHomography(sch_pts, src_pts, cv2.RANSAC)
        M, mask = cv2.findHomography(sch_pts, src_pts, cv2.USAC_MAGSAC, 4.0, None, 2000, 0.99)
    except cv2.error:
        import traceback
        traceback.print_exc()
        raise Exception("OpenCV error in _find_homography()...")
    else:
        if mask is None:
            raise Exception("In _find_homography(), find no mask...")
        else:
            return M, mask


def _perspective_transform(im_source, im_search, src, dst):
    """
    根据四对对应点计算透视变换, 并裁剪相应图片

    Args:
        im_source: 待匹配图像
        im_search: 待匹配模板
        src: 目标图像中相应四边形顶点的坐标 (左上,右上,左下,右下)
        dst: 源图像中四边形顶点的坐标 (左上,右上,左下,右下)

    Returns:

    """
    h, w = im_search.shape
    matrix = cv2.getPerspectiveTransform(src=src, dst=dst)
    # warpPerspective https://github.com/opencv/opencv/issues/11784
    # output = cv2.warpAffine(im_source, matrix, (w, h))
    output = cv2.warpPerspective(im_source,matrix, (w, h), flags=cv2.INTER_CUBIC)
    # im_source.

    return output


def _get_perspective_area_rect(im_source, src):
    """
    根据矩形四个顶点坐标,获取在原图中的最大外接矩形

    Args:
        im_source: 待匹配图像
        src: 目标图像中相应四边形顶点的坐标

    Returns:
        最大外接矩形
    """
    h, w = im_source.shape[:2]

    x = [int(i[0]) for i in src]
    y = [int(i[1]) for i in src]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    # 挑选出目标矩形区域可能会有越界情况，越界时直接将其置为边界：
    # 超出左边界取0，超出右边界取w_s-1，超出下边界取0，超出上边界取h_s-1
    # 当x_min小于0时，取0。  x_max小于0时，取0。
    x_min, x_max = int(max(x_min, 0)), int(max(x_max, 0))
    # 当x_min大于w_s时，取值w_s-1。  x_max大于w_s-1时，取w_s-1。
    x_min, x_max = int(min(x_min, w - 1)), int(min(x_max, w - 1))
    # 当y_min小于0时，取0。  y_max小于0时，取0。
    y_min, y_max = int(max(y_min, 0)), int(max(y_max, 0))
    # 当y_min大于h_s时，取值h_s-1。  y_max大于h_s-1时，取h_s-1。
    y_min, y_max = int(min(y_min, h - 1)), int(min(y_max, h - 1))
    rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    # rect = Rect(x=x_min, y=y_min, width=(x_max - x_min), height=(y_max - y_min))
    return rect


def _handle_one_good_points(im_source, im_search, kp_src, kp_sch, good, angle):
    """
    特征点匹配数量等于1时,根据特征点的大小,对矩形进行缩放,并根据旋转角度,获取识别的目标图片

    Args:
        im_source: 待匹配图像
        im_search: 图片模板
        kp_sch: 关键点集
        kp_src: 关键点集
        good: 描述符集
        angle: 旋转角度

    Returns:
        待验证的图片
    """
    sch_point = get_keypoint_from_matches(kp=kp_sch, matches=good, mode='query')[0]
    src_point = get_keypoint_from_matches(kp=kp_src, matches=good, mode='train')[0]

    scale = src_point.size / sch_point.size
    h, w = im_search.shape[:2]
    _h, _w = h * scale, w * scale
    src = np.float32(rectangle_transform(point=sch_point.pt, size=(h, w), mapping_point=src_point.pt,
                                         mapping_size=(_h, _w), angle=angle))
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    output = _perspective_transform(im_source=im_source, im_search=im_search, src=src, dst=dst)
    rect = _get_perspective_area_rect(im_source=im_source, src=src)
    return output, rect


def _handle_two_good_points( im_source, im_search, kp_src, kp_sch, good, angle):
    """
    特征点匹配数量等于2时,根据两点距离差,对矩形进行缩放,并根据旋转角度,获取识别的目标图片

    Args:
        im_source: 待匹配图像
        im_search: 图片模板
        kp_sch: 关键点集
        kp_src: 关键点集
        good: 描述符集
        angle: 旋转角度

    Returns:
        待验证的图片
    """
    sch_point = get_keypoint_from_matches(kp=kp_sch, matches=good, mode='query')
    src_point = get_keypoint_from_matches(kp=kp_src, matches=good, mode='train')

    sch_distance = keypoint_distance(sch_point[0], sch_point[1])
    src_distance = keypoint_distance(src_point[0], src_point[1])

    try:
        scale = src_distance / sch_distance  # 计算缩放大小
    except ZeroDivisionError:
        if src_distance == sch_distance:
            scale = 1
        else:
            return None, None

    h, w = im_search.shape[:2]
    _h, _w = h * scale, w * scale
    src = np.float32(rectangle_transform(point=sch_point[0].pt, size=(h, w), mapping_point=src_point[0].pt,
                                         mapping_size=(_h, _w), angle=angle))
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    output = _perspective_transform(im_source=im_source, im_search=im_search, src=src, dst=dst)
    rect = _get_perspective_area_rect(im_source=im_source, src=src)
    return output, rect

def _handle_three_good_points( im_source, im_search, kp_src, kp_sch, good, angle):
    """
    特征点匹配数量等于3时,根据三个点组成的三角面积差,对矩形进行缩放,并根据旋转角度,获取识别的目标图片

    Args:
        im_source: 待匹配图像
        im_search: 图片模板
        kp_sch: 关键点集
        kp_src: 关键点集
        good: 描述符集
        angle: 旋转角度

    Returns:
        待验证的图片
    """
    sch_point = get_keypoint_from_matches(kp=kp_sch, matches=good, mode='query')
    src_point = get_keypoint_from_matches(kp=kp_src, matches=good, mode='train')

    def _area(point_list):
        p1_2 = keypoint_distance(point_list[0], point_list[1])
        p1_3 = keypoint_distance(point_list[0], point_list[2])
        p2_3 = keypoint_distance(point_list[1], point_list[2])

        s = (p1_2 + p1_3 + p2_3) / 2
        area = (s * (s - p1_2) * (s - p1_3) * (s - p2_3)) ** 0.5
        return area

    sch_area = _area(sch_point)
    src_area = _area(src_point)

    try:
        scale = src_area / sch_area  # 计算缩放大小
    except ZeroDivisionError:
        if sch_area == src_area:
            scale = 1
        else:
            return None, None

    h, w = im_search.shape[:2]
    _h, _w = h * scale, w * scale
    src = np.float32(rectangle_transform(point=sch_point[0].pt, size=(h, w), mapping_point=src_point[0].pt,
                                         mapping_size=(_h, _w), angle=angle))
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    output = _perspective_transform(im_source=im_source, im_search=im_search, src=src, dst=dst)
    rect = _get_perspective_area_rect(im_source=im_source, src=src)
    return output, rect

def _handle_many_good_points(im_source, im_search, kp_src, kp_sch, good):
    """
    特征点匹配数量>=4时,使用单矩阵映射,获取识别的目标图片

    Args:
        im_source: 待匹配图像
        im_search: 图片模板
        kp_sch: 关键点集
        kp_src: 关键点集
        good: 描述符集

    Returns:
        透视变换后的图片
    """

    sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in good]).reshape(
        -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # M是转化矩阵
    M, mask = _find_homography(sch_pts, img_pts)
    # 计算四个角矩阵变换后的坐标，也就是在大图中的目标区域的顶点坐标:
    h, w = im_search.shape[:2]
    h_s, w_s = im_source.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    try:
        dst: np.ndarray = cv2.perspectiveTransform(pts, M)
        # img = im_source.clone().data
        # img2 = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # Image(img).imshow('dst')
        pypts = [tuple(npt[0]) for npt in dst.tolist()]
        src = np.array([pypts[0], pypts[3], pypts[1], pypts[2]], dtype=np.float32)
        dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        output = _perspective_transform(im_source=im_source, im_search=im_search, src=src, dst=dst)
    except cv2.error as err:
        import traceback
        traceback.print_exc()
        raise err

    rect = _get_perspective_area_rect(im_source=im_source, src=src)
    return output, rect


def cal_rgb_confidence(im_source, im_search):
    """
    计算两张图片图片rgb三通道的置信度

    Args:
        im_source: 待匹配图像
        im_search: 图片模板

    Returns:
        float: 最小置信度
    """
    # im_search = im_search.copyMakeBorder(10, 10, 10, 10, cv2.BORDER_REPLICATE)
    #
    # img_src_hsv = im_source.cvtColor(cv2.COLOR_BGR2HSV)
    # img_sch_hsv = im_search.cvtColor(cv2.COLOR_BGR2HSV)

    # src_split = im_source.split()
    # sch_split = im_search.split()
    src_split = cv2.split(im_source)
    sch_split = cv2.split(im_search)

    # 计算BGR三通道的confidence，存入bgr_confidence:
    bgr_confidence = [0, 0, 0]
    for i in range(3):
        res_temp = cv2.matchTemplate(sch_split[i], src_split[i], cv2.TM_CCOEFF_NORMED)
        # res_temp = self.match(sch_split[i], src_split[i])
        # self.minMaxLoc(res_temp)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_temp)
        bgr_confidence[i] = max_val

    return min(bgr_confidence)

def cal_ccoeff_confidence(im_source, im_search):
    if len(im_source.shape) == 3 and im_source.shape[2] == 3:
        img_src_gray = im_source.cvtColor(cv2.COLOR_BGR2GRAY)
    else:
        img_src_gray = im_source

    if len(im_search.shape) == 3 and im_search.shape[2] == 3:
        img_sch_gray = im_search.cvtColor(cv2.COLOR_BGR2GRAY)
    else:
        img_sch_gray = im_search

    res_temp = cv2.matchTemplate(img_sch_gray, img_src_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_temp)
    return max_val

def _cal_confidence(im_source, im_search, rgb = True):
    """
    将截图和识别结果缩放到大小一致,并计算可信度

    Args:
        im_source: 待匹配图像
        im_search: 图片模板
        rgb:是否使用rgb通道进行校验

    Returns:

    """
    h, w = im_source.shape[:2]
    # im_search = im_search.resize(w, h)
    im_search = cv2.resize(im_search, (w, h))
    if rgb:
        confidence = cal_rgb_confidence(im_source=im_source, im_search=im_search)
    else:
        confidence = cal_ccoeff_confidence(im_source=im_source, im_search=im_search)

    confidence = (1 + confidence) / 2
    return confidence

def extract_good_points( im_source, im_search, kp_src, kp_sch, good, angle, rgb):
    """
    根据匹配点(good)数量,提取识别区域

    Args:
        im_source: 待匹配图像
        im_search: 图片模板
        kp_src: 关键点集
        kp_sch: 关键点集
        good: 描述符集
        angle: 旋转角度
        rgb: 是否使用rgb通道进行校验

    Returns:
        范围,和置信度
    """
    len_good = len(good)
    confidence, rect, target_img = None, None, None

    if len_good == 0:
        pass
    elif len_good == 1:
        target_img, rect = _handle_one_good_points(im_source=im_source, im_search=im_search,
                                                        kp_sch=kp_sch, kp_src=kp_src, good=good, angle=angle)
    elif len_good == 2:
        target_img, rect = _handle_two_good_points(im_source=im_source, im_search=im_search,
                                                        kp_sch=kp_sch, kp_src=kp_src, good=good, angle=angle)
    elif len_good == 3:
        target_img, rect = _handle_three_good_points(im_source=im_source, im_search=im_search,
                                                          kp_sch=kp_sch, kp_src=kp_src, good=good, angle=angle)
    else:  # len > 4
        target_img, rect = _handle_many_good_points(im_source=im_source, im_search=im_search,
                                                         kp_sch=kp_sch, kp_src=kp_src, good=good)

    if target_img.size:
        confidence = _cal_confidence(im_source=im_search, im_search=target_img, rgb=rgb)

    return rect, confidence



def generate_result(rect, confi):
    """Format the result: 定义图像识别结果格式."""
    ret = {
        'rect': rect,
        'confidence': confi,
    }
    return ret

def find_template_result(im_search, im_source, kp1, kp2, matches,threshold=0.8,max_count=10, max_iter_counts=20,distance_threshold=150,rgb=False):
    """
        通过特征点匹配,在im_source中找到全部符合im_search的范围

        Args:
            im_source: 待匹配图像
            im_search: 图片模板
            threshold: 识别阈值(0~1)
            rgb: 是否使用rgb通道进行校验
            max_count: 最多可以返回的匹配数量
            max_iter_counts: 最大的搜索次数,需要大于max_count
            distance_threshold: 距离阈值,特征点(first_point)大于该阈值后,不做后续筛选

        Returns:

    """
    # ===========================
    kp_src, kp_sch = list(kp2), list(kp1)
    # 在特征点集中,匹配最接近的特征点
    matches = np.array(matches)
    kp_sch_point = np.array([(kp.pt[0], kp.pt[1], kp.angle) for kp in kp_sch])
    kp_src_matches_point = np.array([[(*kp_src[dMatch.trainIdx].pt, kp_src[dMatch.trainIdx].angle)
                                      if dMatch else np.nan for dMatch in match] for match in matches])
    _max_iter_counts = 0
    src_pop_list = []
    result = []
    while True:
        # 这里没有用matches判断nan, 是因为类型不对
        if (np.count_nonzero(~np.isnan(kp_src_matches_point)) == 0) or (len(result) == max_count) or (
                _max_iter_counts >= max_iter_counts):
            break
        _max_iter_counts += 1
        filtered_good_point, angle, first_point = filter_good_point(matches=matches, kp_src=kp_src,
                                                                         kp_sch=kp_sch,
                                                                         kp_sch_point=kp_sch_point,
                                                                         kp_src_matches_point=kp_src_matches_point)
        if first_point.distance > distance_threshold:
            break

        rect, confidence = None, 0
        try:
            rect, confidence = extract_good_points(im_source=im_source, im_search=im_search, kp_src=kp_src,
                                                        kp_sch=kp_sch, good=filtered_good_point, angle=angle, rgb=rgb)
            # print(f'good:{len(filtered_good_point)}, rect={rect}, confidence={confidence}')
        except Exception as e:
            print(f'extract_good_points error:{e}')
            pass
        finally:

            if rect and confidence >= threshold:
                # br右下 tl左上
                br, tl = (rect[0]+rect[2],rect[1]+rect[3]), rect[:2]
                # 移除 范围内的所有特征点 ??有可能因为透视变换的原因，删除了多余的特征点
                for index, match in enumerate(kp_src_matches_point):
                    x, y = match[:, 0], match[:, 1]
                    flag = np.argwhere((x < br[0]) & (x > tl[0]) & (y < br[1]) & (y > tl[1]))
                    for _index in flag:
                        src_pop_list.append(matches[index, _index][0].trainIdx)
                        kp_src_matches_point[index, _index, :] = np.nan
                        matches[index, _index] = np.nan
                result.append(generate_result(rect, confidence))
            else:
                for match in filtered_good_point:
                    flags = np.argwhere(matches[match.queryIdx, :] == match)
                    for _index in flags:
                        kp_src_matches_point[match.queryIdx, _index, :] = np.nan
                        matches[match.queryIdx, _index] = np.nan

    return result


def match_template(im_search, im_source):
    """
    使用sift算法进行模板匹配
    """

    s = time.time()
    if im_search is None or im_source is None:
        return None
    if len(im_search.shape) == 3:
        im_search = cv2.cvtColor(im_search, cv2.COLOR_BGR2GRAY)
    if len(im_source.shape) == 3:
        im_source = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)
    # 初始化创建
    sift = cv2.SIFT_create()

    # 关键点检测和特征计算
    kp1, des1 = sift.detectAndCompute(im_search, None)
    kp2, des2 = sift.detectAndCompute(im_source, None)
    if len(kp1) == 0 or len(kp2) == 0:
        return []
    # FLANN parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 第一个参数是模版，第二个是原图
    matches = flann.knnMatch(des1, des2, k=2)
    res = find_template_result(im_search, im_source, kp1, kp2, matches, threshold=0.8)
    print(f'模版匹配耗时检测，time:{int((time.time() - s)*1000)}ms')
    return res


def match_template_best(im_search, im_source, *crop, resize_rate=1):
    """
    使用sift算法进行模板匹配，并且取出匹配度最高的那一个
    :param im_search: 模版图
    :param im_source: 原图
    :param crop: 截图的区域 用于指定从原图的匹配区域 (x,y,w,h)
    :param resize_rate: 原图的缩放比例，默认1，数值越小，匹配越快，但也越不准确
    :return:
    """
    try:
        # 截取im_source的指定坐标范围部分来匹配
        x, y, w, h = 0, 0, im_source.shape[1], im_source.shape[0]
        if len(crop) > 0:
            x, y, w, h = crop[0]
        # 截剪im_source
        im_source = im_source[y:y + h, x:x + w]
        # 缩放im_source
        im_source = cv2.resize(im_source, dsize=(int(im_source.shape[1]*resize_rate), int(im_source.shape[0]*resize_rate)))
        # im_search = cv2.resize(im_search, dsize=(int(im_search.shape[1]*resize_rate), int(im_search.shape[0]*resize_rate)))

        res = match_template(im_search, im_source)
        res = sorted(res, key=lambda l: l['confidence'], reverse=True)
        res = res[0] if res else None
        if res is None:
            return res
        # 恢复到原图的坐标
        rect = res['rect']
        if rect:
            res['rect'] = (int(rect[0] / resize_rate + x), int(rect[1] / resize_rate + y), int(rect[2] / resize_rate), int(rect[3] / resize_rate))
        return res
    except Exception as e:
        print(f'match_template_best error:{e}')
        return None


if __name__ == '__main__':
    # 模版
    img1 = cv2.imread("../../template/再次挑战按钮.jpg", 0)
    # 原图
    img2 = cv2.imread("../../template/jh2_screen_263.jpg")
    s = time.time()
    crop = (1090, 60, 180, 180)
    result = match_template_best(img1, img2 ,crop)
    print(f'time:{(time.time() - s)*1000}ms')
    print(result)
    print(result['rect'])
    print(result['confidence'])
    cv2.rectangle(img2, result['rect'], (0, 255, 0), 2)

    cv2.imwrite("result.jpg", img2)


# opencv-contrib-python-headless 4.10.0.84
# opencv-python                  4.9.0.80
# opencv-python-headless         4.10.0.84

#
# # 初始化创建
# sift = cv2.SIFT_create()
#
# # 关键点检测和特征计算
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
#
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# # 第一个参数是模版，第二个是原图
# matches = flann.knnMatch(des1, des2, k=2)
#
# results = find_template_result(img1, img2, kp1, kp2, matches,threshold = 0.5)
# print(results)
# # cv2.rectangle(img2, results[0]['rect'], (255, 255, 0), 2)
# # cv2.imshow("sift-flannmatches", img2)
# # cv2.waitKey(0)
#
#
# # Need to draw only good matches, so create a mask
# matchesMask = [[0, 0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.5*n.distance:
#         matchesMask[i] = [1, 0]
#
#
# good = []
# for m, n in matches:
#     if m.distance < 0.5*n.distance:
#         good.append([m])
#
#
#
#
#
#
#
#
#
#
# img_res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
#                             None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("sift-flannmatches1", img_res)
# cv2.waitKey(0)
#
# draw_params = dict(matchColor=(0, 255, 0),
#                     singlePointColor=(255, 0, 0),
#                     matchesMask=matchesMask,
#                     flags=cv2.DrawMatchesFlags_DEFAULT)
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
#
# cv2.imshow("sift-flannmatches", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

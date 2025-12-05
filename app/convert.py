import numpy as np
import logging

logger = logging.getLogger(__name__)



def convert_masks_to_json(
    masks,
    input_boxes,
    labels=None,
    confidences=None,
    w=None,
    h=None,
    epsilon=1.0,
    enable_visualization=False,
    original_image_path=None,
):
    """
    将 numpy 格式的分割结果转换为前端需要的 JSON 格式。
    支持SAM2和SAM3模型的结果处理。

    支持 numpy.ndarray 和 PyTorch tensor：

    Args:
        masks:       list[np.ndarray] 或 list[torch.Tensor]，每个元素为 mask (H, W)，dtype=bool 或 0/1
        input_boxes: list[np.ndarray] 或 list[list/tuple] 或 list[torch.Tensor]，每个为 (x1, y1, x2, y2)
        labels:      list[str] 或 list[int] 或 str 或 int 或 None，类别标签
                     - None: 使用默认标签 "object"
                     - 单个值: 应用到所有mask
                     - 列表: 每���mask对应一个标签
        confidences: list[float] 或 float 或 None，置信度
                     - None: 使用默认置信度 1.0
                     - 单个值: 应用到所有mask
                     - 列表: 每个mask对应一个置信度
        w:           图像宽度，如果为None则从第一个mask推断
        h:           图像高度，如果为None则从第一个mask推断
        epsilon:     多边形简化精度参数
        enable_visualization: 是否生成可视化图片
        original_image_path: 原始图片路径（支持本地文件路径和HTTP URL，可视化时需要）

    Returns:
        dict: 用于前端的 JSON 对象，包含可视化图片路径（如果启用）
    """
    try:
        # 检查 PyTorch 是否可用
        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False
        
        def _to_numpy(data):
            """将 tensor 或其他格式转换为 numpy array"""
            if isinstance(data, np.ndarray):
                return data
            elif HAS_TORCH and isinstance(data, torch.Tensor):
                return data.cpu().detach().numpy()
            elif isinstance(data, (list, tuple)):
                return np.array(data)
            else:
                raise TypeError(f"不支持的数据类型: {type(data)}")
        
        # 参数验证和预处理
        if masks is None or len(masks) == 0:
            raise ValueError("masks 不能为空")
        
        # 推断图像尺寸（如果未提供）
        if w is None or h is None:
            first_mask = _to_numpy(masks[0])
            if not isinstance(first_mask, np.ndarray):
                raise TypeError(f"mask 转换后应该是 numpy.ndarray，但得到类型: {type(first_mask)}")
            
            mask_h, mask_w = first_mask.shape[:2]
            if w is None:
                w = mask_w
            if h is None:
                h = mask_h
        
        # 标准化 labels 参数
        if labels is None:
            # 没有提供标签，使用默认标签
            labels = ["object"] * len(masks)
        elif isinstance(labels, (str, int)):
            # 单个标签，应用到所有 mask
            labels = [labels] * len(masks)
        elif isinstance(labels, list):
            # 标签列表，检查长度是否匹配
            if len(labels) != len(masks):
                raise ValueError(f"labels 长度 ({len(labels)}) 与 masks 长度 ({len(masks)}) 不匹配")
        else:
            raise TypeError(f"labels 类型不支持: {type(labels)}")
        
        # 标准化 confidences 参数
        if confidences is None:
            # 没有提供置信度，使用默认值
            confidences = [1.0] * len(masks)
        elif isinstance(confidences, (int, float)):
            # 单个置信度，应用到所有 mask
            confidences = [float(confidences)] * len(masks)
        elif isinstance(confidences, list):
            # 置信度列表，检查长度是否匹配
            if len(confidences) != len(masks):
                raise ValueError(f"confidences 长度 ({len(confidences)}) 与 masks 长度 ({len(masks)}) 不匹配")
            confidences = [float(c) for c in confidences]
        else:
            raise TypeError(f"confidences 类型不支持: {type(confidences)}")
        
        # 检查 input_boxes 长度
        if len(input_boxes) != len(masks):
            raise ValueError(f"input_boxes 长度 ({len(input_boxes)}) 与 masks 长度 ({len(masks)}) 不匹配")
        
        # 转换 input_boxes 为 numpy arrays
        input_boxes_np = [_to_numpy(box) for box in input_boxes]

        results = []
        visualization_path = None
        skipped_count = 0  # 记录跳过的数量
        skipped_reasons = {}  # 记录跳过原因统计

        # 如果启用可视化，先准备可视化环境
        if enable_visualization:
            # visualization_path = _generate_visualization(
            #     masks, input_boxes, labels, confidences, w, h, original_image_path
            # )
            pass  # 暂时禁用可视化功能

        for idx in range(len(masks)):
            mask = _to_numpy(masks[idx])
            box = input_boxes_np[idx]
            label = labels[idx]
            confidence = confidences[idx]

            # 强制转换为 numpy
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"mask 转换后应该是 numpy.ndarray，但得到类型: {type(mask)}")

            # 置信度转 float（
            confidence = float(confidence)

            # 确保标签不为空（已在预处理中处理，这里作为保险）
            label_str = str(label).strip()
            if not label_str:
                label_str = "object"  # 使用默认标签
            
            # 提取多边形轮廓
            try:
                polygon = _mask_to_polygon_json(
                    mask=mask,
                    box=box,
                    label=label_str,
                    score=confidence,
                    order=idx + 1,
                    epsilon=float(epsilon),
                )
            except Exception as e:
                skipped_count += 1
                skipped_reasons['polygon_conversion_error'] = skipped_reasons.get('polygon_conversion_error', 0) + 1
                logger.warning(f"结果 {idx+1} (label={label_str}) 多边形转换失败: {e}")
                continue

            if polygon is not None:
                # 保证 bbox 为 list
                bbox = box.tolist() if isinstance(box, np.ndarray) else list(box)

                results.append(
                    {
                        "id": polygon.get("id"),
                        "type": polygon.get("type"),
                        "points": polygon.get("points"),
                        "label": polygon.get("label"),
                        "score": polygon.get("score"),
                        "order": polygon.get("order"),
                        "bbox": bbox,
                    }
                )
            else:
                skipped_count += 1
                skipped_reasons['no_contour'] = skipped_reasons.get('no_contour', 0) + 1
                logger.debug(f"跳过结果 {idx+1} (label={label_str}): 未找到有效轮廓")
        
        # 记录转换统计信息
        if skipped_count > 0:
            logger.warning(
                f"多边形转换统计: 总输入={len(masks)}, 成功={len(results)}, 跳过={skipped_count}, "
                f"跳过原因={skipped_reasons}"
            )

        result_dict = {
            "status": "success",
            "results": results,
            "count": len(results),
            "image_shape": {
                "width": w,
                "height": h,
            },
        }

        # 如果有可视化图片，添加到结果中
        if visualization_path:
            result_dict["visualization_path"] = visualization_path

        return result_dict

    except Exception as e:
        raise Exception(f"convert_masks_to_json error: {e}")


def convert_sam3_masks_to_json(
    masks,
    input_boxes,
    label="object",
    confidence=1.0,
    epsilon=1.0,
    enable_visualization=False,
    original_image_path=None,
    w=None,
    h=None,
):
    """
    SAM3专用的mask转换函数，简化调用接口。
    
    Args:
        masks: list[np.ndarray]，每个元素为 mask (H, W)，dtype=bool 或 0/1
        input_boxes: list[np.ndarray] 或 list[list/tuple]，每个为 (x1, y1, x2, y2)
        label: str，统一的标签名称，默认为 "object"
        confidence: float，统一的置信度，默认为 1.0
        epsilon: float，多边形简化精度参数
        enable_visualization: bool，是否生成可视化图片
        original_image_path: str，原始图片路径
        w:           图像宽度，如果为None则从第一个mask推断
        h:           图像高度，如果为None则从第一个mask推断
    Returns:
        dict: 用于前端的 JSON 对象
        
    Example:
        ```python
        # 基本用法
        result = convert_sam3_masks_to_json(masks, boxes)
        
        # 自定义标签
        result = convert_sam3_masks_to_json(masks, boxes, label="person")
        
        # 自定义置信度
        result = convert_sam3_masks_to_json(masks, boxes, label="car", confidence=0.95)
        ```
    """
    return convert_masks_to_json(
        masks=masks,
        input_boxes=input_boxes,
        labels=label,
        confidences=confidence,
        epsilon=epsilon,
        enable_visualization=enable_visualization,
        original_image_path=original_image_path,
        w=w,
        h=h,
    )





def _mask_to_polygon_json(mask, box, label, score, order, epsilon=1.0):
    """
    将 mask 转换为前端多边形 JSON 格式（参考 grounded_sam2_local_demo.py）

    支持 numpy.ndarray 和 PyTorch tensor（自动转换为 numpy）。

    Args:
        mask:   全图尺寸的布尔 mask (H, W)，numpy.ndarray 或 torch.Tensor
        box:    边界框 (x1, y1, x2, y2) 原图坐标系，numpy.ndarray 或 list/tuple 或 torch.Tensor
        label:  类别标签
        score:  检测框置信度分数
        order:  顺序编号
        epsilon: 多边形简化精度参数

    Returns:
        dict | None: 前端多边形 JSON 对象，如果没有有效轮廓则返回 None
    """
    try:
        import cv2
        import random
        import string
        
        # 检查 PyTorch 是否可用
        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False
        
        def _to_numpy_local(data):
            """将 tensor 或其他格式转换为 numpy array"""
            if isinstance(data, np.ndarray):
                return data
            elif HAS_TORCH and isinstance(data, torch.Tensor):
                return data.cpu().detach().numpy()
            elif isinstance(data, (list, tuple)):
                return np.array(data)
            else:
                raise TypeError(f"不支持的数据类型: {type(data)}")
        
        # 确保 mask 和 box 都是 numpy arrays
        mask = _to_numpy_local(mask)
        box = _to_numpy_local(box)

        def generate_random_id():
            """生成随机 ID（11 位，小写字母 + 数字）"""
            chars = string.digits + string.ascii_lowercase
            return "".join(random.choices(chars, k=11))

        def extract_mask_contour_from_box(mask_, box_):
            """从全图 mask 中提取框内区域的轮廓（只支持 numpy）"""
            if not isinstance(mask_, np.ndarray):
                raise TypeError(f"mask 应该已经转换为 numpy.ndarray，但得到类型: {type(mask_)}")

            # 确保 mask 是 2 维数组
            if mask_.ndim != 2:
                if mask_.ndim == 3:
                    mask_ = mask_.squeeze(0)
                else:
                    raise ValueError(
                        f"mask 应该是2维数组 (H, W)，但得到 {mask_.ndim} 维，形状: {mask_.shape}"
                    )

            # box 也转成 numpy
            if not isinstance(box_, np.ndarray):
                box_ = np.array(box_)
            x1, y1, x2, y2 = box_.astype(int)

            # 确保坐标在有效范围内
            mask_h, mask_w = mask_.shape
            x1_actual = max(0, min(x1, mask_w - 1))
            y1_actual = max(0, min(y1, mask_h - 1))
            x2_actual = max(x1_actual + 1, min(x2, mask_w))
            y2_actual = max(y1_actual + 1, min(y2, mask_h))

            actual_box_ = np.array([x1_actual, y1_actual, x2_actual, y2_actual])

            # 检查裁剪区域是否有效
            if x2_actual <= x1_actual or y2_actual <= y1_actual:
                return [], actual_box_

            # 裁剪到框内区域（注意：mask 索引是 [y, x] 顺序）
            box_mask = (
                mask_[y1_actual:y2_actual, x1_actual:x2_actual].astype(np.uint8) * 255
            )

            if box_mask.sum() == 0:
                return [], actual_box_

            # 提取轮廓（只提取外部轮廓）
            contours_, _ = cv2.findContours(
                box_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            contour_list_ = []
            for contour in contours_:
                if len(contour) >= 3:
                    contour_2d = contour.reshape(-1, 2).astype(float)
                    contour_list_.append(contour_2d)

            return contour_list_, actual_box_

        def simplify_polygon(contour, epsilon_=2.0):
            """使用 Douglas-Peucker 算法简化多边形"""
            if not isinstance(contour, np.ndarray):
                contour = np.array(contour)

            if len(contour) < 3:
                return contour

            # 确保 (N, 2)
            if contour.ndim == 1:
                if len(contour) == 2:
                    contour = contour.reshape(1, 2)
                else:
                    raise ValueError(f"contour 形状无效: {contour.shape}")
            elif contour.ndim == 2:
                if contour.shape[1] != 2:
                    raise ValueError(
                        f"contour 应该是 (N, 2) 形状，但得到 {contour.shape}"
                    )
            else:
                raise ValueError(f"contour 应该是2维数组，但得到 {contour.ndim} 维")

            contour_int = contour.astype(np.int32)

            if contour_int.ndim == 2:
                contour_int = contour_int.reshape(-1, 1, 2)

            epsilon_val = epsilon_ * cv2.arcLength(contour_int, closed=True) / 100.0

            simplified = cv2.approxPolyDP(contour_int, epsilon_val, closed=True)

            if simplified.shape[0] == 0:
                return contour

            return simplified.reshape(-1, 2).astype(float)

        def local_to_global_coords(local_points, box_, actual_box_):
            """将框局部坐标系转换为原图全局坐标系"""
            if not isinstance(local_points, np.ndarray):
                local_points = np.array(local_points)
            if not isinstance(box_, np.ndarray):
                box_ = np.array(box_)
            if not isinstance(actual_box_, np.ndarray):
                actual_box_ = np.array(actual_box_)

            if local_points.ndim == 1:
                local_points = local_points.reshape(1, -1)
            if local_points.shape[1] != 2:
                raise ValueError(
                    f"local_points 应该是 (N, 2) 形状，但得到 {local_points.shape}"
                )

            x1_actual, y1_actual, x2_actual, y2_actual = actual_box_.astype(float)
            actual_box_w = x2_actual - x1_actual
            actual_box_h = y2_actual - y1_actual

            x1_orig, y1_orig, x2_orig, y2_orig = box_.astype(float)
            orig_box_w = x2_orig - x1_orig
            orig_box_h = y2_orig - y1_orig

            global_points = local_points.copy()

            if actual_box_w != orig_box_w or actual_box_h != orig_box_h:
                scale_x = orig_box_w / actual_box_w if actual_box_w > 0 else 1.0
                scale_y = orig_box_h / actual_box_h if actual_box_h > 0 else 1.0

                global_points[:, 0] = local_points[:, 0] * scale_x
                global_points[:, 1] = local_points[:, 1] * scale_y

                global_points[:, 0] = x1_orig + global_points[:, 0]
                global_points[:, 1] = y1_orig + global_points[:, 1]
            else:
                global_points[:, 0] = x1_orig + local_points[:, 0]
                global_points[:, 1] = y1_orig + local_points[:, 1]

            return global_points

        # 步骤1: 从 mask 提取框内轮廓
        contours, actual_box = extract_mask_contour_from_box(mask, box)

        if not contours:
            # 记录为什么没有找到轮廓（用于调试）
            mask_sum = mask.sum() if isinstance(mask, np.ndarray) else 0
            box_str = f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]" if isinstance(box, (list, np.ndarray)) and len(box) >= 4 else str(box)
            logger.debug(
                f"未找到有效轮廓: mask_sum={mask_sum}, box={box_str}, "
                f"mask_shape={mask.shape if isinstance(mask, np.ndarray) else 'N/A'}"
            )
            return None

        # 选择最大的轮廓作为主要轮廓
        main_contour = max(contours, key=len)

        # 步骤2: 简化轮廓
        simplified_contour = simplify_polygon(main_contour, epsilon_=epsilon)

        # 步骤3: 局部坐标转换为全局坐标
        global_points = local_to_global_coords(simplified_contour, box, actual_box)

        # 步骤4: 组装前端 JSON
        polygon_id = generate_random_id()
        points = [
            {
                "id": generate_random_id(),
                "x": float(x),
                "y": float(y),
            }
            for x, y in global_points
        ]

        polygon_json = {
            "id": polygon_id,
            "type": "line",
            "points": points,
            "label": label,
            "score": float(score),
            "order": int(order),
        }

        return polygon_json

    except Exception as e:
        raise Exception(f"convert_masks_to_json error: {e}")


import os
import cv2
import numpy as np

def draw_annotations_and_save(
    image_path: str,
    convert_result: dict,
    save_path: str,
    close_polygon_if_needed: bool = True,  # 对非闭合点集是否自动闭合
    draw_bbox: bool = False,               # 是否画bbox
    draw_points: bool = True,              # 是否画顶点
    thickness: int = 3,
):
    """
    将 convert_result['results'] 中的 points 画到 image_path 上，并保存到 save_path。
    - 支持 type: 'line' / 'polygon'（实际上只要有points都能画）
    - 自动裁剪到图像范围内，避免越界点(-1之类)造成问题
    """
    print("1111111111111111",image_path)
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]
    results = (convert_result or {}).get("results", [])

    # 给不同label定个颜色（稳定：同label同色）
    def color_for_label(label: str):
        # 简单hash到颜色（BGR）
        h = abs(hash(label)) % (256 * 256 * 256)
        b = (h) & 255
        g = (h >> 8) & 255
        r = (h >> 16) & 255
        # 避免太暗
        return (max(b, 60), max(g, 60), max(r, 60))

    # 画每个实例
    for i, ann in enumerate(results):
        label = str(ann.get("label", "unknown"))
        score = ann.get("score", None)
        order = ann.get("order", i + 1)
        pts = ann.get("points", [])
        if not pts:
            continue

        color = color_for_label(label)

        # 组装点列表 -> int32
        xy = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)

        # 裁剪到图像范围（你的数据里有 x/y = -1 的情况）
        xy[:, 0] = np.clip(xy[:, 0], 0, W - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, H - 1)

        poly = xy.astype(np.int32).reshape((-1, 1, 2))

        # 决定是否闭合：如果type明确是line就不闭合；否则按参数
        ann_type = str(ann.get("type", "")).lower()
        is_closed = False if ann_type == "line" else close_polygon_if_needed

        # 画轮廓/折线
        cv2.polylines(img, [poly], isClosed=is_closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # 画点
        if draw_points:
            for (x, y) in xy.astype(np.int32):
                cv2.circle(img, (int(x), int(y)), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)

        # 画bbox（可选）
        if draw_bbox and "bbox" in ann and ann["bbox"]:
            x1, y1, x2, y2 = ann["bbox"]
            x1 = int(np.clip(x1, 0, W - 1)); y1 = int(np.clip(y1, 0, H - 1))
            x2 = int(np.clip(x2, 0, W - 1)); y2 = int(np.clip(y2, 0, H - 1))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        # 写文字（label / score / order）
        text = f"{order}:{label}"
        if score is not None:
            text += f" ({float(score):.2f})"

        # 文本放在第一个点附近
        tx, ty = xy[0].astype(int)
        ty = max(20, ty)
        cv2.putText(img, text, (tx, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise IOError(f"Failed to write image: {save_path}")
    return save_path

# ============ 辅助：一个简单的 ID 生成器，替代 global ============

@dataclass
class IdGenerator:
    """简单的递增 ID 生成器，保证在当前进程内唯一。"""
    current: int = 1

    def next(self) -> int:
        value = self.current
        self.current += 1
        return value


# ============ 辅助函数：为单张图构造 Datapoint ============

def build_datapoint_for_image(
    image: Image.Image,
    text_prompts: List[str],
    transform: ComposeAPI,
    id_gen: IdGenerator,
) -> Tuple[Datapoint, List[int]]:
    """
    为一张图片构造一个 Datapoint，并挂上多条 text prompt。
    返回:
        datapoint: 经过 transform 之后可直接 collate 的 Datapoint
        query_ids: 与 text_prompts 一一对应的 query_id 列表
    """
    w, h = image.size

    # 1. 构造基础 datapoint（1 张图 + 空的 query 列表）
    datapoint = Datapoint(
        find_queries=[],
        images=[SAMImage(data=image, objects=[], size=[w, h])],
    )

    # 2. 为该图片附加多条 text prompt
    query_ids: List[int] = []

    for text_query in text_prompts:
        qid = id_gen.next()
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=qid,
                    original_image_id=qid,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        query_ids.append(qid)

    # 3. 做 API transform（resize + to_tensor + normalize）
    datapoint = transform(datapoint)

    return datapoint, query_ids


# ============ 主函数：多图 × 多个 text prompt 的 batch 推理 ============

def sam3_batch_text_inference(
    image_paths: List[str],
    text_prompts: List[str],
    device: str | None = None,
    detection_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    在 SAM3 上做 batch text-prompt segmentation：
        - 支持多张图片
        - 每张图片跑同一组 text_prompts

    Args:
        image_paths: 每张图片的路径列表
        text_prompts: 要在每张图上跑的文本 prompt 列表，如 ["person", "car", "truck"]
        device: "cuda" / "cpu"，默认自动选择
        detection_threshold: 后处理的 detection 阈值

    Returns:
        results_per_image: 长度 = len(image_paths)
            results_per_image[i][prompt_str] = 该图片 + 该 prompt 的分割结果
            每个结果是 postprocessor 返回的字典（里有 masks / boxes / scores 等）
    """
    # -------- 设备选择 --------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- SAM3 模型与组件 --------
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

    model = build_sam3_image_model(bpe_path=bpe_path).to(device)
    model.eval()

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=detection_threshold,
        to_cpu=False,
    )

    id_gen = IdGenerator()

    # -------- 构造所有 Datapoint --------
    datapoints: List[Datapoint] = []
    query_ids_per_image: List[List[int]] = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        datapoint, query_ids = build_datapoint_for_image(
            image=image,
            text_prompts=text_prompts,
            transform=transform,
            id_gen=id_gen,
        )
        datapoints.append(datapoint)
        query_ids_per_image.append(query_ids)

    if not datapoints:
        return []

    # -------- collate 成 batch 并搬到设备 --------
    batch = collate(datapoints, dict_key="data")["data"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    # -------- 前向推理 --------
    with torch.inference_mode():
        output = model(batch)

    # -------- 后处理 & 结果重排 --------
    processed_results: Dict[int, Any] = postprocessor.process_results(
        output, batch.find_metadatas
    )
    # processed_results: key = query_id, value = 该 query 的检测/分割结果

    results_per_image: List[Dict[str, Any]] = []

    for qids in query_ids_per_image:
        per_image_results: Dict[str, Any] = {}
        for prompt_str, qid in zip(text_prompts, qids):
            per_image_results[prompt_str] = processed_results.get(qid, None)
        results_per_image.append(per_image_results)

    return results_per_image
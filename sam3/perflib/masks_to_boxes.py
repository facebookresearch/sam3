import torch


def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        # Sanity check based on callsite for replacement
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        # Based on torchvision masks_to_boxes
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        N, H, W = masks.shape
        device = masks.device
        y = torch.arange(H, device=device).view(1, H)
        x = torch.arange(W, device=device).view(1, W)

        masks_with_obj = masks != 0  # N, H, W
        masks_with_obj_x = masks_with_obj.amax(
            dim=1
        )  # N, H (which columns have objects)
        masks_with_obj_y = masks_with_obj.amax(dim=2)  # N, W (which rows have objects)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        bounding_boxes_0 = torch.amin(
            (masks_without_obj_x * W) + (masks_with_obj_x * x), dim=1
        )
        bounding_boxes_1 = torch.amin(
            (masks_without_obj_y * H) + (masks_with_obj_y * y), dim=1
        )
        bounding_boxes_2 = torch.amax(masks_with_obj_x * x, dim=1)
        bounding_boxes_3 = torch.amax(masks_with_obj_y * y, dim=1)

        bounding_boxes = torch.stack(
            [bounding_boxes_0, bounding_boxes_1, bounding_boxes_2, bounding_boxes_3],
            dim=1,
        ).to(dtype=torch.float)
        assert bounding_boxes.shape == (N, 4)
        assert bounding_boxes.device == masks.device
        assert bounding_boxes.dtype == torch.float
        return bounding_boxes

        # # torchvision reference code
        # for index, mask in enumerate(masks):
        #     y, x = torch.where(mask != 0)
        #
        #     bounding_boxes[index, 0] = torch.min(x)
        #     bounding_boxes[index, 1] = torch.min(y)
        #     bounding_boxes[index, 2] = torch.max(x)
        #     bounding_boxes[index, 3] = torch.max(y)
        #
        # return bounding_boxes

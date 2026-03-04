
from nesvor.preprocessing import n4_bias_field_correction, assess, brain_segmentation
from nesvor.image import Stack
from typing import List
def _segment_stack(args, data: List[Stack]) -> List[Stack]:
    if args.nz_mask:
        for i, stack in enumerate(data):
            stack.mask = stack.slices>0
            if not stack.mask.any():
                logging.warning(
                    "One of the input stack is all zero after brain segmentation. Please check your data!"
                )
    else:
        data = brain_segmentation(
            data,
            args.device,
            args.batch_size_seg,
            not args.no_augmentation_seg,
            args.dilation_radius_seg,
            args.threshold_small_seg,
        )
    return data
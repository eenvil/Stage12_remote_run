import numpy as np

def align_video_depth(depth_list, INFER_LEN, KEYFRAMES, OVERLAP):
    depth_list = [depth.squeeze().cpu().numpy() for depth in depth_list]

    depth_list_aligned = []
    ref_align = []

    for i in range(0, len(depth_list)):
        if i == 0:
            depth_list_aligned.extend(depth_list[i][j] for j in range(INFER_LEN))
            for kf_id in KEYFRAMES:
                ref_align.append(depth_list[0][kf_id])
        if i != 0:
            cur_align = []
            for k in range(len(KEYFRAMES)):
                cur_align.append(depth_list[i][k])
            scale, shift = compute_scale_and_shift_full(np.concatenate(cur_align),
                                np.concatenate(ref_align),
                                np.concatenate(np.ones_like(ref_align)==1))

            new_depth = depth_list[i] * scale + shift
            depth_list_aligned.extend(new_depth[j] for j in range(OVERLAP, INFER_LEN))

            ref_align = ref_align[:1]
            for kf_id in KEYFRAMES[1:]:
                ref_align.append(new_depth[kf_id])
    return np.stack(depth_list_aligned, axis=0)


def compute_scale_and_shift_full(prediction, target, mask):
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1
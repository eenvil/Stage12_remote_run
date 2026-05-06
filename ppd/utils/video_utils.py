# This file is originally from Video Depth Anything
import numpy as np
import matplotlib.cm as cm
import imageio
import cv2

def read_video_frames(video_path, process_length=-1, target_fps=-1, max_res=-1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if target_fps < 0 else target_fps

    stride = max(round(original_fps / fps), 1)

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (process_length > 0 and frame_count >= process_length):
            break
        if frame_count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            if max_res > 0 and max(original_height, original_width) > max_res:
                frame = cv2.resize(frame, (width, height))  # Resize frame
            frames.append(frame)
        frame_count += 1
    cap.release()
    frames = np.stack(frames, axis=0)

    return frames, fps

def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()
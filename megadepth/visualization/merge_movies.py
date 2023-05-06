"""Script to merge movies of the same scene."""
import os

import cv2
import numpy as np
from tqdm import tqdm


def merge_frame(frame0, frame1, t):
    """Define here the transition between the two movies."""
    # if t < 0.4:
    #     return frame0
    # if t > 0.6:
    #     return frame1
    s = t
    cut = int(frame0.shape[1] * s)
    if cut > 0:
        frame0[:, -cut:] = frame1[:, -cut:]
        frame0[:, -cut] = 0
    return frame0


def blend_frame(frame0, frame1, t):
    """Define here the transition between the two movies."""
    # if t < 0.4:
    #     return frame0
    # if t > 0.6:
    #     return frame1
    s = t
    if t > 0:
        frame0 = (frame0 * (1 - s) + s * frame1).astype(np.uint8)
    return frame0


def merge_videos(filepaths, output_path, labels=False, blend=False, n_frames_per_cycle=200):
    """Takes a list of paths to movies.

    assume all have the same length
    And creates a movie that flips through them.
    """
    caps = [cv2.VideoCapture(filepath) for filepath in filepaths]

    n_videos = len(filepaths)
    length = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_rate = int(caps[0].get(cv2.CAP_PROP_XI_FRAMERATE))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    current_video = 0.0
    for i in tqdm(range(length)):
        for j, cap in enumerate(caps):
            success, frame = cap.read()

            if labels:
                font = cv2.FONT_HERSHEY_COMPLEX
                org = (50, 50)
                fontScale = 2
                color = (0, 0, 0)
                thickness = 1
                frame = cv2.putText(
                    frame, labels[j], org, font, fontScale, color, thickness, cv2.LINE_AA
                )
            if not success:
                break
            if j == int(current_video) % n_videos:
                current_frame = frame
            if j == int(current_video + 1) % n_videos:
                next_frame = frame
        if blend:
            merged_frame = blend_frame(current_frame, next_frame, current_video % 1)
        else:
            merged_frame = merge_frame(current_frame, next_frame, current_video % 1)

        out.write(merged_frame)

        # cycle step
        current_video += n_videos / n_frames_per_cycle
        current_video = current_video

    for cap in caps:
        cap.release()
    out.release()


def compress(path):
    """Takes a path to a video and creates a compressed copy."""
    name, ext = os.path.splitext(path)

    cmd = "ffmpeg "
    cmd += f"-i {path} "
    cmd += "-vcodec libx264 -crf 30 -pix_fmt yuv420p "
    cmd += f"{name}_compressed{ext} -y"

    os.system(cmd)


if __name__ == "__main__":
    paths = ["mvs0.mp4", "mvs1.mp4"]
    labels = ["Baseline", "Super"]
    out_file = "out.mp4"
    merge_videos(paths, out_file, labels, blend=False)
    compress(out_file)

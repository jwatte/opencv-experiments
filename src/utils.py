import os

import cv2

def show_video(video_file):
    """Open video and show it in window."""
    video = cv2.VideoCapture(video_file)

    ret, frame = video.read()
    window_name = "Current frame"
    while frame is not None:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = video.read()
    cv2.destroyWindow(window_name)

def split_video(video_file, frame_interval, output_folder):
    """
    Split a video into individual frames.

    Open ``video``, extract frames separated by ``frame_interval``, and write
    them to ``output_folder``.
    """
    video = cv2.VideoCapture(video_file)
    ret, frame = video.read()
    i = 0
    frame_count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    while frame is not None:
        i += 1
        if i != frame_interval:
            pass
        else:
            i = 0
            filename = os.path.join(output_folder,
                                    "out_{}.jpg".format(frame_count))
            cv2.imwrite(filename, frame)
            ret, frame = video.read()
            frame_count += 1

VIDEO_FILE = "data/test.avi"
OUTPUT_FOLDER = "data/test-output/"

def main():
     show_video(VIDEO_FILE)
     split_video(VIDEO_FILE, 10, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()

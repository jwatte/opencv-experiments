import argparse

import cv2

def show_video(devices):
    """
    Open video and show it in window.

    ``devices`` should be an iterable containing the devices to use.
    """
    videos = []
    for device in devices:
        videos.append(cv2.VideoCapture(device))
    while True:
        for device, video in zip(devices, videos):
            ret, frame = video.read()
            window_name = str(device)
            cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for device in devices:
        cv2.destroyWindow(str(device))

def main():
    """
    Show the video from two webcams successively.

    For best results, connect the webcams while starting the computer.
    I have noticed that in some cases, if the webcam is not already connected
    when the computer starts, the USB device runs out of memory. Switching the
    camera to another USB port has also caused this problem in my experience.
    """
    parser = argparse.ArgumentParser(description="Show video from two "
                                     "webcams.\n\nPress 'q' to exit.")
    parser.add_argument("devices", type=int, nargs=2, help="Device numbers "
                        "for the cameras that should be accessed.")
    args = parser.parse_args()

    show_video(args.devices)

if __name__ == "__main__":
    main()

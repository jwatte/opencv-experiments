#!/bin/env python
"""A base class for working with stereo cameras."""

import argparse
import os
import time
import numpy as np

import cv2

class StereoPair(object):
    """
    A stereo pair of cameras.

    Should be initialized with a context manager to ensure that the cameras are
    freed properly after use.
    """

    def __init__(self, devices):
        """
        Initialize cameras.

        ``devices`` is an iterable containing the device numbers.
        """
        #: Video captures associated with the ``StereoPair``
        self.captures = [cv2.VideoCapture(device) for device in devices]
        #: Window names for showing captured frame from each camera
        self.windows = ["{} camera".format(side) for side in ("Left", "Right")]
        self.total_white = 0.98
        self.total_black = 0.05

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for capture in self.captures:
            capture.release()
        for window in self.windows:
            cv2.destroyWindow(window)

    def get_frames(self):
        """Get current frames from cameras."""
        ret = [capture.read()[1] for capture in self.captures]
        # apply auto-contrast
        return [self.normalize(x) for x in ret]

    # jam a sharper contrast ratio into the image
    def normalize(self, img):
        hist = [x[0] for x in cv2.calcHist([img], [0], None, [256], [0,255])]
        total = sum(hist)
        black = total * self.total_black
        white = total * self.total_white
        val = 0
        bot = -1
        top = -1
        n = 0
        for x in hist:
            val = val + x
            if bot == -1 and val >= black:
                bot = n
            if top == -1 and val >= white:
                top = n
            n += 1
        mul = 255.0 / (top - bot + 1)
        offset = - bot
        return np.array(np.clip((img + offset) * mul, 0, 255), dtype=np.uint8)

    def show_frames(self, wait=0):
        """
        Show current frames from cameras.

        ``wait`` is the wait interval before the window closes.
        """
        n = 0
        for window, frame in zip(self.windows, self.get_frames()):
            cv2.imshow(window, frame)
            cv2.moveWindow(window, n * 660 + 20, 40)
            n += 1
        cv2.waitKey(wait)

    def show_videos(self):
        """Show video from cameras."""
        while True:
            self.show_frames(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
                        "for the cameras that should be accessed in order "
                        " (left, right).")
    parser.add_argument("--output_folder",
                        help="Folder to write output images to.")
    parser.add_argument("--interval", type=float, default=1,
                        help="Interval (s) to take pictures in.")
    args = parser.parse_args()

    with StereoPair(args.devices) as pair:
        if not args.output_folder:
            pair.show_videos()
        else:
            i = 1
            while True:
                start = time.time()
                while time.time() < start + args.interval:
                    pair.show_frames(1)
                images = pair.get_frames()
                for side, image in zip(("left", "right"), images):
                    filename = "{}_{}.ppm".format(side, i)
                    output_path = os.path.join(args.output_folder, filename)
                    cv2.imwrite(output_path, image)
                i += 1

if __name__ == "__main__":
    main()

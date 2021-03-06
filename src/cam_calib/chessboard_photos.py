#!/bin/env python
"""
Module for finding chessboards with a stereo rig.
"""

import argparse
import calibrate_stereo
import os
import sys
import webcams

import cv2


class ChessboardFinder(webcams.StereoPair):
    """A ``StereoPair`` that can find chessboards."""

    def __init__(self, *args):
        super(ChessboardFinder, self).__init__(*args)
        self.n = 1

    def get_chessboard(self, columns, rows, show=False):
        """
        Take a picture with a chessboard visible in both captures.

        ``columns`` and ``rows`` should be the number of inside corners in the
        chessboard's columns and rows. ``show`` determines whether the frames
        are shown while the cameras search for a chessboard.
        """
        self.total_white = 0.95
        self.total_black = 0.25
        found_chessboard = [False, False]
        m = 0
        while not all(found_chessboard):
            m += 1
            sys.stderr.write("Looking... %r %r\r" % (self.n, m))
            frames = self.get_frames()
            if show:
                self.show_frames(1)
            # fast check isn't used in later analysis, and if fast 
            # check succeeds but regular does not, that means an 
            # error during processing. It's still not perfect, because 
            # the later processing converts to gray
            (found_chessboard[0], corners) = \
                cv2.findChessboardCorners(frames[0], (rows, columns))
            if not found_chessboard[0]:
                next
            (found_chessboard[1], corners) = \
                cv2.findChessboardCorners(frames[1], (rows, columns))

        sys.stderr.write("\nFound %r\n" % (self.n,))
        self.n += 1
        return frames

PROGRAM_DESCRIPTION=(
"Take a number of pictures with a stereo camera in which a chessboard is "
"visible to both cameras. The program waits until a chessboard is detected in "
"both camera frames. The pictures are then saved to a file in the specified "
"output folder. After five seconds, the cameras are rescanned to find another "
"chessboard perspective. This continues until the specified number of pictures "
"has been taken."
)

def main():
    """
    Take a pictures with chessboard visible to both cameras in a stereo pair.
    """
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION,
                                parents=[calibrate_stereo.CHESSBOARD_ARGUMENTS])
    parser.add_argument("left", metavar="left", type=int,
                        help="Device numbers for the left camera.")
    parser.add_argument("right", metavar="right", type=int,
                        help="Device numbers for the right camera.")
    parser.add_argument("num_pictures", type=int, help="Number of valid "
                        "chessboard pictures that should be taken.")
    parser.add_argument("output_folder", help="Folder to save the images to.")
    parser.add_argument("--calibration-folder", help="Folder to save camera "
                        "calibration to.")
    args = parser.parse_args()
    if (args.calibration_folder and not args.square_size):
        args.print_help()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    with ChessboardFinder((args.left, args.right)) as pair:
        for i in range(args.num_pictures):
            frames = pair.get_chessboard(args.columns, args.rows, True)
            print "Found frame %r of %r" % ((i+1), args.num_pictures)
            for side, frame in zip(("left", "right"), frames):
                number_string = str(i + 1).zfill(len(str(args.num_pictures)))
                filename = "{}_{}.ppm".format(side, number_string)
                output_path = os.path.join(args.output_folder, filename)
                cv2.imwrite(output_path, frame)
            for i in range(10):
                pair.show_frames(1)
    if args.calibration_folder:
        args.input_files = calibrate_stereo.find_files(args.output_folder)
        args.output_folder = args.calibration_folder
        args.show_chessboards = True
        calibrate_stereo.calibrate_folder(args)

if __name__ == "__main__":
    main()

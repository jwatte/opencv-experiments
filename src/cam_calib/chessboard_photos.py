#!/bin/env python

import argparse
import os
import time

import cv2

import webcams

class ChessboardFinder(webcams.StereoPair):
    """A ``StereoPair`` that can find chessboards."""
    
    def get_chessboard(self, columns, rows):
        """
        Take a picture with a chessboard visible in both captures.
        
        ``columns`` and ``rows`` should be the number of inside corners in the
        chessboard's columns and rows.
        """
        found_chessboard = [False, False]
        while not all(found_chessboard):
            frames = self.get_frames()
            for i, frame in enumerate(frames):
                (found_chessboard[i],
                corners) = cv2.findChessboardCorners(frame, (columns, rows))
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
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument("left", metavar="left", type=int, 
                        help="Device numbers for the left camera.")
    parser.add_argument("right", metavar="right", type=int,
                        help="Device numbers for the right camera.")
    parser.add_argument("rows", type=int, help="Number of inside corners in "
                        "the chessboard's rows.")
    parser.add_argument("columns", type=int, help="Number of inside corners in "
                        "the chessboard's columns.")
    parser.add_argument("num_pictures", type=int, help="Number of valid "
                        "chessboard pictures that should be taken.")
    parser.add_argument("output_folder", help="Folder to save the results to.")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    with ChessboardFinder((args.left, args.right)) as pair:
        for i in range(args.num_pictures):
            print("Capturing picture {}...".format(i + 1))
            frames = pair.get_chessboard(args.columns, args.rows)
            for side, frame in zip(("left", "right"), frames):
                number_string = str(i + 1).zfill(len(str(args.num_pictures)))
                filename = "{}_{}.ppm".format(side, number_string)
                output_path = os.path.join(args.output_folder, filename)
                cv2.imwrite(output_path, frame)
            time.sleep(5)

if __name__ == "__main__":
    main()

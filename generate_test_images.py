#!/usr/bin/env python3
"""
Generate a synthetic stereo pair with known disparities.

Shapes are drawn at known horizontal disparities so the resulting disparity
map can be checked by eye:
- image size: 1280x640
- circle: disparity 64 px
- square: disparity 128 px
- star:   disparity 255 px

Outputs (in the current directory):
- input_left.png / input_right.png : for the file-based sample (image_cpu)
- left/pair.png / right/pair.png   : for the directory-based sample (image_opencl)
"""

import os
import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 640

SHAPE_COLOR = (255, 255, 255)

CIRCLE_RADIUS = 80
SQUARE_SIZE = 160
STAR_SIZE = 100


def draw_star(img, center_x, center_y, size, color):
    """Draw a five-pointed star."""
    pts = []
    for i in range(5):
        # outer point
        angle = np.pi / 2 + (2 * np.pi * i / 5)
        pts.append([int(center_x + size * np.cos(angle)),
                    int(center_y - size * np.sin(angle))])
        # inner point
        angle = np.pi / 2 + (2 * np.pi * i / 5) + np.pi / 5
        pts.append([int(center_x + size * 0.4 * np.cos(angle)),
                    int(center_y - size * 0.4 * np.sin(angle))])
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    return img


def create_stereo_pair():
    """Create a left/right stereo image pair."""
    left_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    right_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    center_y = HEIGHT // 2

    # circle (disparity 64 px)
    circle_x = 300
    cv2.circle(left_img, (circle_x, center_y), CIRCLE_RADIUS, SHAPE_COLOR, -1)
    cv2.circle(right_img, (circle_x - 64, center_y), CIRCLE_RADIUS, SHAPE_COLOR, -1)

    # square (disparity 128 px)
    square_x = 640
    half = SQUARE_SIZE // 2
    cv2.rectangle(left_img, (square_x - half, center_y - half),
                  (square_x + half, center_y + half), SHAPE_COLOR, -1)
    cv2.rectangle(right_img, (square_x - 128 - half, center_y - half),
                  (square_x - 128 + half, center_y + half), SHAPE_COLOR, -1)

    # star (disparity 255 px)
    star_x = 1000
    draw_star(left_img, star_x, center_y, STAR_SIZE, SHAPE_COLOR)
    draw_star(right_img, star_x - 255, center_y, STAR_SIZE, SHAPE_COLOR)

    return left_img, right_img


def main():
    print("Generating a synthetic stereo pair...")
    print(f"  size: {WIDTH}x{HEIGHT}")
    print("  circle disparity: 64 px")
    print("  square disparity: 128 px")
    print("  star disparity:   255 px")

    left_img, right_img = create_stereo_pair()

    # Files for the file-based sample (image_cpu).
    cv2.imwrite("input_left.png", left_img)
    cv2.imwrite("input_right.png", right_img)

    # Directories for the directory-based sample (image_opencl); the left and
    # right frames must share the same filename.
    os.makedirs("left", exist_ok=True)
    os.makedirs("right", exist_ok=True)
    cv2.imwrite("left/pair.png", left_img)
    cv2.imwrite("right/pair.png", right_img)

    print("Wrote: input_left.png, input_right.png, left/pair.png, right/pair.png")


if __name__ == "__main__":
    main()

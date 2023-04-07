import cv2
from bounding_box import bounding_box as bb
import os

def show_and_save(image, path):
    cv2.imwrite(path, image)

def main():
    in_path = os.path.join("docs", "images", "winton.jpg")
    out_path = os.path.join("docs", "images", "winton_bb.png")
    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    bb.add(image, 281, 12, 744, 431, "Winton", "maroon")
    bb.add(image, 166, 149, 500, 297, "Trumpet", "yellow")
    show_and_save("Winton MARSALIS", image, out_path)

if __name__ == "__main__":
    main()
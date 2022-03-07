import os
import cv2
from glob import glob
from parser import parse_image

IMAGES_PATH = 'assets/images'


def parse_folder_images():
    image_paths = sorted(glob(os.path.join(IMAGES_PATH, '*.png')))
    parsed_path = 'parsed.csv'

    print(f'Parsing {len(image_paths)} images to {parsed_path}')
    with open(parsed_path, 'w') as fp:
        fp.write('file, symbol, c1, c2, p1, p2\n')
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            parsed = parse_image(image)

            # Write parsed file
            for line in parsed:
                fp.write(filename + ',' + ','.join(map(str, line)) + '\n')
        print('End')


if __name__ == '__main__':
    parse_folder_images()

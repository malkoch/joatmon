import numpy as np
from PIL import Image

path = r'C:\Users\malkoch\Downloads\Biqu_H2_Mount___Cover_-_Ender_3_CR10_4740250\images\Base_Cover_RevA01.png'

gscale = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
# gscale = '@%#*+=-:. '

def get_average_l(image):
    im = np.array(image)
    w, h = im.shape
    return np.average(im.reshape(w * h))


def covert_image_to_ascii(file_name, cols, scale):
    global gscale

    image = Image.open(file_name).convert('L')

    width, height = image.size[0], image.size[1]
    print("input image dims: %d x %d" % (width, height))

    w = width / cols
    h = w / scale

    rows = int(height / h)

    print("cols: %d, rows: %d" % (cols, rows))
    print("tile dims: %d x %d" % (w, h))

    if cols > width or rows > height:
        print("Image too small for specified cols!")
        exit(0)

    aimg = []
    for j in range(rows):
        y1 = int(j * h)
        y2 = int((j + 1) * h)

        if j == rows - 1:
            y2 = height

        aimg.append("")

        for i in range(cols):

            x1 = int(i * w)
            x2 = int((i + 1) * w)

            if i == cols - 1:
                x2 = width

            img = image.crop((x1, y1, x2, y2))
            avg = int(get_average_l(img))

            gsval = gscale[int((avg * (len(gscale) - 1)) / 255)]
            aimg[j] += gsval

    return aimg


# main() function
def main():
    scale = 0.43
    cols = 80

    aimg = covert_image_to_ascii(path, cols, scale)

    for row in aimg:
        print(row)


# call main
if __name__ == '__main__':
    main()

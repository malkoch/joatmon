import cv2

from joatmon.bot.image import (
    draw_box,
    grab_region_np,
    ocr_words,
    resize_image
)


possible_texts = {'shop', 'missions', 'quests', 'challenges', 'clan', 'index', 'champions', 'battle'}

while True:
    image = grab_region_np(0, 0, 2294, 1490)
    texts = ocr_words(image)
    for text in texts:
        if text['text'].lower() not in possible_texts:
            continue
        image = draw_box(image, text['x'], text['y'], text['w'], text['h'])

    image = resize_image(image, 1366, 720)
    cv2.imshow("image", image)
    cv2.moveWindow("image", -1600, 0)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC key to break
        break

cv2.destroyAllWindows()

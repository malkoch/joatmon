import platform


if platform.system() == "Windows":
    from joatmon.bot.image._windows import grab_region_np
elif platform.system() == "Linux":
    from joatmon.bot.image._linux import grab_region_np
elif platform.system() == "Darwin":
    from joatmon.bot.image._mac import grab_region_np
else:
    raise RuntimeError("Unsupported platform")


def resize_image(image, width, height):
    import cv2

    return cv2.resize(image, (width, height))


def ocr_words(image, lang: str = "eng", psm: int = 6):
    import pytesseract
    import cv2

    """
    Returns a list of dicts: {text, conf, x, y, w, h}
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cfg = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(bw, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = data["conf"][i]
        if txt and conf >= 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append({"text": txt, "conf": conf, "x": x, "y": y, "w": w, "h": h})
    return out


def draw_box(image, x, y, w, h, color=(0, 255, 0), thickness=2):
    import cv2

    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

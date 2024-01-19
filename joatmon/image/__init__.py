import sys

import cv2
import numpy
import numpy as np

from joatmon.system.hid.screen import grab

__all__ = ['watermark', 'crop', 'show', 'save', 'mask', 'match_template']


def watermark(image=None, mark=None, region=None, alpha=0.2) -> numpy.ndarray:
    """
    Adds a watermark to an image.

    # Arguments
        image (numpy.ndarray): The image to watermark. If None, the current screen is grabbed.
        mark (numpy.ndarray): The watermark to add.
        region (tuple): The region to apply the watermark to. Should be a tuple of (left, top, right, bottom).
        alpha (float): The transparency of the watermark. Default is 0.2.

    # Returns
        numpy.ndarray: The watermarked image.
    """
    if mark is None:
        raise Exception

    if image is None:
        image = grab()

    if region:
        left, top, right, bottom = region
    else:
        raise Exception

    image_copy = image.copy()
    image_copy = numpy.dstack([image_copy, numpy.ones(image_copy.shape[:2], dtype='uint8') * 255])

    overlay = numpy.zeros(image_copy.shape[:2] + (4,), dtype='uint8')
    overlay[top:bottom, left:right] = mark

    output = image_copy.copy()
    cv2.addWeighted(overlay, alpha, output, 1.0, 0.0, output)
    return cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)


def crop(image=None, region=None) -> numpy.ndarray:
    """
    Crops an image to a specified region.

    # Arguments
        image (numpy.ndarray): The image to crop. If None, the current screen is grabbed.
        region (tuple): The region to crop to. Should be a tuple of (left, top, right, bottom).

    # Returns
        numpy.ndarray: The cropped image.
    """
    if image is None:
        image = grab()

    if region:
        left, top, right, bottom = region
    else:
        left, top, (right, bottom) = 0, 0, *image.size

    return image[top:bottom, left:right]


def show(image, title) -> None:
    """
    Displays an image in a new window.

    # Arguments
        image (numpy.ndarray): The image to display.
        title (str): The title of the window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(1)


def save(image, path) -> None:
    """
    Saves an image to a file.

    # Arguments
        image (numpy.ndarray): The image to save.
        path (str): The path to save the image to.
    """
    cv2.imwrite(path, image)


def mask(image=None, r1=(0, 255), g1=(0, 255), b1=(0, 255), r2=None, g2=None, b2=None):
    """
    Applies a mask to an image.

    # Arguments
        image (numpy.ndarray): The image to mask. If None, the current screen is grabbed.
        r1, g1, b1 (tuple): The lower and upper bounds for the first color range.
        r2, g2, b2 (tuple): The lower and upper bounds for the second color range.

    # Returns
        numpy.ndarray: The masked image.
    """
    if image is None:
        image = grab()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower1 = np.array([r1[0], g1[0], b1[0]])
    upper1 = np.array([r1[1], g1[1], b1[1]])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    masked = mask1

    if r2 is not None and g2 is not None and b2 is not None:
        lower2 = np.array([r2[0], g2[0], b2[0]])
        upper2 = np.array([r2[1], g2[1], b2[1]])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        masked += mask2

    copy = image.copy()
    copy[np.where(masked == 0)] = 0
    return copy


def match_template(image=None, template=None, threshold=0.8) -> list:
    """
    Matches a template to an image.

    # Arguments
        image (numpy.ndarray): The image to match the template to. If None, the current screen is grabbed.
        template (numpy.ndarray): The template to match.
        threshold (float): The matching threshold. Default is 0.8.

    # Returns
        list: A list of points where the template matches the image.
    """
    if template is None:
        raise Exception

    if image is None:
        image = grab()

    res = cv2.matchTemplate(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED
    )
    loc = numpy.where(res >= threshold)
    return list(zip(*loc[::-1]))

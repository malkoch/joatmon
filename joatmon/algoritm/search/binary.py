def binary_search(iterable, value, item_getter):
    low = 0
    high = len(iterable) - 1

    while low <= high:
        mid = low + (high - low) // 2

        if item_getter(iterable[mid]) == value:
            return iterable[mid]
        elif item_getter(iterable[mid]) < 0:
            low = mid + 1
        else:
            high = mid - 1


def binary_search_helper(iterable, value, item_getter, low, high):
    if high >= low:
        mid = low + (high - low) // 2

        if item_getter(iterable[mid]) == value:
            return iterable[mid]
        elif item_getter(iterable[mid]) > 0:
            return binary_search_helper(iterable, value, item_getter, low, mid - 1)
        else:
            return binary_search_helper(iterable, value, item_getter, mid + 1, high)


def binary_search_r(iterable, value, item_getter):
    return binary_search_helper(iterable, value, item_getter, 0, len(iterable) - 1)

def linear_search(iterable, value, item_getter):
    for item in iterable:
        if item_getter(item) == value:
            return item

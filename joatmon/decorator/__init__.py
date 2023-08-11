def register(*decorators):
    def register_wrapper(func):
        for deco in decorators[::-1]:
            func = deco(func)
        func._decorators = decorators
        return func

    return register_wrapper

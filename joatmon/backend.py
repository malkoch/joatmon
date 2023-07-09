arr_backend = None
nn_backend = None


def set_arr_backend(b):
    global arr_backend
    arr_backend = b
    return arr_backend


def set_nn_backend(b):
    global nn_backend
    nn_backend = b
    return nn_backend


def get_arr_backend():
    return arr_backend


def get_nn_backend():
    return nn_backend

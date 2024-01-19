from joatmon.nn import functional as f

__all__ = ['MaxPool']


class MaxPool:
    """
    Applies max pooling to the input.

    # Arguments
        kernel_size (int or tuple): Size of the window to take max over.
        stride (int or tuple): Stride of the window. Default value is `kernel_size`.
        padding (int or tuple): Implicit zero padding to be added on both sides.

    # Attributes
        _kernel_size (int or tuple): Size of the window to take max over.
        _stride (int or tuple): Stride of the window.
        _padding (int or tuple): Implicit zero padding to be added on both sides.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        self._kernel_size = kernel_size
        self._stride = stride if (stride is not None) else kernel_size
        self._padding = padding

    def forward(self, inp):
        """
        Applies max pooling to the input.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying max pooling.
        """
        return f.max_pool(
            inp=inp,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
        )

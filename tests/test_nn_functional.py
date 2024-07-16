def test_pad():
    from joatmon.nn.functional import pad, zeros

    zero = zeros((2, 2, 2, 2))
    padded = pad(zero, (1, 1, 1, 1))

    assert zero.shape == (2, 2, 2, 2)
    assert padded.shape == (2, 2, 4, 4)

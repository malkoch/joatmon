def test_pad():
    from joatmon.nn.functional import pad, zeros

    zero = zeros((2, 2, 2, 2))
    padded = pad(zero, (1, 1, 1, 1))

    assert zero.shape == (2, 2, 2, 2)
    assert padded.shape == (2, 2, 4, 4)


def test_concat():
    from joatmon.nn.functional import concat, zeros, ones

    zero = zeros((2, 2))
    one = ones((2, 2))
    concated_4x2 = concat([zero, one], axis=0)
    concated_2x4 = concat([zero, one], axis=1)

    assert concated_4x2.shape == (4, 2)
    assert concated_2x4.shape == (2, 4)

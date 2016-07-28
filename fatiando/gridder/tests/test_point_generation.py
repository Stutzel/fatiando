from __future__ import division, absolute_import, print_function
import numpy.testing as npt
import numpy as np
from pytest import raises

from ... import gridder


def test_regular():
    "Regular grid generation works in the correct order of points in the array"
    shape = (5, 3)
    x, y = gridder.regular((0, 10, 0, 5), shape)
    x_true = np.array([[0., 0., 0.],
                       [2.5, 2.5, 2.5],
                       [5., 5., 5.],
                       [7.5, 7.5, 7.5],
                       [10., 10., 10.]])
    npt.assert_allclose(x.reshape(shape), x_true)
    y_true = np.array([[0., 2.5, 5.],
                       [0., 2.5, 5.],
                       [0., 2.5, 5.],
                       [0., 2.5, 5.],
                       [0., 2.5, 5.]])
    npt.assert_allclose(y.reshape(shape), y_true)
    # Test that the z variable is returned correctly
    x, y, z = gridder.regular((0, 10, 0, 5), shape, z=-10)
    z_true = -10 + np.zeros(shape)
    npt.assert_allclose(z.reshape(shape), z_true)
    # Test a case with a single value in x
    shape = (1, 3)
    x, y = gridder.regular((0, 0, 0, 5), shape)
    x_true = np.array([[0., 0., 0.]])
    npt.assert_allclose(x.reshape(shape), x_true)
    y_true = np.array([[0., 2.5, 5.]])
    npt.assert_allclose(y.reshape(shape), y_true)


def test_regular_fails():
    "gridder.regular should fail for invalid input"
    # If the area parameter is specified in the wrong order
    with raises(AssertionError):
        x, y = gridder.regular((1, -1, 0, 10), (20, 12))
    with raises(AssertionError):
        x, y = gridder.regular((0, 10, 1, -1), (20, 12))


def test_scatter():
    "Scatter point generation returns sane values with simple inputs"
    # Can't test random points for equality. So I'll test that the values are
    # in the correct ranges.
    area = (-1287, 5433, 0.1234, 0.1567)
    xmin, xmax, ymin, ymax = area
    n = 10000000
    x, y = gridder.scatter(area, n=n, seed=0)
    assert x.size == n
    assert y.size == n
    npt.assert_almost_equal(x.min(), xmin, decimal=1)
    npt.assert_almost_equal(x.max(), xmax, decimal=1)
    npt.assert_almost_equal(y.min(), ymin, decimal=1)
    npt.assert_almost_equal(y.max(), ymax, decimal=1)
    npt.assert_almost_equal(x.mean(), (xmax + xmin)/2, decimal=1)
    npt.assert_almost_equal(y.mean(), (ymax + ymin)/2, decimal=1)
    # Test that the z array is correct
    n = 1000
    x, y, z = gridder.scatter(area, n=n, z=-150, seed=0)
    assert z.size == n
    npt.assert_allclose(z, -150 + np.zeros(n))


def test_scatter_fails():
    "gridder.scatter should fail for invalid input"
    # If the area parameter is specified in the wrong order
    with raises(AssertionError):
        x, y = gridder.scatter((1, -1, 0, 10), 20, seed=1)
    with raises(AssertionError):
        x, y = gridder.scatter((0, 10, 1, -1), 20, seed=2)

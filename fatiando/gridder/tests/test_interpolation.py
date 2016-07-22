from __future__ import division, absolute_import, print_function
import numpy.testing as npt
import numpy as np
from pytest import raises

from ... import gridder


def test_exatrapolate_nans():
    "extrapolate_nans recovers values that were removed from a simple grid"
    x, y, z = gridder.regular((0, 20, -10, 2), (10, 10), z=500)
    z_missing = np.copy(z)
    z_missing[[0, 15, 67, 32, 12]] = np.nan
    assert np.any(z_missing != z)
    gridder.extrapolate_nans(x, y, z_missing)
    npt.assert_allclose(z_missing, z)
    # Test using a masked array as well
    mask = np.zeros_like(z)
    mask[[35, 43, 1, 46, 83, 99, 14, 78]] = 1
    z_missing = np.ma.masked_array(z, mask=mask)
    assert np.any(z_missing.mask)
    gridder.extrapolate_nans(x, y, z_missing)
    assert np.all(~z_missing.mask)
    npt.assert_allclose(z_missing, z)

"""
Create and operate on grids and profiles.

**Grid generation**

* :func:`~fatiando.gridder.regular`
* :func:`~fatiando.gridder.scatter`

**Grid operations**

* :func:`~fatiando.gridder.cut`
* :func:`~fatiando.gridder.profile`

**Interpolation**

* :func:`~fatiando.gridder.interp`
* :func:`~fatiando.gridder.interp_at`
* :func:`~fatiando.gridder.extrapolate_nans`

**Padding**

* :func:`~fatiando.gridder.pad_array`
* :func:`~fatiando.gridder.unpad_array`
* :func:`~fatiando.gridder.pad_coords`

**Input/Output**

* :func:`~fatiando.gridder.load_surfer`: Read a Surfer grid file and return
  three 1d numpy arrays and the grid shape

**Misc**

* :func:`~fatiando.gridder.spacing`

----

"""
from .slicing import inside, cut
from .interpolation import interp, interp_at, extrapolate_nans, profile
from .padding import pad_array, unpad_array, pad_coords
from .point_generation import regular, scatter
from .utils import spacing


# This function has no home for now but it will be moved to the datasets
# package later on.
import numpy as np


def load_surfer(fname, fmt='ascii'):
    """
    Read a Surfer grid file and return three 1d numpy arrays and the grid shape

    Surfer is a contouring, gridding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * fmt : str
        File type, can be 'ascii' or 'binary'

    Returns:

    * x : 1d-array
        Value of the North-South coordinate of each grid point.
    * y : 1d-array
        Value of the East-West coordinate of each grid point.
    * data : 1d-array
        Values of the field in each grid point. Field can be for example
        topography, gravity anomaly etc
    * shape : tuple = (nx, ny)
        The number of points in the x and y grid dimensions, respectively

    """
    assert fmt in ['ascii', 'binary'], "Invalid grid format '%s'. Should be \
        'ascii' or 'binary'." % (fmt)
    if fmt == 'ascii':
        # Surfer ASCII grid structure
        # DSAA            Surfer ASCII GRD ID
        # nCols nRows     number of columns and rows
        # xMin xMax       X min max
        # yMin yMax       Y min max
        # zMin zMax       Z min max
        # z11 z21 z31 ... List of Z values
        with open(fname) as ftext:
            # DSAA is a Surfer ASCII GRD ID
            id = ftext.readline()
            # Read the number of columns (ny) and rows (nx)
            ny, nx = [int(s) for s in ftext.readline().split()]
            shape = (nx, ny)
            # Read the min/max value of columns/longitude (y direction)
            ymin, ymax = [float(s) for s in ftext.readline().split()]
            # Read the min/max value of rows/latitude (x direction)
            xmin, xmax = [float(s) for s in ftext.readline().split()]
            area = (xmin, xmax, ymin, ymax)
            # Read the min/max value of grid values
            datamin, datamax = [float(s) for s in ftext.readline().split()]
            data = np.fromiter((float(i) for line in ftext for i in
                                   line.split()), dtype='f')
            data = np.ma.masked_greater_equal(data, 1.70141e+38)
            assert np.allclose(datamin, data.min()) \
                and np.allclose(datamax, data.max()), \
                "Min and max values of grid don't match ones read from file." \
                + "Read: ({}, {})  Actual: ({}, {})".format(
                    datamin, datamax, data.min(), data.max())
        # Create x and y coordinate numpy arrays
        x, y = regular(area, shape)
    if fmt == 'binary':
        raise NotImplementedError(
            "Binary file support is not implemented yet.")
    return x, y, data, shape

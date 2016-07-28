"""
Gridding: Load a Surfer ASCII grid file
"""
from fatiando import datasets
import matplotlib.pyplot as plt

# Fetching Bouguer anomaly model data (Surfer ASCII grid file)"
# Will download the archive and save it with the default name
archive = datasets.fetch_bouguer_alps_egm()

# Load the GRD file and convert in three numpy-arrays (x, y, bouguer)
x, y, bouguer, shape = datasets.load_surfer(archive, fmt='ascii')

plt.figure()
plt.title("Data loaded from a Surfer ASCII grid file")
plt.contourf(y.reshape(shape)/1000, x.reshape(shape)/1000,
             bouguer.reshape(shape), 15, cmap='RdBu_r')
plt.colorbar().set_label('mGal')
plt.xlabel('y points to East (km)')
plt.ylabel('x points to North (km)')
plt.show()

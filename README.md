A tool for correlating peaks in Xray Absorption Spectra to physical attributes in the molecule in order to express peak positions as functions of physical attributes like distances and angles between atoms within the molecule.

To run the code, clone the repository, and run python main.py "excited atom" snapshots folder xas folder radius "periodicity" lattice-a lattice-b lattice-c alpha beta gamma
radius -- float -- represents the neighborhood around an excited atom for which we want to calculate distance and angle relationships and then correlate them to that atom's intensity spectra
periodicity -- true or false -- whether the system is periodic. If "true", then the following arguments are required
lattice-a, lattice-b, lattice-c -- float -- the three lattice vectors that define the unit cell
alpha, beta, gamme -- float -- the three angles that define the unit cell

To run an example (this will run CO2 through the tool):
python main.py "C" snapshots xas 7 "false"

How the tool works:


DISCLAIMER:  
NOTE: THIS CODE BELONGS TO THE LAWRENCE BERKELEY NATIONAL LABORATORY. THIS CODE BEING ON MY GITHUB IS NOT TO BE INTERPRETED AS ME CLAIMING OWNERSHIP. I AM ONLY HOSTING THE CODE.  
  
General description:  
A tool for correlating peaks in Xray Absorption Spectra to physical attributes in the molecule in order to express peak positions as functions of physical attributes like distances and angles between atoms within the molecule.  
  
To run the code, clone the repository, and run:  
  
	python main.py "excited atom" snapshots_folder xas_folder radius "periodicity" lattice-a lattice-b lattice-c alpha beta gamma  
  
Parameters:  
  
	radius (float)  
		Represents the neighborhood around an excited atom for which we want to calculate distance  
		and angle relationships and then correlate them to that atom's intensity spectra.  
	periodicity (true of false)  
		Whether the system is periodic. If "true", then the following arguments are required.  
	lattice-a, lattice-b, lattice-c (3 floats)  
		The three lattice vectors that define the unit cell.  
	alpha, beta, gamma (3 floats)   
		The three angles that define the unit cell.  
  
To run an example (this will run CO2 through the tool):  
  
    python main.py "C" snapshots xas 7 "false"
  
How the tool works:  
1. Pick a snapshot.  
2. Parse xyz coordinates for every atom and all xas data for every excited atom in this snapshot.  
3. Pick an excited atom in this snapshot.  
4. Generate all distance and angle relationships for any atoms within the specified radius of this excited atom.  
5. Associate this list of relationships with the energy vs. intensity spectra for that excited atom in that snapshot.  
6. Repeat for all excited atoms in this snapshot.  
7. Repeat for all snapshots.  
8. Now, we have a pairing between relationships surrounding an excited atom and its spectra.  
9. Extract the peaks from each spectra by fitting a sum of Gaussians to each spectra.  
10. Now, we have a pairing between relationships surrounding an excited atom and one of its peaks. There is a separate pairing for each peak that is in a particular atom's spectra. Note the degeneracy/repitition in the input.  
11. Roughly cluster this peak data to separate different peaks from each other.  
12. Check the linearity of each cluster to see whether it contains more than one peak.  
13. If the cluster has low linearity, there could be multiple lines intersecting, so use Simulated Annealing to find the optimal labeling of these points (points that correspond to the same peak should be put together).  
14. Now that every peak is labelled, run linear regression to express each peak's movement throughout snapshots as a linear combination of the relationships surrounding the excited atoms.  

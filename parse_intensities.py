import re
import numpy as np
import os

XAS_FOLDER_NAME = "xas"

def parse_intensities(atomName, atomNum, xyz_name):
        """
        Takes the name of an xyz snapshot, an atom name (label), and which line the atom is found on, and then constructs the file name of the corresponding xas file. It then takes the 1000 energy intensity key value pairs in that xas file and converts them into a dictionary, which exists inside of a 2D numpy array. 
        Invariant: XYZ file must end in .xyz. Otherwise, xyzWithSnapNum = xyz_name[0:len(xyz_name)-4] will fail. It depends on the last four characters to be .xyz
        """
        xyzWithSnapNum = xyz_name[0:len(xyz_name)-4]
        xyzWithoutSnapNum = re.sub(r'(.*)_[0-9]+[.]xyz', r'\1', xyz_name)
        xas_filename = XAS_FOLDER_NAME + '/' + xyzWithSnapNum + '-' + xyzWithoutSnapNum + '.' + atomName + str(atomNum) + '-XCH.xas.5.xas'
        
        intensities = np.array([[{}]])
        if os.path.isfile(xas_filename) == True:
                raw_xas = np.loadtxt(xas_filename, usecols=(0, 1))
		if type(raw_xas[0]) != type(np.array([[]])):
			intensities[0][0][raw_xas[0]] = raw_xas[1]
		else:
	                for row in raw_xas:
                     		intensities[0][0][row[0]] = row[1]
        else:
                intensities[0][0] = -1

        return intensities

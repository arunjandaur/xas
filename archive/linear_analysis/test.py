import numpy as np
import matplotlib.pyplot as plt
import mdp

x = np.random.normal(2,0.5,1000)
x2 = np.random.normal(1,.25,1000)
x3 = np.random.normal(5,1,1000)

k_means = mpd.nodes.KMeansClassifier(3)
k_means.train(self, x, x2, x3)

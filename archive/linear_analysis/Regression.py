import numpy as np
import matplotlib.pyplot as plt
import mdp

if __name__ == "__main__":
    print "Running Main"
    t = np.array([np.linspace(0,5,100)]).T 
    x = np.concatenate((t**2, t), axis=1) + np.random.normal(0,5,(100,2))
    y = (x.dot(np.array([[-5],[65]]))) + np.random.normal(0,2,(100,1))
    reg = mdp.nodes.LinearRegressionNode()
    reg.train(x,y)
    reg.execute(x)
    
    plt.figure(1)
    plt.plot(t,y,"ro")

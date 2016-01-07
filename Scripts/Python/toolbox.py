import math
import numpy as np


"""
Start of ams-stuff
"""

def calcAMS(s,b):    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        ams = math.sqrt(radicand)
        return ams

def calcWeightSums(weights,predictions,labels):
    s = 0
    b = 0
    for j in list(range(0,len(predictions))):
        prediction = predictions[j]
        label = labels[j]
        weight = weights[j]
        if prediction > 0.:
            if label > 0.:
                s += weight
            else:
                b += weight
    return s,b

def calcMaxAMS(weights,labels):
    s,b = calcWeightSums(weights,labels,labels)
    ams = calcAMS(s,b)
    print("Found", int(labels.cumsum()[-1]), "signals.")
    print("Weightsums signal:", s, "| background:", b)
    print("Maximum AMS possible with this Data:", ams)
    return ams

"""
Start of data-generation
"""
def generateFeature(label, mu_s, mu_b, sigma_s=5, sigma_b=5):
    if label is 1:
        mu = mu_s
        sigma = sigma_s
    else:
        mu = mu_b
        sigma = sigma_b
    return np.random.normal(mu,sigma)

"""
createToyData(int n, int dim, float s_prob):
create good seperable toydata for testing-purposes
"""
def createToyData(n = 100,dim = 3,s_prob = 0.05):
    data= np.zeros(shape = (n,dim),dtype=float)
    if dim < 3:
        print("Operation canceled.",
              "Data should have at least one",
              "additional dimension besides weights and labels.",
              "(dim >=3)")
        return None
    data[:,0] = np.random.rand(n) #weights
    for i in range(0,n):
        if data[i,0] <= s_prob: # label-determination
            label = 1
        else:
            label = 0
        data[i,1] = label
        for j in range(2,dim):
            data[i,j]=generateFeature(label,mu_s=(j-1)*5,mu_b=(j-1)*20)
    return data

"""
misc tools
"""
def splitList(xList,n):
    aList = xList[:n]
    bList = xList[n:]
    return aList,bList
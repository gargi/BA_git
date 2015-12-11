"""
actual Data is assumed to be Poisson-distributed => Toy data should be Poisson-distributed ??
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def generateToyDict(n_events=10000,m_features=1,prob_s=0.005):
	toyDict = {}
	
	for i in range(0,n_events):
		row = list([])	
		i+=100000
		row.append(i)
		#generate Label
		row.append(random.random())
		if float(row[1]) <= prob_s:
			row.append("s")
		else:
			row.append("b")
		label = row[2]
		#generate Feature
		row.append(generateFeature(label,40,50))
		
		toyDict[i] = row
	return toyDict
			




def generateFeature(label, mu_s, mu_b, sigma_s=15, sigma_b=15):
	if label is "s":
		return mu_s + sigma_s * np.random.rand()
	else:
		return mu_b + sigma_b * np.random.rand()




if __name__ == "__main__":
	toyDict = generateToyDict()
	#print(toyDict)
	for event in toyDict:
		if toyDict[event][2] is "s":
			print(event)






# plt.figure(1)
# plt.subplot(221)
# count, bins, ignored = plt.hist(s, 20, normed=True)

# plt.subplot(222)
# count, bins, ignored = plt.hist(b, 20, normed=True)
# plt.show()


# plt.show()
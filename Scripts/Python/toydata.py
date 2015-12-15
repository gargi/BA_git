"""
actual Data is assumed to be Poisson-distributed => Toy data should be Poisson-distributed ??
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def generateToyDict(n_events=10000,m_features=1,prob_s=0.05):
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
		row.append(generateFeature(label,10,15))
		#row.append(generateFeature(label,100,100))
		
		toyDict[i] = row
	return toyDict  
			
def generateFeature(label, mu_s, mu_b, sigma_s=5, sigma_b=5):
	if label is "s":
		return np.random.normal(mu_s,sigma_s)
	else:
		return np.random.normal(mu_b,sigma_b)

def dict2List(dict):
	rows = toyDict.values()
	cols = transposeList(rows)
	return cols


def transposeList(aList):
	newList = zip(*aList)  # transpose
	newList = list(map(list,newList)) # convert generators to things that can be indexed
	return newList

def splitDict(dictionary,trainLen):
	rows = dictionary.values()
	rows = list(map(list,rows))

	trainingData = rows[:trainLen]
	testData = rows[trainLen:]
	return trainingData, testData

def scatterData(data,x_index=3,y_index=4):
	#events = cols[0]
	#probabilities = cols[1]
	labels = data[2]
	for i in range(0,len(labels)):
		if labels[i] is "s":
			labels[i] = "b"
		elif labels[i] is "b":
			labels[i] = "r"
	x = data[x_index]
	y = data[y_index]
	plt.scatter(x, y, s=1, edgecolor="", c = labels)
	plt.show()


if __name__ == "__main__":
	toyDict = generateToyDict()
	toyList = transposeList(toyDict.values())
	trainingRows,testRows = splitDict(toyDict,1000)
	
	scatterData(toyList)
import toydata as td
import higgsml_opendata_kaggle3 as higgs
import math
import numpy as np

def getToyData():
	toyDict = td.generateToyDict(n_events = 10000, prob_s=0.025)
	return toyDict

def splitToyData(toyDict):
	trainingData, testData = td.splitDict(toyDict,1000)
	return trainingData, testData

def calcWeightSums(testData):
	s = 0
	b = 0
	for event in range(0,len(testData)):
		weight = testData[event][1]
		label = testData[event][2]
		if label is "s":
			s += weight
		elif label is "b":
			continue
		#	b += weight
		else:
			print("ERROR! Wrong label '" + label + "'")
			break

	return s,b

def calcAMS(s, b):
	"""
	Approximate Median Significance defined as:
		AMS = sqrt(2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s})        
		where b_r = 10, b = background, s = signal, log is natural logarithm
	"""

	br = 10.0
	radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
	if radicand < 0:
		print('radicand is negative. Exiting')
		exit()
	else:
		return math.sqrt(radicand)

def calcMaxAMS(testData):
	s,b = calcWeightSums(testData)
	ams = calcAMS(s,b)
	print("Maximum AMS possible with this Data:", ams)
	return ams


def testLogisticRegression(trainData,testData):
	from sklearn import linear_model 
	logReg = linear_model.LogisticRegression(C=1e5)

	x_train = trainData[-2]
	#x_train = trainData[-1]
	y_train = []
	eventList = testData[0]
	x_test = testData[-2]
	y_test = []

	
	for i in range(0,(np.shape(trainData)[1])):
		if trainData[2][i] is "s":
			y_train.append(1)
		else:
			y_train.append(0)

	for i in range(0,len(testData)):
		
		if testData[2][i] is "s":
			y_test.append(1)
		else:
			y_test.append(0)

	#expData = logReg.calcSolution(x_train,y_train,eventList,x_test,y_test)
	print(x_train[150])
	print(np.shape(y_train))
	print(y_train[150])
	#print(x_train[150][1])
	print(trainData[0:2][0])

	return "0"


if __name__ == "__main__":
	toyDict = getToyData()
	a,b = splitToyData(toyDict)
	a = td.transposeList(a)
	b = td.transposeList(b)
	#print(a[0][2])
	print(testLogisticRegression(a,b))
	#maxAMS = calcMaxAMS(b)
	#b = td.transposeList(b)
	#print("Signal-Quantity in Test Data:" + str(b[2].count("s")))

	#td.scatterData(b)
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import csv
#import random
import os

folderPath = "F:\BA\\"
dataPath = folderPath + "Data\\"
csvFile = dataPath + "Atlas-higgs-challenge-2014-v2.csv"

#define wanted features in this method
def getFeatureList():
	featureList = []

	featureList.append("DER_mass_MMC")
	featureList.append("PRI_tau_phi")
	#featureList.append("DER_lep_eta_centrality")

	return featureList

	##create Dictionary out of csv-File
def createCsvDictionary(csvF):
	csvDict = {}
	print("\n\n\n")
	print("=== Reading csv file ",csvF, "===\n\n")
	with open(csvF, 'r') as f:
		csvR = csv.reader(f)
		header=next(csvR) # header
		iid=header.index("EventId")
		for row in csvR:
			csvDict[row[iid]] = row
	return csvDict,header


	##create csv-Dictionary and cut unwanted Features => dataDict
def getDataDictionary(kaggleSet = "all"):
	csvDict,header = createCsvDictionary(csvFile)

	indexKaggleSet = header.index("KaggleSet")
	indexLabel=header.index("Label")

	featureList = getFeatureList()
	featureIndex = []

	dataHeader = []
	dataHeader.append("EventId")
	

	for feature in featureList:
		dataHeader.append(feature)
		featureIndex.append(header.index(feature))

	dataDict = {}


	for event in csvDict:
		if csvDict[event][indexKaggleSet] is kaggleSet or kaggleSet is "all":
			helplist = []
			i=0
			for featureName in dataHeader:
				featureID = header.index(featureName)
				featureValue = csvDict[event][featureID]
				# dataDict[eventID][i] = csvDict[eventID][featureID]
				helplist.append(featureValue)
				i = i + 1

			label = csvDict[event][indexLabel]
			if label == "s":
				labelColor = "b" ##signal = blue
				helplist.append(1)
			elif label == "b":
				labelColor = "r" ##background = red
				helplist.append(0)
			else:
				print("ERROR in Labels!")

			helplist.append(labelColor)
				##cutting Errors
			if "-999.0" in helplist:
				continue
			dataDict[event] = helplist
	print("Dictionary-Length:", len(dataDict))
	return dataDict




def getData():
	dataDict = getDataDictionary(kaggleSet = "t")
	print(dataDict["297317"])

	x = np.ndarray([len(dataDict),len(getFeatureList())])
	y = np.ndarray([len(dataDict),1])
	colors = []

	i=0
	for event in dataDict:
		colors.append(dataDict[event][-1])
		y[i]=dataDict[event][-2]
		#print("Y = ",y[i])
		#print("Color = ", colors[i])
		x[i][:]=dataDict[event][1:-2]
		#print("X = ",x[i])
		i = i + 1

	print(x)
	return x,y,colors



def plot(x, y, colors, regr):
	x_axis1 = x[:,0]
	x_axis2 = x[:,1]

	plt.scatter(x_axis1, x_axis2,s=2, edgecolor="", c=colors, alpha=0.1)
	plt.plot(x, regr.predict(x), color='blue',linewidth=1)

	plt.xticks(())
	plt.yticks(())

	plt.show()


def test_genX(size):
	x = np.random.random(size)
	return x

def test_genY(size):
	y = np.random.randint(0,2,(size))
	return y

def test_genData(size):
	x = test_genX(size)
	y = test_genY(size)
	return x,y

def test_linreg():
	x,y,colors = getData()

	print("Shape x:",x.shape)
	print("Shape y:",y.shape)

	x_train = x[:-20000]
	x_test = x[-20000:]

	y_train = y[:-20000]
	y_test = y[-20000:]

	colors_test = colors[-20000:]

	print("Length x_train:", len(x_train))
	print("Shape x_train:",x_train.shape)
	
	print("Length y_train:", len(y_train))
	print("Shape y_train:",y_train.shape)

	print("Length x_test:", len(x_test))
	print("Shape x_test:",x_test.shape)
	print("Length y_test:", len(y_test))
	print("Shape y_test:",y_test.shape)
	y_test2 = np.empty(x_test.shape)
	(rows,columns) = y_test2.shape
	print("Shape y_test2:",y_test2.shape)
	for i in range(rows):
		for j in range(columns):
			y_test2[i][j] = y_test[i]

	print("Shape y_test2:",y_test2.shape)
	print("Length colors_test:", len(colors_test))


	# print(x_test)
	# print(y_test)

	regr = linear_model.LinearRegression()

	regr.fit(x_train,y_train)

	print('Coefficients: \n', regr.coef_)
	# The mean square error
	print("Residual sum of squares: %.2f"
		% np.mean((regr.predict(x_test) - y_test) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(x_test, y_test))

	# x_test2 = x_test[:,0]
	# print(x_test2)
	# y_test2 = y_test[:,0]
	# print(y_test2)
	

	plot( x_test, y_test2, colors_test, regr)



if __name__ == "__main__":

	#read csv
	#csvDict,header = createCsvDictionary(csvFile)
	#getDataDictionary(None,None)
	test_linreg()
	print("all done(?)")
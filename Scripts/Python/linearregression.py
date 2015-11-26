from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import KaggleData as KD

#define wanted features in this method
def getFeatureList():

	featureList = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 
	'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
	'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
	'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt',
	'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
	'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
	'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']



	# featureList = []
	# featureList.append("EventId")


	# featureList.append("DER_mass_MMC")
	# featureList.append("DER_mass_vis")
	
	# featureList.append("DER_pt_ratio_lep_tau")
	# featureList.append("DER_lep_eta_centrality")

	# featureList.append("PRI_tau_phi")



	featureList.append("Label")
	return featureList

def getFeatureIndexList(featureList,header):
	indexFeatures = []
	for feature in featureList:
		indexFeatures.append(header.index(feature))
	return indexFeatures


def getIgnoreFeatureList():
	ignoreFeatureList = []
	ignoreFeatureList.append("EventId")
	ignoreFeatureList.append("Label")
	return ignoreFeatureList


def getData(kaggleSets = ["t","b","v","u"],hasErrorValues=False):
	featList = getFeatureList()
	ignFeatList = getIgnoreFeatureList()
	dataDict, dataHeader = KD.getCustomKaggleData(kaggleSets,featList,hasErrorValues)

	indexBinLabel = dataHeader.index("BinaryLabel")
	indexLabelColor = dataHeader.index("LabelColor")

	x = np.ndarray([len(dataDict),(len(featList)-len(ignFeatList))])
	y = np.ndarray([len(dataDict)])
	eventList = np.ndarray([len(dataDict)])
	colors = []

	i=0
	for event in dataDict:
		colors.append(dataDict[event][indexLabelColor])
		eventList[i]=event
		y[i]=dataDict[event][indexBinLabel]
		x[i][:]=dataDict[event][1:(indexBinLabel-1)]
		i = i + 1

	return [eventList,x,y,colors]


def plot(x, y, colors, regr):
	x_axis1 = x[:,0]
	x_axis2 = x[:,1]

	plt.scatter(x_axis2, x_axis1 ,s=2, edgecolor="", c=colors, alpha=0.5)
	plt.plot(x, regr.predict(x), color='blue',linewidth=1)

	plt.xticks(())
	plt.yticks(())

	plt.show()

def optimizeThresh(prediction,test):
	threshold = 1.0
	oldprecision = 0
	deltaPrecision = 1
	while deltaPrecision > 0.001:
		sumCorrect = 0
		i = 0
		for value in prediction:
			if value < threshold:
				binValue = 0
			else:
				binValue = 1
			if int(binValue) is int(test[i]):
				sumCorrect = sumCorrect + 1
			i = i + 1

		newprecision = sumCorrect / len(test)
		deltaPrecision = abs(newprecision - oldprecision)
		oldprecision = newprecision
		print("current Precision:", newprecision, "| deltaPrecision:", deltaPrecision, "| Threshold:", threshold)
		if deltaPrecision < 0.001:
			return threshold
		else:
			threshold = threshold - 0.1

def linreg(kaggleTrainingSet = ["t"], kaggleTestSet = ["b","v"]):
	x_training,y_training = getData(kaggleTrainingSet)[1:3]
	test_eventList,x_test,y_test = getData(kaggleTestSet,hasErrorValues=True)[0:3]

	regr = linear_model.LinearRegression()
	regr.fit(x_training,y_training)

	predTrue = regr.predict(x_test)

	i = 0
	sumCorrect = 0
	pred = []
	threshold = optimizeThresh(predTrue,y_test)

	for value in predTrue:
		if value < threshold:
			binValue = 0
		else:
			binValue = 1
		pred.append(binValue)
		if int(binValue) is int(y_test[i]):
			sumCorrect = sumCorrect + 1
		i = i + 1

	precisionCorrect = sumCorrect / len(y_test)

	print('Coefficients: \n', regr.coef_)
	# The mean square error
	print("Residual sum of squares: %.2f"
		% np.mean((predTrue - y_test) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(x_test, y_test))

	print("Precision:", precisionCorrect)
	return test_eventList,predTrue,threshold

def calcSolution():
	import operator

	
	eventList,predTrue,threshold = linreg()

	expData = np.ndarray([len(eventList),3])
	for i  in range(0,len(expData)):
		expData[i]= [eventList[i],0,predTrue[i]]

	expData = expData[np.argsort(expData[:,2])]
	for i in range(1,(len(eventList)+1)):
		expData[i-1][1] = i

	return expData,threshold

	
if __name__ == "__main__":
	calcSolution()
	print("all done(?)")
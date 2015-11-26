from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import KaggleData as KD

#define wanted features in this method
def getFeatureList():

	# featureList = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 
	# 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
	# 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
	# 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt',
	# 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
	# 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
	# 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']



	featureList = []
	featureList.append("EventId")


	# featureList.append("DER_mass_MMC")
	# featureList.append("DER_mass_vis")
	
	featureList.append("DER_pt_ratio_lep_tau")
	featureList.append("DER_lep_eta_centrality")

	featureList.append("PRI_tau_phi")
	featureList.append("PRI_tau_pt")
	featureList.append("PRI_lep_pt")



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


def plot(X,Y,logreg):
	print(X.shape)
	print(Y.shape)
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = .02
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), sparse = True)
	Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1, figsize=(4, 3))
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel("DER_mass_MMC")
	plt.ylabel("DER_mass_vis")

	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())

	plt.show()

def logreg():
	x_train,y_train = getData(["t"])[1:3]
	eventList,x_test,y_test = getData(["v","b"],hasErrorValues = True)[0:3]
	logreg = linear_model.LogisticRegression(C=1e5)

	logreg.fit(x_train,y_train)

	logreg.sparsify()

	predProb = logreg.predict_proba(x_test)
	#pred = logreg.predict(x_test)
	score = logreg.score(x_test,y_test)


	print("Score:", score)

	return eventList,predProb




def calcSolution():
	import operator
	
	eventList,predProb = logreg()

	expData = np.ndarray([len(eventList),3])
	for i  in range(0,len(expData)):
		expData[i]= [eventList[i],0,predProb[i][0]]

	expData = expData[np.argsort(expData[:,2])]
	for i in range(1,(len(eventList)+1)):
		expData[i-1][1] = i

	return expData

	
if __name__ == "__main__":
	# logreg()
	calcSolution()
	print("all done(?)")
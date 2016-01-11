import csv
import os
import numpy as np



scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
dataPath = (mainFolderPath + "/Data")
csvFile = dataPath + "/Atlas-higgs-challenge-2014-v2.csv"


def _checkCustomFeatureList(customFeatureList):
	originalFeatureList = getFeatureListAll()
	return set(customFeatureList).issubset(originalFeatureList)

	##create Dictionary out of csv-File
def createCsvDictionary():
	csvF = csvFile
	csvDict = {}
	#print("Reading csv file ",csvF)
	with open(csvF, 'r') as f:
		csvR = csv.reader(f)
		header=next(csvR) # header of Dictionary
		iid=header.index("EventId")
		for row in csvR:
			csvDict[row[iid]] = row
	return csvDict,header


"""
returns one feature of given set, making no further changes to it
"""
def getFeatureAsList(csvDict,header,featureName,kaggleSets,hasErrorValues=False):
	if type(csvDict) is not dict:
		print("ERROR! Attribute 1 is not of dict-type!")
		return None
	if featureName not in getFeatureListAll():
		print("ERROR! Given feature is not part of CERN-Data")
		return None
	indexKaggleSet = header.index("KaggleSet")
	indexFeature = header.index(featureName)

	#start extracting wanted Data from csvDict(ionary)
	featList = []
	for event in csvDict:
		if csvDict[event][indexKaggleSet] in kaggleSets:
			feature = csvDict[event][indexFeature]
			if feature is 's':
				feature = 1
			elif feature is 'b':
				feature = 0
			featList.append(feature)
	return featList

def getFeatureAsNpArray(csvDict,header,featureName,kaggleSets,hasErrorValues=False):
	featList = getFeatureAsList(csvDict,header,featureName,kaggleSets,hasErrorValues=False)
	n = len(featList)
	featArray = np.zeros(n)
	for i in range(0,n):
		featArray[i] = float(featList[i])
	return featArray


#original feature-list
def getFeatureListAll():
	featureList = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 
	'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
	'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
	'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt',
	'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
	'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
	'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt', 'Weight', 'Label', 'KaggleSet', 'KaggleWeight']
	return featureList

#original feature-list without
def getFeatureListOnlyData():
	featureList = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 
	'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
	'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
	'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt',
	'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
	'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
	'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
	return featureList

def getFeatureSets(featureName,header,test_data,train_data):
    index = header.index(featureName)
    trainFeature = train_data[:,index]
    testFeature = test_data[:,index]
    return trainFeature, testFeature

def getCustomDataSet(featureList,csvDict=None,header=None,kSet = "v"):
	if csvDict is None:
		csvDict,header = createCsvDictionary()

	labels = getFeatureAsNpArray(csvDict,header,"Label",kSet,hasErrorValues=True)
	kWeights = getFeatureAsNpArray(csvDict,header,"KaggleWeight",kSet,hasErrorValues=True)
	data = np.zeros(shape = (len(labels),len(featureList)),dtype=float)
	
	for	feature in featureList:
		i = featureList.index(feature)
		data[:,i] = getFeatureAsNpArray(csvDict,header,feature,kSet,hasErrorValues=True)
	new_header = featureList

	return new_header, data, kWeights, labels

def getOneDataSet(csvDict=None,header=None,kSet = "v"):
	if csvDict is None:
		csvDict,header = createCsvDictionary()
	featureList = getFeatureListOnlyData()

	return getCustomDataSet(featureList,csvDict,header,kSet)

def getWholeDataSet(csvDict=None,header=None,kSet = "v"):
	if csvDict is None:
		csvDict,header = createCsvDictionary()
	test_data, test_weights, test_labels = getOneDataSet(csvDict,header,kSet)[1:4]
	header, train_data, train_weights, train_labels = getOneDataSet(csvDict,header,"t")
	return header, test_data, test_weights, test_labels, train_data, train_weights, train_labels
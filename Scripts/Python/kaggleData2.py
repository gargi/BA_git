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

	##create array out of csv-File
def csvToArray(rows=818238,columns=35):
    csvF = csvFile
    csv_data = np.empty([rows,columns],"<U16")
    with open(csvF, 'r') as f:
        csvR = csv.reader(f)
        csv_header=next(csvR) # header of Dictionary
        samples = 0
        i = 0
        for row in csvR:
            csv_data[i]= row
            i+=1
    return csv_data,csv_header

def _extractFeature(featureName,data,data_header,dtype=float):
    featureIndex = data_header.index(featureName)
    featureData = np.empty([len(data),1],dtype)
    featureData = data[:,featureIndex].astype(dtype)
    return featureData

def _extractFeatures(featureList,data,data_header,dtype="<U16"):
	featuresData = np.empty([len(data),len(featureList)],dtype)
	for featureName in featureList:
	    featuresData[:,featureList.index(featureName)] = _extractFeature(featureName,data,data_header,dtype)
	return featuresData

def _extractKaggleDataset(data,data_header,kSets=["t"],dtype="<U16"):
    if kSets == ["t"]:
        setLength = 250000
    elif kSets == ["b"]:
        setLength = 100000
    elif kSets == ["v"]:
        setLength = 450000
    elif "b" in kSets and "v" in kSets:
        setLength = 550000
    elif "u" in kSets:
        setLength = 18238
    else:
        print("Error, invalid Kaggleset (kSet)")
        return None
    indexKaggleSet = data_header.index("KaggleSet")
    kaggleDataset = np.empty([setLength,len(data_header)],dtype)
    j = 0
    for i in range(0,len(data)):
        if data[i][indexKaggleSet] in kSets:
            kaggleDataset[j][:] = data[i][:]
            j+=1
    return kaggleDataset

def getSolutionKey(csvDict=None,header=None):
	if csvDict is None:
		csvDict,header = createCsvDictionary()
	events, kWeights, labels = getCustomDataSet(['EventId'],csvDict,header,["b","v"])
	events, kWeights, labels

def getOriginalKaggleSets(data,data_header):
    
    train_header = getFeatureListTraining()
    train_data = _extractKaggleDataset(data,data_header,kSets=["t"])
    train_data = _extractFeatures(train_header,train_data,data_header)
    
    test_header = getFeatureListTest()
    test_data = _extractKaggleDataset(data,data_header,kSets=["b","v"])
    test_data = _extractFeatures(test_header,test_data,data_header)
    
    return train_data,train_header,test_data,test_header

def getSolutionKey(data,data_header):
    sol_header = ["EventId","Label","KaggleSet","KaggleWeight"]
    sol_data = _extractKaggleDataset(data,data_header,kSets=["b","v"])
    sol_data = _extractFeatures(sol_header,sol_data,data_header)
    return sol_data,sol_header

def translateLabels(data,data_header=None):
	translated_data = data.copy()
	if len(data_header) > 1:
		indexLabels = data_header.index("Label")
		for i in range(0,len(data)):
			if data[i,indexLabels] == "s":
				translated_data[i,indexLabels] = 1
			else:
				translated_data[i,indexLabels] = 0
	else:
		for i in range(0,len(data)):
			if data[i] == "s":
				translated_data[i] = 1
			else:
				translated_data[i] = 0
	return translated_data

"""
frequently used feature-lists
"""
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

def getFeatureListTraining():
	featureList = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 
	'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
	'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
	'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt',
	'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
	'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
	'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt', 'KaggleWeight', 'Label']
	return featureList

def getFeatureListTest():
	featureList = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 
	'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
	'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
	'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt',
	'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num',
	'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
	'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
	return featureList

def getFeatureListNoErrors(sloppy = True):
	featureList = getFeatureListOnlyData()
	errorFeatures = ['DER_mass_MMC',
					 'DER_deltaeta_jet_jet',
 					 'DER_mass_jet_jet',
 					 'DER_prodeta_jet_jet',
 					 'DER_lep_eta_centrality',
 					 'PRI_jet_leading_pt',
 					 'PRI_jet_leading_eta',
 					 'PRI_jet_leading_phi',
 					 'PRI_jet_subleading_pt',
 					 'PRI_jet_subleading_eta',
 					 'PRI_jet_subleading_phi']
	if sloppy is True:
		errorFeatures.remove('DER_mass_MMC')
	for feature in errorFeatures:
		featureList.remove(feature)
	return featureList
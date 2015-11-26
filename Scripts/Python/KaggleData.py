import csv
import os


scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
dataPath = (mainFolderPath + "\\Data")
csvFile = dataPath + "\\Atlas-higgs-challenge-2014-v2.csv"


def _checkCustomFeatureList(customFeatureList):
	originalFeatureList = getFeatureListAll()
	return set(customFeatureList).issubset(originalFeatureList)

	##create Dictionary out of csv-File
def _createCsvDictionary():
	csvF = csvFile
	csvDict = {}
	print("Reading csv file ",csvF)
	with open(csvF, 'r') as f:
		csvR = csv.reader(f)
		header=next(csvR) # header of Dictionary
		iid=header.index("EventId")
		for row in csvR:
			csvDict[row[iid]] = row
	return csvDict,header

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

def getCustomKaggleData(kaggleSets,featureList,hasErrorValues=False):
	if featureList is None:
		featureList = getFeatureListAll()
	elif _checkCustomFeatureList(featureList) is False:
		print("Error! Custom Featurelist not a subset of original Featurelist")
		return None,None


	csvDict,header = _createCsvDictionary()

	indexKaggleSet = header.index("KaggleSet")

	dataHeader = []
	dataHeader = list(featureList)

	if "Label" in featureList:
		indexLabel = header.index("Label")
		hasLabel = True
		dataHeader.append("BinaryLabel")
		dataHeader.append("LabelColor")
	else:
		hasLabel = False

	dataDict={}

	for event in csvDict:
		if csvDict[event][indexKaggleSet] in kaggleSets:

			row = []

			for featureName in featureList:
				featureID = header.index(featureName)
				featureValue = csvDict[event][featureID]
				row.append(featureValue)

			if hasLabel is True:
				label = csvDict[event][indexLabel]
				if label == "s":
					labelColor = "b" ##signal = blue
					row.append(1)
				elif label == "b":
					labelColor = "r" ##background = red
					row.append(0)
				else:
					print("ERROR in Labels!")
				row.append(labelColor)
				
			##cutting Errors
			if "-999.0" in row and hasErrorValues is False:
				continue

			#row-format: [feat1,feat2,...,featn (,color)]
			dataDict[event] = row
	print("Dictionary-Length:", len(dataDict))
	return dataDict, dataHeader



def getKaggleDataTraining(featureList = None):
	dataDict, dataHeader = getCustomKaggleData(["t"],featureList)
	return dataDict, dataHeader

def getKaggleDataPublic(featureList = None):
	dataDict, dataHeader = getCustomKaggleData(["p"],featureList)
	return dataDict, dataHeader

def getKaggleDataPrivate(featureList = None):
	dataDict, dataHeader = getCustomKaggleData(["v"],featureList)
	return dataDict, dataHeader

def getKaggleDataUnused(featureList = None):
	dataDict, dataHeader = getCustomKaggleData(["u"],featureList)
	return dataDict, dataHeader

def getKaggleDataAll(featureList = None):
	dataDict, dataHeader = getCustomKaggleData(["t","p","v","u"],featureList)
	return dataDict, dataHeader
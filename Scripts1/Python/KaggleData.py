import csv
import os
import numpy as np
import urllib.request
import urllib.error
from lxml import html

scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
dataPath = (mainFolderPath + "/data/")
csvFile = dataPath + "atlas-higgs-challenge-2014-v2.csv"

	##create array out of csv-File
def csvToArray(csvF = csvFile, rows=818238,columns=35):
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
	if not set(featureList).issubset(data_header):
		print(featureList, "\n is no subset of \n", data_header)
		return None
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

def getFeatureSubset(train_data,test_data,train_header,test_header,x):
	header = getFeatureListX(x)
	train = _extractFeatures(header,train_data,train_header).astype(float)
	test = _extractFeatures(header,test_data,test_header).astype(float)
	return header,train,test

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

def getFeatureListX(x):
	x = x - 1
	if x < 1 or x > 9:
		print("Parameter x must be a integer in range [1,9], as indexed in the Thesis.")
		return None
	featureLists = [# Set 1
					["DER_mass_MMC",
            		"DER_mass_transverse_met_lep",
            		"DER_mass_vis",
            		"DER_met_phi_centrality",
            		"DER_pt_ratio_lep_tau",
            		"PRI_tau_pt",
            		"DER_pt_h"],
            		# Set 2
            		['DER_mass_MMC',
            		'DER_mass_transverse_met_lep',
            		'DER_pt_h',
            		'PRI_met',
            		'PRI_tau_pt'],
            		# Set 3
            		['DER_deltaeta_jet_jet',
 					'DER_mass_MMC',
 					'DER_mass_jet_jet',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_pt_h',
 					'DER_pt_ratio_lep_tau',
 					'DER_sum_pt',
 					'PRI_jet_all_pt',
 					'PRI_jet_num',
 					'PRI_jet_subleading_eta',
 					'PRI_lep_eta',
 					'PRI_lep_phi',
 					'PRI_lep_pt',
 					'PRI_tau_pt'],
 					# Set 4
 					['DER_mass_MMC',
 					'DER_mass_jet_jet',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_pt_h',
 					'DER_sum_pt',
 					'PRI_jet_all_pt',
 					'PRI_jet_num',
 					'PRI_lep_eta',
 					'PRI_lep_phi',
 					'PRI_tau_pt'],
 					# Set 5
 					['DER_mass_MMC',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_pt_h',
 					'DER_sum_pt',
 					'PRI_jet_all_pt',
 					'PRI_jet_num',
 					'PRI_tau_pt'],
 					# Set 6
 					['DER_mass_MMC',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_met_phi_centrality',
 					'DER_pt_ratio_lep_tau',
 					'PRI_tau_pt',
 					'DER_pt_h',
 					'PRI_jet_num'],
 					# Set 7
 					['DER_mass_MMC',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_met_phi_centrality',
 					'DER_pt_h',
 					'DER_pt_ratio_lep_tau',
 					'DER_sum_pt',
 					'PRI_jet_all_pt',
 					'PRI_jet_num',
 					'PRI_tau_pt'],
 					# Set 8
 					['DER_deltar_tau_lep',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_pt_h',
 					'DER_pt_ratio_lep_tau',
 					'DER_pt_tot',
 					'PRI_lep_phi',
 					'PRI_lep_pt',
 					'PRI_tau_phi'],
 					# Set 9
 					['DER_mass_MMC',
 					'DER_mass_transverse_met_lep',
 					'DER_mass_vis',
 					'DER_pt_h',
 					'DER_deltaeta_jet_jet',
 					'DER_mass_jet_jet',
 					'DER_prodeta_jet_jet',
 					'DER_deltar_tau_lep',
 					'DER_pt_tot',
 					'DER_sum_pt',
 					'DER_pt_ratio_lep_tau',
 					'DER_met_phi_centrality',
 					'DER_lep_eta_centrality',
 					'PRI_tau_pt',
 					'PRI_lep_pt',
 					'PRI_met',
 					'PRI_met_sumet',
 					'PRI_jet_num',
 					'PRI_jet_leading_pt',
 					'PRI_jet_leading_eta',
 					'PRI_jet_subleading_pt',
 					'PRI_jet_subleading_eta',
 					'PRI_jet_all_pt']
				]
	return featureLists[x]

"""Web-Stuff"""

def getLeaderBoard(url):
	try:
		page = html.fromstring(urllib.request.urlopen(url).read())
	except urllib.error.HTTPError:
		print("Leaderboard-URL not found.")
		raise
		return []

	ids = []
	for userid in page.xpath("//tr[@id]"):
		ids.append(userid.values()[0].split("-")[1])
	scores = []
	for score in page.xpath("//a [@name]"):
		scores.append(score.values()[0])
	ids = np.asarray(ids).astype(float)
	scores = np.asarray(scores).astype(float)
	leaderBoard = np.empty([len(scores),3],dtype=float)
    
	leaderBoard[:,0] = ids[:]
	leaderBoard[:,1] = scores[:]
	leaderBoard[:,2] = np.arange(1,(len(ids)+1))
    
	return leaderBoard

def getLeaderBoards(url="https://www.kaggle.com/c/higgs-boson/leaderboard"):
	pub_url = url + "/public"
	priv_url = url + "/private"
	pub_lB = getLeaderBoard(pub_url)

	try:
		priv_lB = getLeaderBoard(priv_url)
	except urllib.error.HTTPError:
		print("Private Leaderboard has not been found, is the competition still running?"
			+ "\nUse getLeaderBoard(url) for a single leaderboard.")
		return pub_lB,None
	return pub_lB,priv_lB
import csv
import os
import numpy as np
import higgsml_opendata_kaggle3 as higgsK

scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
dataPath = (mainFolderPath + "\\Data")
solPath = (dataPath + "\\Solutions\\")


def getLinRegrSolution():
	import linearregression as linr

	solutionData,threshold = linr.calcSolution()
	solutionData = solutionData.tolist()
	for row in solutionData:
		row[0] = int(row[0])
		row[1] = int(row[1])
		if float(row[2]) < float(threshold):
			eventClass = "b"
		else:
			eventClass = "s"
		row[2]=eventClass
	filename="solution_LinR_thresh" + str(threshold) + ".csv"
	writeSolutionCsv(filename,solutionData)

def getLogRegrSolution():
	import logisticregression as logr
	solutionData = logr.calcSolution()
	solutionData = solutionData.tolist()
	for row in solutionData:
		row[0] = int(row[0])
		row[1] = int(row[1])
		if float(row[2]) <= float(0.5):
			eventClass = "b"
		else:
			eventClass = "s"
		row[2]=eventClass
	filename="solution_LogR.csv"
	return writeSolutionCsv(filename,solutionData)

def writeSolutionCsv(fname,solutionData):
	fname = (solPath + fname)
	print("Writing csv-file:",fname)
	header = ["EventId","RankOrder","Class"]
	with open(fname, 'w', newline='') as csvfile:
	    writer = csv.writer(csvfile, delimiter=',',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    writer.writerow(header)
	    for row in solutionData:
	    	writer.writerow(row)
	return fname


if __name__ == "__main__":
	# getLinRegrSolution()
	higgsK.calculateFor(getLogRegrSolution())

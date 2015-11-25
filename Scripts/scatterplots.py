import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import csv
import random
import os

folderPath = "F:\BA\\"
dataPath = folderPath + "Data\\"
plotPath = folderPath + "Plots\\"
csvFile = dataPath + "Atlas-higgs-challenge-2014-v2.csv"

def cont(header,ignoreIndex,kSet):
	print("Checking for complete Plots...")

	xLabel = ""
	xfolder = ""
	ixData = 1
	xFig = 1
	#check all planned directories
	for ixData in list(range(1,len(header)-1)):
		if ixData not in ignoreIndex:
			xLabel = header[ixData]
			xfolder = (plotPath + xLabel + "\\")
			#if directory NOT exists, step one back
			if not os.path.exists(xfolder):
				if ixData is 1:
					return 1,1
				else:
					ixData = ixData - 1
				break
	xLabel = header[ixData]
	xfolder = (plotPath + xLabel + "\\")
	for xFig in list(range(1,4)):
		
		xfile = str(xLabel + "_set_" + kSet + "_"+ str(xFig) + ".png")
		savepath = str(xfolder + xfile)
		if not os.path.exists(savepath):
			break

	print("continuing with plotting", savepath, "\n")
	return ixData,xFig

def createCsvDictionary(csvF):
	#""" Read solution file, return a dictionary with key EventId and value the row, as well as the header
    #Solution file headers: EventId, Label, Weight """
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

def extractData(csvDict,header,ixdata,iydata,kSet = "all"):
	ilabel=header.index("Label")
	iset=header.index("KaggleSet")

	x = []
	y = []
	colors = []
	for event in csvDict:
		s = csvDict[event][iset]
		if s == kSet or kSet == "all":
			x.append(float(csvDict[event][ixdata]))
			y.append(float(csvDict[event][iydata]))
			label = csvDict[event][ilabel]
			if label == "s":
				labelColor = "b" ##signal = blue
			elif label == "b":
				labelColor = "r" ##background = red
			else:
				print("ERROR in Labels!")
			colors.append(labelColor)
	return x,y,colors

#shuffle given Data but also keep relation || optional
def shuffleData(xdata,ydata,cdata):
	print("      shuffeling Data...")
	x_shuf = []
	y_shuf = []
	c_shuf = []
	index_shuf = list(range(len(xdata)))
	random.shuffle(index_shuf)
	for i in index_shuf:		
	    x_shuf.append(xdata[i])
	    y_shuf.append(ydata[i])
	    c_shuf.append(cdata[i])
	return x_shuf,y_shuf,c_shuf

def scaleAxis(data,cutPercent):
	print("      scaling Axis, cutting", str(cutPercent) + "%")
	sData = sorted(data)
	l = len(sData)
	cutPoint = int(cutPercent*l/100)
	axisMax = sData[l - cutPoint]
	axisMin = sData[cutPoint]
	if axisMin == -999.0:
		for x in sData:
			if x > axisMin:
				axisMin = x
				break
	print("         Datapoints cut:", cutPoint)
	print("         Values:",axisMin, "|", axisMax)
	return axisMin,axisMax


def plotFigure(subN,kSet,xLabel,yLabel,xData,yData,cData,mainFig):
	print("      creating subplot")
	scale = 0.5
	ax = mainFig.add_subplot(3,3,subN)
	ax.scatter(xData, yData, s=1, edgecolor="", c=cData, alpha=0.05)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)
	xmin,xmax = scaleAxis(xData,scale)
	ymin,ymax = scaleAxis(yData,scale)
	# plt.axis([xmin,xmax,ymin,ymax])
	ax.set_xlim(xmin,xmax)
	ax.set_ylim(ymin,ymax)
	xticks = np.linspace(xmin,xmax,5)
	yticks = np.linspace(ymin,ymax,5)
	ax.set_xticks(xticks)
	ax.set_yticks(yticks)
	blue_patch = mpatches.Patch(color='blue', label='signal')
	red_patch = mpatches.Patch(color='red', label='background')
	ax.legend(handles=[blue_patch,red_patch])

	print("   done!")
	return

def plotAll(csvDict,header,shuffle=False,new = False):
	iid=header.index("EventId")
	iweight=header.index("Weight")
	ilabel=header.index("Label")
	iset=header.index("KaggleSet")
	ikweight=header.index("KaggleWeight")
	#masses were not given for the challenge
	imass1=header.index("DER_mass_MMC")
	imass2=header.index("DER_mass_transverse_met_lep")
	imass3=header.index("DER_mass_vis")
	imass4=header.index("DER_mass_jet_jet")
	ignoreMasses = False
	kSet = "all"

	#these features just don't make sense to be plotted
	ignoreIndex = [iid,iweight,ilabel,iset,ikweight]

	if ignoreMasses:
		ignoreIndex.extend([imass1,imass2,imass3,imass4])

	print("ignoring Labels:")
	for i in list(ignoreIndex):
		print("   " + header[i])
	print("\n")

	ixdata = 0
	iydata = 0
	xLabel = ""
	yLabel = ""
	x = []
	y = []
	colors = []
	if new is False:
		xStart,xFig = cont(header,ignoreIndex,kSet)
		xFig = (xFig*9 - 8)
	
	#xdata
	for ixdata in list(range(xStart,len(header)-1)):
		#ignore specified Indezes
		if ixdata not in list(ignoreIndex):
			xLabel = str(header[ixdata])
			print("Start: creating plots for", xLabel)
			subN = 0
			xfolder = (plotPath + xLabel + "\\")
			if not os.path.exists(xfolder):
				print("did not find folder, creating...")
				os.makedirs(xfolder)
			else:
				print("found folder")
			print("Folderpath:", xfolder)
			mainFig = plt.figure(figsize=(32,16))
			figureNumber = 0

			#ydata
			for iydata in list(range(1,len(header)-1)):
				#ignore specified Indezes
				if iydata not in list(ignoreIndex):
					subN = subN+1
					#new figure after nine plots
					if subN is 10:
						subN = 1
						figureNumber = figureNumber + 1
						xfile = str(xLabel + "_set_" + kSet + "_"+ str(figureNumber) + ".png")
						savepath = str(xfolder + xfile)
						print("   saving", xfile, "...")
						mainFig.savefig(savepath)
						plt.close("all")
						mainFig = plt.figure(figsize=(32,16))
					yLabel = str(header[iydata])
					
					print("   " + str(subN) + ": plotting X=", xLabel, "to Y=", yLabel)
					#extract wanted data
					x,y,colors = extractData(csvDict,header,ixdata,iydata,kSet)

					#shuffle lists, keep relationship between 
					if shuffle is True:
						x,y,colors = shuffleData(x,y,colors) ##optional

					#plot figure
					plotFigure(subN,kSet,xLabel,yLabel,x,y,colors,mainFig)

			#save remaining plots in last figure for that x
			figureNumber = figureNumber + 1
			xfile = str(xLabel + "_set_" + kSet + "_"+ str(figureNumber) + ".png")
			savepath = str(xfolder + xfile)
			print("   saving", xfile, "...")
			mainFig.savefig(savepath)
			plt.close("all")
			print("   Figure", xfile, "saved to", savepath)



if __name__ == "__main__":

	plt.close("all")
	#read csv
	folderPath = "F:\BA\\"
	dataPath = folderPath + "Data\\"
	plotPath = folderPath + "Plots\\"
	csvFile = dataPath + "Atlas-higgs-challenge-2014-v2.csv"

	csvDict,header = createCsvDictionary(csvFile)

	plotAll(csvDict,header,shuffle=True)
	print("all done(?)")
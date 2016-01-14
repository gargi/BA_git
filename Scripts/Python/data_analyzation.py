import kaggleData as kD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
plotPath = (mainFolderPath + "/Plots")
scatterPath = (plotPath + "/Scatter") 
histPath = (plotPath + "/Hist") 


def countErrorsByJets(header,data):
    index_PRI_jet_num = np.int(header.index("PRI_jet_num"))
    ######### 0 1 2 3 #########
    totals = np.zeros((4),dtype=np.int)
    errors = np.zeros((4,len(header)),dtype=np.int)
    for i in range(0,len(data[:,index_PRI_jet_num])):
        jets = np.int(data[i,index_PRI_jet_num])
        totals[jets] += 1
        for feat in header:
            index_feat = np.int(header.index(feat))
            if data[i,index_feat] == -999:
                errors[jets,index_feat] +=1
    return totals,errors

def plot_PJN_Errors(header,data):
    totals,errors = countErrorsByJets(header,data)

    font = {'weight' : 'bold',
            'size'   : 10}


    fig = plt.figure(figsize=(15,10))
    
    b = 0
    colors = ['green','red','blue','grey']
    plot = [0,0,0,0]
    for i in [0,1,2,3]:
        plot[i] = plt.bar(left = np.arange(31), height = errors[i,:],
               bottom = b, color=colors[i], alpha=0.5)
        b += errors[i,:]

    plt.yticks(np.unique(b))
    
    plt.ylabel('Error-Values')
    plt.xlabel('Features')
    plt.title('Error-Values in features related to PRI_jet_num (Pjn)')

    plt.xticks(np.arange(31), header, rotation=-90)
    matplotlib.rc('font', **font)
    plt.legend((plot[0][0],plot[1][0],plot[2][0],plot[3][0]), ('Pjn = 0', 'Pjn = 1','Pjn = 2','Pjn = 3',))

def scaleAxis(data,cutPercent):
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
    return axisMin,axisMax

def generateLabelsColors(labels):
    colors = []
    for i in range(0,len(labels)):
        label = labels[i]
        if label == 1:
            labelColor = "b" ##signal = blue
        elif label == 0:
            labelColor = "r" ##background = red
        else:
            print("ERROR in Labels!")
        colors.append(labelColor)
    return colors

def scattered(xName,yName,xData,yData,labels):
    colors = generateLabelsColors(labels)

    scale = 0.5
    xmin,xmax = scaleAxis(xData,scale)
    ymin,ymax = scaleAxis(yData,scale)
    
    scat = plt.scatter(xData, yData, s=1, edgecolor="", c=colors, alpha=0.2)   
    
    plt.xlabel(xName)
    plt.ylabel(yName)
    
    plt.axis([xmin,xmax,ymin,ymax])
    
    title = ("Scatterplot: "+ xName+ " to "+ yName)
    
    blue_patch = mpatches.Patch(color='blue', label='signal')
    red_patch = mpatches.Patch(color='red', label='background')
    plt.legend(handles=[blue_patch,red_patch])
    
    plt.title(title)
    
    return plt

"""
WARNING: This method takes hours to terminate, use with caution!
"""
def createAllScatterplots(header,data,labels,scatterPath,x_size=3,y_size=2):
    n = 0
    p = 1
    
    for x in range(1,len(header)):
        xName = header[x]
        xData = data[:,x]
        xFolder = (scatterPath + "/" + xName)
        if not os.path.exists(xFolder):
            os.makedirs(xFolder)
        y = 1
        while y in range(1,len(header)):
            mainFig = plt.figure(figsize=(20,15))
            for i in range(1,(1+(x_size*y_size))):
                yName = header[y]
                yData = data[:,y]    
                ax = mainFig.add_subplot(x_size,y_size,i)
                y += 1
                ax = scattered(xName,yName,xData,yData,labels)

            savePath = (xFolder + "/" + str(n))
            n += 1
            mainFig.savefig(savePath)
            mainFig = None
            
        plt.close("all")


def histo(featName,data,labels):
    font = {'weight' : 'bold',
            'size'   : 10}

    b = 100

    title = str("Histogram: "+ featName)
    sdata = []
    bdata = []
    for i in range(0,len(data)):
        if labels[i] == 0:
            bdata.append(data[i])
        else:
            sdata.append(data[i])

    colors = generateLabelsColors(labels)
    
    scale = 0.1
    xmin,xmax = scaleAxis(data,scale)
    
    xmin = int(xmin)
    xmax = int(xmax)

    shist = plt.hist(sdata,bins = np.linspace(xmin,xmax,b), normed=1, facecolor='red', alpha=0.3)
    bhist = plt.hist(bdata,bins = np.linspace(xmin,xmax,b), normed=1, facecolor='blue', alpha=0.3)

    plt.legend(('Signal', 'Background'))

    plt.ylabel('Percentage in data')
    plt.xlabel('Values')
    plt.title(title)

    return plt


def createAllHistograms(header,data,labels,histPath,x=3,y=2):
    n = 0
    p = 1
    while p in range(1,len(header)):
        mainFig = plt.figure(figsize=(20,15))
        for i in range(1,(1+(x*y))):
            featName = header[p]
            feat_data = data[:,p]    
            ax = mainFig.add_subplot(x,y,i)
            ax = histo(featName,feat_data,labels);
            p += 1
        savePath = (histPath + "/hist_" + str(n))
        n += 1
        mainFig.savefig(savePath)
        
        mainFig = None
        plt.close("all")
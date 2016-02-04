import math
import os
import numpy as np
import kaggleData as kD
import csv

scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
dataPath = (mainFolderPath + "/Data")
submissionPath = (dataPath + "/Submissions/")


"""
Start of ams-stuff
"""
def calcSetAMS(prediction,sol_data):
    b_pred = np.empty(100000)
    b_labels = np.empty(100000)
    b_weights = np.empty(100000)
    v_pred = np.empty(450000)
    v_labels = np.empty(450000)
    v_weights = np.empty(450000)

    b_i = 0
    v_i = 0
    for i in range(0,len(prediction)):
        pred = prediction[i]
        if sol_data[i,1] == "s":
            label = 1
        else:
            label = 0
        weight = sol_data[i,3]
        if sol_data[i,2] == "b":
            b_pred[b_i]=pred
            b_labels[b_i]=label
            b_weights[b_i]=weight
            b_i += 1
        else: #sol_data[i][2] == "v":#
            v_pred[v_i]=pred
            v_labels[v_i]=label
            v_weights[v_i]=weight
            v_i += 1
    b_ams = calcAMS(b_weights,b_pred,b_labels)
    v_ams = calcAMS(v_weights,v_pred,v_labels)
    return b_ams,v_ams


def calcAMS(weights,predictions,labels):  
    s,b = calcWeightSums(weights,predictions,labels)  
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        print("s=",s,"b=",b)
        exit()
    else:
        ams = math.sqrt(radicand)
        return ams,s,b

def calcWeightSums(weights,predictions,labels):
    s = 0
    b = 0
    for j in list(range(0,len(predictions))):
        prediction = predictions[j]
        label = labels[j]
        weight = weights[j]
        if prediction > 0.:
            if label > 0.:
                s += weight
            else:
                b += weight
    return s,b

def calcMaxAMS(weights,labels):
    ams,s,b = calcAMS(weights,labels,labels)
    print("Found", int(labels.cumsum()[-1]), "signals.")
    print("Weightsums signal:", s, "| background:", b)
    print("Maximum AMS possible with this Data:", ams)
    return ams


def calcMaxSetAMS(sol_data):
    labels = np.empty(len(sol_data))
    for i in range(0,len(sol_data)):
        if sol_data[i][1] == "s":
            labels[i] = 1
        else:
            labels[i] = 0

    b_ams,v_ams = calcSetAMS(labels,sol_data)
    print("Maximum AMS possible with this Data:")
    print("Public Leaderboard:", b_ams)
    print("Private Leaderboard:", v_ams)

def customThreshold(pred,t = 0.5):
    newPred = np.zeros(len(pred))
    for i in range(0,len(pred)):
        if pred[i] > t:
            newPred[i]=1
    return newPred

def bestThreshold(soft_pred,sol_data):
    bestPred = soft_pred
    thresh = 0
    maxAMS = 0
    maxThresh = 0
    newSignals = 0
    for thresh in np.linspace(1.0,0.0,10):
        newPred = customThreshold(soft_pred,thresh)
        b_ams = calcSetAMS(newPred,sol_data)[0][0]
        #print("ams:",ams)
        if b_ams > maxAMS:
            bestPred = newPred
            maxThresh = thresh
            maxAMS = b_ams
    return bestPred,maxAMS,maxThresh

def fullEvaluation(predProb,pred,test_weights,test_labels):
    signals = pred.cumsum()[-1]
    correct = np.equal(pred,test_labels).cumsum()[-1]    
    ams = calcAMS(test_weights,pred,test_labels)[1]
   
    newPred,maxAMS,maxThresh,newSignals = bestThreshold(predProb[:,1],test_weights,test_labels)
    newcorrect = np.equal(newPred,test_labels).cumsum()[-1]
    
    T_eq1,T_eq0 = compareBinaryArrays(test_labels,test_labels)[1:]
    o_eqt,o_eq1,o_eq0 = compareBinaryArrays(pred,test_labels)
    a_eqt,a_eq1,a_eq0 = compareBinaryArrays(newPred,test_labels)
    
    print("Signals in test-data:", test_labels.cumsum()[-1])
    print("Comparison of [o]riginal and [a]djusted predictions:\n"+
          " - [o]Signals read:", signals,"\n"+
          " - [a]Signals read:", newSignals,"\n"+
          " -- Difference:", (signals-newSignals),"\n"+
          " - [o]Correct labels:", o_eqt,"| signals:", o_eq1,"| background:",o_eq0,"\n"+
          " ----- wrong signals:", (T_eq1-o_eq1), "| background:", (T_eq0-o_eq0),"\n"+
          " - [a]Correct labels:", a_eqt,"| signals:", a_eq1,"| background:",a_eq0,"\n"+
          " ----- wrong signals:", (T_eq1-a_eq1), "| background:", (T_eq0-a_eq0),"\n"+
          " -- Difference labels:", (a_eqt-o_eqt),"| signals:", (a_eq1-o_eq1),"| background:",(a_eq0-o_eq0),"\n"+
          " - [o]AMS:", ams,"\n"+
          " - [a]AMS:", maxAMS,"(threshold =",maxThresh,")\n"+
          " -- Difference:", (ams-maxAMS),"\n"
         )
    return pred,newPred

"""
Start of data-generation
"""
def generateFeature(label, mu_s, mu_b, sigma_s=5, sigma_b=5):
    if label is 1:
        mu = mu_s
        sigma = sigma_s
    else:
        mu = mu_b
        sigma = sigma_b
    return np.random.normal(mu,sigma)

"""
createToyData(int n, int dim, float s_prob):
create good seperable toydata for testing-purposes
"""
def createToyData(n = 100,dim = 3,s_prob = 0.05):
    data= np.zeros(shape = (n,dim),dtype=float)
    if dim < 3:
        print("Operation canceled.",
              "Data should have at least one",
              "additional dimension besides weights and labels.",
              "(dim >=3)")
        return None
    data[:,0] = np.random.rand(n) #weights
    for i in range(0,n):
        if data[i,0] <= s_prob: # label-determination
            label = 1
        else:
            label = 0
        data[i,1] = label
        for j in range(2,dim):
            data[i,j]=generateFeature(label,mu_s=(j-1)*5,mu_b=(j-1)*20)
    return data

"""
misc tools
"""
def splitList(xList,n):
    aList = xList[:n]
    bList = xList[n:]
    return aList,bList


def compareBinaryArrays(a,b):
    if np.shape(a) != np.shape(b):
        print("ERROR: Arrays must have same shape.")
        return None
    eq_total = 0
    eq_ones = 0
    eq_zeros = 0
    for i in range(0,len(a)):
        if a[i] == b[i]:
            eq_total += 1
            if a[i] == 1:
                eq_ones += 1
            else:
                eq_zeros += 1
    return eq_total,eq_ones,eq_zeros

def sortByColumn(array,column):
    order = array[:, column].argsort()
    sorted = np.take(array, order, 0)
    return sorted

"""
csv-tools
"""

def createSubmissionArray(eventList,soft_prediction):
    expData = np.empty([len(eventList),3])
    for i  in range(0,len(expData)):
        expData[i]= [eventList[i],0,soft_prediction[i]]

    expData = expData[np.argsort(expData[:,2])]

    for i in range(1,(len(eventList)+1)):
        expData[(i-1),1] = i

    return expData

def createSubmissionFile(eventList,soft_prediction,fname,threshold=0.5):
    expData = createSubmissionArray(eventList,soft_prediction)
    filePath = str(submissionPath + fname)
    outputfile=open(filePath,"w")
    outputfile.write("EventId,RankOrder,Class\n")

    for i in range(0,len(eventList)):
        event = int(expData[i,0])
        rank = int(expData[i,1])
        label = "b"

        if expData[i,2] >= threshold:
            label="s"

        outputfile.write(str(event)+",")
        outputfile.write(str(rank)+",")
        outputfile.write(label)            
        outputfile.write("\n")

    outputfile.close()

#result-recording
def newRunRecord(fname = "records_1.csv"):
    file=open(fname,"w")
    file.write("Classifier,Featurelist,CV_Score,PublicAMS,PrivateAMS,time_train,time_pred,Settings\n")
    file.close()

#classifier,settings,featureList,cv_score,public_ams,private_ams
def recordRun(res,fname = "records_1.csv"):
    #check if fname is valid
    try:
        file=open(fname,"a")
    except:
        print("Error: ",fname,"not found.")
        return
    #check input-array
    if res.dtype != '<U16':
        print("Error, res must be '<U16'-array of shape (20,).")
        print("Data should have form:\n",
              "Classifier,Featurelist,CV_Score,PublicAMS,PrivateAMS,time_train,time_pred,Settings")
        print("Canceling recording.")
        return
    for data in res:
        file.write(str(data + ","))
    file.write("\n")
    file.close()

def getRecord(fname = "records_1.csv"):
    try:
        file=open(fname,"r")
    except:
        print("Error: ",fname,"not found.")
        return

    csvR = csv.reader(file)
    header=next(csvR) # header of Data
    data=next(csvR)
    for row in csvR:
        data = np.vstack((data,row))
    file.close()

    return data,header
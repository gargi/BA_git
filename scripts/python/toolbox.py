"""
Author Michael Janschek.
"""

import math
import os
import numpy as np
import kaggleData as kD
import csv

# set up filepaths
scriptFolderPath = os.path.dirname(os.getcwd())
mainFolderPath = os.path.dirname(scriptFolderPath)
dataPath = (mainFolderPath + "/data")
submissionPath = (dataPath + "/submissions")
recordPath = (dataPath + "/records")



"""""""""start of AMS stuff"""""""""

def calcSetAMS(prediction,sol_data):
    """
    Returns the AMS for both leaderboards.
    Requires an array of 550000 binary predictions.
    """

    # Sets up needed arrays.
    b_pred = np.empty(100000)
    b_labels = np.empty(100000)
    b_weights = np.empty(100000)
    v_pred = np.empty(450000)
    v_labels = np.empty(450000)
    v_weights = np.empty(450000)

    # Converts solution data so it can be used by _calcAMS() iterating through events
    b_i = 0
    v_i = 0
    for i in range(0,len(prediction)):

        pred = prediction[i]

        # convert labels
        if sol_data[i,1] == "s":
            label = 1
        else:
            label = 0
        weight = sol_data[i,3]

        # Checks which leaderboard used this event for scoring. "b" refers to the puBlic leaderboard
        if sol_data[i,2] == "b":
            b_pred[b_i]=pred
            b_labels[b_i]=label
            b_weights[b_i]=weight
            b_i += 1

        # If the event was not used for the public leaderboard, it had to be used for the private one
        else:
            v_pred[v_i]=pred
            v_labels[v_i]=label
            v_weights[v_i]=weight
            v_i += 1

    # Provides _calcAMS() with converted data for actual calculations.

    #Calculates public AMS
    b_ams = _calcAMS(b_weights,b_pred,b_labels)
    #Calculates private AMS
    v_ams = _calcAMS(v_weights,v_pred,v_labels)

    return b_ams,v_ams


def _calcAMS(weights,predictions,labels):  
    """
    Performs AMS calculation.
    Refer to Sect. 2.3 of the thesis for deeper understanding.

    This method is inspired by higgsml_opendata_kaggle3.AMS()
    """

    # Receive sum of weights
    s,b = _calcWeightSums(weights,predictions,labels)  

    # Perform remaining calculation
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        print("s=",s,"b=",b)
        exit()
    else:
        ams = math.sqrt(radicand)
        return ams,s,b


def _calcWeightSums(weights,predictions,labels):
    """
    Calculates the sums of weights for signal and background predictions.
    """

    # Iterates through events
    s = 0
    b = 0
    for j in list(range(0,len(predictions))):
        prediction = predictions[j]
        label = labels[j]
        weight = weights[j]
        if prediction > 0.: # Only events predicted as signal are evaluated
            if label > 0.:
                s += weight
            else:
                b += weight
    return s,b


def calcMaxSetAMS(sol_data):
    """
    Provides a "perfect prediction" for calcSetAMS().
    """

    # Converts solution data to mimic prediction arrays
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
    """
    Thresholds prediction by given value.
    This is needed for AMS optimization.
    """

    newPred = np.zeros(len(pred))
    for i in range(0,len(pred)):
        if pred[i] > t:
            newPred[i]=1
    return newPred


def bestThreshold(soft_pred,sol_data,n=10):
    """
    Determines best threshold to maximize AMS by brute force.
    """

    bestPred = soft_pred
    thresh = 0
    maxAMS = 0
    maxThresh = 0
    newSignals = 0

    # Run tests with n thresholds between 1 and 0
    for thresh in np.linspace(1.0,0.0,n):

        # Thresholds the prediction with current value
        newPred = customThreshold(soft_pred,thresh)

        # Calculates AMS for new thresholded prediction
        b_ams = calcSetAMS(newPred,sol_data)[0][0]

        # Saves result if it beats the former best
        if b_ams > maxAMS:
            bestPred = newPred
            maxThresh = thresh
            maxAMS = b_ams

    return bestPred,maxAMS,maxThresh



"""""""""data generation"""""""""

def _generateFeature(label, mu_s, mu_b, sigma_s=5, sigma_b=5):
    """
    Creates a separable feature for toydata using normal distribution.
    """

    if label is 1:
        mu = mu_s
        sigma = sigma_s
    else:
        mu = mu_b
        sigma = sigma_b
    return np.random.normal(mu,sigma)


def createToyData(n = 100,dim = 3,s_prob = 0.05):
    """
    Creates good seperable toydata for testing-purposes.

    The random values are used as weights.
    """

    data= np.zeros(shape = (n,dim),dtype=float)
    if dim < 3:
        print("Operation canceled.",
              "Data should have at least one",
              "additional dimension besides weights and labels.",
              "(dim >=3)")
        return None

    # Initializes weights and thresholds them.
    data[:,0] = np.random.rand(n) #weights

    # Thresholding
    for i in range(0,n):
        if data[i,0] <= s_prob: # label determination
            label = 1
        else:
            label = 0
        data[i,1] = label

        # Generate separable features for this event
        for j in range(2,dim):
            data[i,j]=_generateFeature(label,mu_s=(j-1)*5,mu_b=(j-1)*20)
    return data


""""""""" misc tools """""""""


def sortByColumn(array,column):
    """
    Copies and sorts an array w.r.t. to a chosen column
    """

    order = array[:, column].argsort()
    sorted = np.take(array, order, 0)
    return sorted


""""""""" csv tools """""""""


def createSubmissionArray(soft_prediction):
    """
    Creates a submission array to be further processed by createSubmissionFile()
    """

    expData = np.empty([len(soft_prediction),3])

    # Prediction is sorted w.r.t. event id, so we rather generate than import the IDs 
    expData[:,0] = np.arange(350000,900000)
    expData[:,2] = soft_prediction[:]
    expData = sortByColumn(expData,2)
    # We sort w.r.t. prediction values and generate the ranks 
    expData[:,1] = np.arange(1,550001)

    return expData


def createSubmissionFile(soft_prediction,fname,threshold=0.5, rankThreshold = False):
    """
    Creates a submission file corresponding to the format specified by the challenge.

    We might not want to threshold by prediction value, but by ranking.
    rankThreshold is used as flag for this decision.
    """

    # If thresholding shall be performed by ranking, proceed to _createSubmissionFileByThreshold()
    if rankThreshold:
        _createSubmissionFileByThreshold(soft_prediction,fname,threshold)
    else:
        # Converts a soft prediction to array and sorts w.r.t. event IDs
        expData = createSubmissionArray(soft_prediction)
        expData = sortByColumn(expData,0)

        # Starts writing submission csv file
        filePath = str(submissionPath + "/" + fname)
        outputfile=open(filePath,"w")
        outputfile.write("EventId,RankOrder,Class\n")

        # Iterates through predicted events, thresholds prediction value to signal and background.
        for i in range(0,len(expData[:,0])):
            event = int(expData[i,0])
            rank = int(expData[i,1])
        
            if expData[i,2] >= threshold:
                label="s"
            else:
                label = "b"

            # Writes the thresholded converted event into submission file
            outputfile.write(str(event)+",")
            outputfile.write(str(rank)+",")
            outputfile.write(label)            
            outputfile.write("\n")

        outputfile.close()


def _createSubmissionFileByThreshold(soft_prediction,fname,threshold):
    """
    Creates a submission file corresponding to the format specified by the challenge.

    This method uses ranking thresholding, as it should be used with ranking classification like xgboost.
    threshold is the procentual amount of ranks, that shall be classified as background.
    """

    # Converts a soft prediction to array and sort w.r.t. event IDs, does not sort yet
    expData = createSubmissionArray(soft_prediction)

    # Starts writing submission csv file
    filePath = str(submissionPath + "/" + fname)
    outputfile=open(filePath,"w")
    outputfile.write("EventId,RankOrder,Class\n")

    # Thresholds by procentual ranking, like "Bottom 86% of predictions as background"
    n_top = int(threshold * len(expData))
    expData[:n_top,2]=0
    expData[n_top:,2]=1

    # Sorts data and continues like createSubmissionFile()
    expData = sortByColumn(expData,0)
    for i in range(0,len(expData[:,0])):
        event = int(expData[i,0])
        rank = int(expData[i,1])
    
        if int(expData[i,2]) == 1:
            label="s"
        else:
            label = "b"

        outputfile.write(str(event)+",")
        outputfile.write(str(rank)+",")
        outputfile.write(label)            
        outputfile.write("\n")

    outputfile.close()


def newRunRecord(fname,headerStr="Classifier,Featurelist,CV_Score,PublicAMS,PrivateAMS,time_train,time_pred,Settings"):
    """
    Creates a new record file with specified columns.
    """

    filePath = str(recordPath + "/" + fname)
    file=open(filePath,"w")
    file.write(str(headerStr+"\n"))
    file.close()


def recordRun(res,fname = "records_1.csv"):
    """
    Records a testing run and saves provided data.
    """

    filePath = str(recordPath + "/" + fname)

    # Checks if fname is valid
    try:
        file=open(filePath,"a")
    except:
        print("Error: ",filePath,"not found.")
        return

    # Checks input array, for our case we specified the format
    if res.dtype != '<U16':
        print("Error, res must be '<U16'-array of shape (20,).")
        print("Data should have form:\n",
              "Classifier,Featurelist,CV_Score,PublicAMS,PrivateAMS,time_train,time_pred,Settings")
        print("Canceling recording.")
        return
    i = 1

    # Writes run's data value wise
    for data in res:
        file.write(str(data))
        if i != 20:
            file.write(",")
        i += 1
    file.write("\n")
    file.close()


def getRecord(fname = "records_1.csv"):
    """
    Reads record file and returns data as array.
    """

    filePath = str(recordPath + "/" + fname)

    # Checks if fname is valid
    try:
        f=open(filePath,"r")
    except:
        print("Error: ",filePath,"not found.")
        return

    # Creates a header and reads data by iterating through each row
    csvR = csv.reader(f)
    header=next(csvR) # header of Data
    data=next(csvR)
    for row in csvR:
        # Stacks up data array
        data = np.vstack((data,row))
    f.close()

    return data,header
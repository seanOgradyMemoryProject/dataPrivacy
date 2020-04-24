import pandas as pd
import numpy as np 
import math
import sys
#read data file
df=pd.read_csv("CASCrefmicrodata.csv") 

#Pre processing
normalized_df=(df-df.mean())/df.std() 
mu, sigma = 0.01, 8500   ##Sigma was changed for each masked data file

# creating a noise with the same dimension as the dataset (1080,13) 
noise = np.random.normal(mu, sigma, [1080,13])
noise_df = df + noise

#create masked data .csv file 
noise_df.to_csv('MaskedData0.3.csv', index=False, header=True) ##This was changed to output different masked files

##Calculate the mean of each column in the original data set
meanAFNLWGT = df["AFNLWGT"].mean()
meanAGI = df["AGI"].mean()
meanEMCONTRB = df["EMCONTRB"].mean()
meanFEDTAX = df["FEDTAX"].mean()
meanPTOTVAL = df["PTOTVAL"].mean()
meanSTATETAX = df["STATETAX"].mean()
meanTAXINC = df["TAXINC"].mean()
meanPOTHVAL = df["POTHVAL"].mean()
meanINTVAL = df["INTVAL"].mean()
meanPEARNVAL = df["PEARNVAL"].mean()
meanFICA = df["FICA"].mean()
meanWSALVAL = df["WSALVAL"].mean()
meanERNVAL = df["ERNVAL"].mean()

##calculate the sum of each column in the original data set
sumAFNLWGT = df["AFNLWGT"].sum()
sumAGI = df["AGI"].sum()
sumEMCONTRB = df["EMCONTRB"].sum()
sumFEDTAX = df["FEDTAX"].sum()
sumPTOTVAL = df["PTOTVAL"].sum()
sumSTATETAX = df["STATETAX"].sum()
sumTAXINC = df["TAXINC"].sum()
sumPOTHVAL = df["POTHVAL"].sum()
sumINTVAL = df["INTVAL"].sum()
sumPEARNVAL = df["PEARNVAL"].sum()
sumFICA = df["FICA"].sum()
sumWSALVAL = df["WSALVAL"].sum()
sumERNVAL = df["ERNVAL"].sum()

##calculate the sum of each column in the masked data set
maskedSumAFNLWGT = noise_df["AFNLWGT"].sum()
maskedSumAGI = noise_df["AGI"].sum()
maskedSumEMCONTRB = noise_df["EMCONTRB"].sum()
maskedSumFEDTAX = noise_df["FEDTAX"].sum()
maskedSumPTOTVAL = noise_df["PTOTVAL"].sum()
maskedSumSTATETAX = noise_df["STATETAX"].sum()
maskedSumTAXINC = noise_df["TAXINC"].sum()
maskedSumPOTHVAL = noise_df["POTHVAL"].sum()
maskedSumINTVAL = noise_df["INTVAL"].sum()
maskedSumPEARNVAL = noise_df["PEARNVAL"].sum()
maskedSumFICA = noise_df["FICA"].sum()
maskedSumWSALVAL = noise_df["WSALVAL"].sum()
maskedSumERNVAL = noise_df["ERNVAL"].sum()


##Using given formula, calculate data loss for each column    
dataLossAFNLWGT =  (((sumAFNLWGT - maskedSumAFNLWGT)**2)/meanAFNLWGT)
print("Total AFNLWGT Data lost: ") 
print(dataLossAFNLWGT)

dataLossAGI = (((sumAGI - maskedSumAGI)**2)/meanAGI)
print("Total AGI Data lost: ") 
print(dataLossAGI)

dataLossEMCONTRB =  (((sumEMCONTRB - maskedSumEMCONTRB)**2)/meanEMCONTRB)
print("Total EMCONTRB Data lost: ") 
print(dataLossEMCONTRB)

dataLossFEDTAX =  (((sumFEDTAX - maskedSumFEDTAX)**2)/meanFEDTAX)
print("Total FEDTAX Data lost: ") 
print(dataLossFEDTAX)

dataLossPTOTVAL =  (((sumPTOTVAL - maskedSumPTOTVAL)**2)/meanPTOTVAL)
print("Total PTOTVAL Data lost: ") 
print(dataLossPTOTVAL)

dataLossSTATETAX =  (((sumSTATETAX - maskedSumSTATETAX)**2)/meanSTATETAX)
print("Total STATETAX Data lost: ") 
print(dataLossSTATETAX)

dataLossTAXINC = (((sumTAXINC - maskedSumTAXINC)**2)/meanTAXINC)
print("Total TAXINC Data lost: ") 
print(dataLossTAXINC)

dataLossPOTHVAL = (((sumPOTHVAL - maskedSumPOTHVAL)**2)/meanPOTHVAL)
print("Total POTHVAL Data lost: ") 
print(dataLossPOTHVAL)

dataLossINTVAL = (((sumINTVAL - maskedSumINTVAL)**2)/meanINTVAL)
print("Total INTVAL Data lost: ") 
print(dataLossINTVAL)

dataLossPEARNVAL = (((sumPEARNVAL - maskedSumPEARNVAL)**2)/meanPEARNVAL)
print("Total PEARNVAL Data lost: ") 
print(dataLossPEARNVAL)

dataLossFICA = (((sumFICA - maskedSumFICA)**2)/meanFICA)
print("Total FICA Data lost: ") 
print(dataLossFICA)

dataLossWSALVAL = (((sumWSALVAL - maskedSumWSALVAL)**2)/meanWSALVAL)
print("Total WSALVAL Data lost: ") 
print(dataLossWSALVAL)

dataLossERNVAL =  (((sumERNVAL - maskedSumERNVAL)**2)/meanERNVAL)
print("Total ERNVAL Data lost: ") 
print(dataLossERNVAL)

##calculate total data loss
totalDataLoss = dataLossAFNLWGT + dataLossAGI + dataLossEMCONTRB + dataLossFEDTAX + dataLossPTOTVAL + dataLossSTATETAX + dataLossTAXINC + dataLossPOTHVAL + dataLossINTVAL + dataLossPEARNVAL + dataLossFICA + dataLossWSALVAL + dataLossERNVAL
print("Total  Data lost: ") 
print(totalDataLoss/1080)

## Euclidean distance between original file and masked file
def distance(record1, record2):

    return (math.sqrt(((record1 - record2) **2).sum()))

def dbrl(original, masked):

    reidentified = 0

    for i in range(df.shape[0]):

        print(i)

        minDist = 100000
        minRecord = -1

        for j in range(noise_df.shape[0]):

            dist = distance(df.iloc[i], noise_df.iloc[j])

            if (dist < minDist):
                minDist = dist
                minRecord = j


        if (minRecord == i):
            reidentified = reidentified + 1

    return reidentified

print(df.shape)
print(dbrl(df, noise_df))
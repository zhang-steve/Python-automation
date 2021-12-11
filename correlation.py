import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import rankdata
import csv
import matplotlib.pyplot as plt
import math
import statistics
import sys

#//////// Changeable Variables ////////////////////////
numberOfBins =  22
failureRateColumnNumber = 49
spearmanWeight = 0.3
pearsonWeight = 0.7
spearmanPValueThreshold = 0.0000001
pearsonPValueThreshold = 0.00000001
top1Top2WaferDifference = 15
binName = ''


def best_fit(xpoints,ypoints):
    xbar = sum(xpoints[1:len(xpoints)])/len(xpoints[1:len(xpoints)])
    ybar = sum(ypoints[1:len(ypoints)])/len(ypoints[1:len(ypoints)])
    n = len(xpoints[1:len(xpoints)]) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(xpoints[1:len(xpoints)], ypoints[1:len(ypoints)])]) - n * xbar * ybar
    denum = sum([xi**2 for xi in xpoints[1:len(xpoints)]]) - n * xbar**2
    if(denum==0):
        b=0
    else:
        b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    #print('\n')
    return a, b

listOfBin = [[] for i in range(numberOfBins)]    # replace 22 with bin number variable
pearsonResults = []     # list that contains the Pearson coefficients only
pearsonPResults = []     # list that contains only the Pearson p values
rankOfpearsonResults = []   # list that contains the ranks of the Pearson coefficients
spearmanResults = []    # list that contains only the Spearman Rho's coefficients
spearmanPResults = []     # list that contains only the Spearman p values

failRate = []   # list that contains the back end failure rates


# get the bin data
with open('temp.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        for i in range(numberOfBins): # replace 22 with bin number variable
            listOfBin[i].append(row[i+2])  

#print(listOfBin)

# convert bin data to float (string originally)
for x in range(len(listOfBin)):
    for i in range(1,len(listOfBin[x])):
        listOfBin[x][i] = float(listOfBin[x][i])

# get the back end failure rates data
with open('temp.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
            failRate.append(row[failureRateColumnNumber])    # 28 has to be a variable
    for i in range(1,len(failRate)):
        failRate[i] = float(failRate[i])

#print(failRate)
#print("HIII: ", len(listOfBin[0]))


####################################################### find Spearman
spearmanResultsabs = []
for i in range(len(listOfBin)):
    s,p = scipy.stats.spearmanr(listOfBin[i],failRate)
    
    if math.isnan(s) is True:    # if coefficient is nan, add 0 instead of nan
        spearmanResults.append(0)
        spearmanResultsabs.append(0)
    else:
        spearmanResults.append(s)
        spearmanResultsabs.append(abs(s))

    spearmanPResults.append(p)
#print(spearmanPResults)
biggest = abs(spearmanResults[0])
highestCoefficientIndex = 0   #index of the highest absolute value coefficient in spearmanResults[]

#find the index of the highest coefficient in the Spearman's list
for i in range(1,len(spearmanResults)):
    current = abs(spearmanResults[i])
    if(current > biggest):
        biggest = current
        highestCoefficientIndex = i

#print("highestCoefficientIndex is ", highestCoefficientIndex)
highestSpearmanBinName = listOfBin[highestCoefficientIndex][0] #  variable that contains the name of the bin that has the highest absolute Spearman's Rho Coefficient

#print(spearmanResultsabs)
print("The bin with the highest Spearman coefficient is " + highestSpearmanBinName)

# rank each graph by their Spearman's coefficient
rankOfBinListspearman = np.ndarray.tolist(scipy.stats.rankdata(spearmanResultsabs))    # list that stores the rank of the bins according to their Spearman Rho's coefficient (Note: lowest coefficient is '1')
#print(rankOfBinListspearman)

####################################################### find Pearson
pearsonResultsabs = []
for i in listOfBin:
    r, p = scipy.stats.pearsonr(i[1:len(i)], failRate[1:len(failRate)])
    
    if math.isnan(r) is True:    # if coefficient is nan, add 0 instead of nan
        pearsonResults.append(0)
        pearsonResultsabs.append(0)
    else:
        pearsonResults.append(r)
        pearsonResultsabs.append(abs(r))

    pearsonPResults.append(p)
#print(pearsonResultsabs)
#print(pearsonPResults)

biggest = abs(pearsonResults[0])
highestCoefficientIndex = 0   #index of the highest absolute value coefficient in pearsonResults[]

#find the index of the highest coefficient in the Pearson's list
for i in range(1,len(pearsonResults)):
    current = abs(pearsonResults[i])
    if(current > biggest):
        biggest = current
        highestCoefficientIndex = i

#print("highestCoefficientIndex is ", highestCoefficientIndex)
highestPearsonBinName = listOfBin[highestCoefficientIndex][0] #  variable that contains the name of the bin that has the highest absolute Spearman's Rho Coefficient

print("The bin with the highest Pearson coefficient is " + highestPearsonBinName)

# rank each graph by their Pearson's coefficient
rankOfBinListpearson = np.ndarray.tolist(scipy.stats.rankdata(pearsonResultsabs))    # list that stores the rank of the bins according to their Spearman Rho's coefficient (Note: lowest coefficient is '1')
#print(rankOfBinListpearson)
totalrank = [(xi*spearmanWeight) + (yi*pearsonWeight) for xi,yi in zip(rankOfBinListspearman, rankOfBinListpearson)]
#print(totalrank)


biggest = totalrank[0]
highestCoefficientIndex = 0   #index of the highest absolute value coefficient in pearsonResults[]

#find the index of the highest coefficient in the totalrank list
for i in range(1,len(totalrank)):
    current = abs(totalrank[i])
    if(current > biggest):
        biggest = current
        highestCoefficientIndex = i

highestTotalBinName = listOfBin[highestCoefficientIndex][0] #  variable that contains the name of the bin that has the highest absolute Spearman's Rho Coefficient
#print("highestCoefficientIndex of highest ranked bin is ", highestCoefficientIndex)
print("The bin with the highest total ranking is " + highestTotalBinName)

totalrankSorted = sorted(totalrank, reverse=True)   # total rank sorted list starting from highest to lowest
#print(totalrankSorted)

#print(pearsonPResults)
#print(spearmanPResults)

indexOfHighestInSortedList = 0
candidateFound = False



# loop that tests if the highest rank bin's Pearson and Spearman p-values passes our threshold,
# if values do not pass then we choose the next highest ranked bin
while(candidateFound is False):

    #print("indexOfHighestInSortedList: ", indexOfHighestInSortedList)

    if (pearsonPResults[highestCoefficientIndex] > pearsonPValueThreshold)  or (spearmanPResults[highestCoefficientIndex] > spearmanPValueThreshold):
        indexOfHighestInSortedList = indexOfHighestInSortedList + 1     # increment the index by 1 to find the second highest rank

        if indexOfHighestInSortedList == len(totalrankSorted):
            print("NONE OF THE BINS HAVE PASSED THE P-VALUE TEST. PLEASE LOWER THE P-VALUE THRESHOLD!\n")
            sys.exit()

        highestCoefficientIndex = totalrank.index(totalrankSorted[indexOfHighestInSortedList])  # find the index of the second highest rank in the totalrank list

    else:   # if the p-values of the candidate pass the threshold, the candidate is the final candidate
        candidateFound = True

highestTotalBinName = listOfBin[highestCoefficientIndex][0] #  variable that contains the name of the bin that has the highest absolute Spearman's Rho Coefficient
#print("highestCoefficientIndex of final chosen bin is ", highestCoefficientIndex)
print("The final chosen bin with the highest total ranking and passed p-value tests is " + highestTotalBinName)
print("Its total rank is ", totalrank[highestCoefficientIndex])

# Get the indices(locations) of the top 5 ranked bins
top5Indices = []
for i in range(5):
    top5Indices.append(totalrank.index(totalrankSorted[i]))

print("\n")
for i in range(5):
    print("#" + str(i+1) + " Bin: ", listOfBin[top5Indices[i]][0])

#print("BYEEEEEE: ", top5Indices)

# listoftop5 = [[] for i in range(5)]  # list that contains the data values (bin values) which will be used  
# badList = [[] for i in range(5)]
# for i in range(5):
#     listoftop5[i] = listOfBin[top5Indices[i]]
#     #median=statistics.median(sorted(listofmedian[1:]))
#     #print("Median: ",median)
#     average = sum(listoftop5[i][1:])/len(listoftop5[i][1:])
#     print("Average: ",average)
#     for j in listoftop5[i][1:]:
#         if(j > average):
#             badList[i].append(j)
#     print("\n")


# Plot Graph of Top 5 Highest Bins

for i in range(5):
    a,b = best_fit(listOfBin[top5Indices[i]][1:],failRate[1:])
    plt.scatter(listOfBin[top5Indices[i]][1:], failRate[1:], s=2,c='red')
    yfit = [a + b * xi for xi in listOfBin[top5Indices[i]][1:]]
    plt.title(listOfBin[top5Indices[i]][0] + ": Correlation")
    plt.xlabel('Front End Fail Rate')
    plt.ylabel('Back End Fail Rate')
    plt.plot(listOfBin[top5Indices[i]][1:], yfit,markersize=0)
    plt.draw()
    plt.show()

#print(badList)
print("================")

binName = input("Which bin's repair density would you like to use? ")

# get the chamber data
chamberBin = [[] for i in range(22)]    # replace 22 with bin number variable, this list is similar to lisOfBin in first part except this list may have some wafers that were 
                                        # originally in listOfBin that are not in chamberBin anymore (after merging with chamber.csv)



# get the updated bin data from chamber_data
with open('chamber_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        for i in range(22): # replace 22 with bin number variable
            chamberBin[i].append(row[i+2])
    
df = pd.read_csv('chamber_data.csv')
averageRepairDensityPerBin = []     # list that contains tuples of the name of each lot and its average repair density
currentLotBinSum = 0    # variable to keep track of the current lot's repair density sums
#currentLotBinAverage = 0
currentLotRepairDensityBin = []     # list that stores the repair densities of the current lot (used to get number of wafers when calculating average)
currentLot = df.at[0,'LotId']   # set current lot to the first 'LotId'
lotRepairDensity = []   # list that stores the 'LotId' and its repair densities
differenceList = []     # list that contains the numerical difference for each step, step name, and top1 chamber in that step: [(difference,(step name,top1 chamber in step)),...,]
#print(currentLot)
#print(df['LotId'])

# find which lot is the most significant by comparing its average repair densities with other lots
for i, (index, row) in enumerate(df.iterrows()):
    #print(i)
    #print(len(df))
    #print(index)    
    #print(row['LotId'])
    if ((df.at[index,'LotId'] is currentLot) and i != len(df) - 1):    # if still processing the same lot
        currentLotBinSum = currentLotBinSum + df.at[index,binName]  # add current repair density to the sum
        currentLotRepairDensityBin.append(df.at[index,binName])     # add current repair density to the list that stores the repair densities for this lot
        #print("CURRENT SUM", currentLotBinSum)
        
    elif (df.at[index,'LotId'] is not currentLot):  # if 'LotId' has changed, meaning moving on to a new lot
        
        averageRepairDensityPerBin.append((currentLot,(currentLotBinSum/len(currentLotRepairDensityBin))))   # calculate the average repair density for that lot in the chosen bin and add to list
        lotRepairDensity.append((currentLot,(binName,currentLotRepairDensityBin.copy())))
        #print("FOR LOT: ", currentLot)
        #print("CURRENT LIST OF REPAIR DENSITIES: ", currentLotRepairDensityBin)
        #print("CURRENT lotRepairDensity: ", lotRepairDensity)
        #print("AVERAGE IS:", averageRepairDensityPerBin)
        #print("TOTAL SUM IS: ", currentLotBinSum)
        #print("FINAL AVG IS: ",averageRepairDensityPerBin)

        currentLotBinSum = 0    # reset currentLotBinSum because new lot
        currentLotRepairDensityBin.clear() # reset currentLotRepairDensityBin because new lot
        #print("AFTER CLEAR LIST OF REPAIR DENSITIES: ", currentLotRepairDensityBin)
        #print("AFTER CLEAR lotRepairDensity: ", lotRepairDensity)
        currentLot = df.at[index,'LotId']   # set currentLot to new lot because lot has changed
        currentLotBinSum = currentLotBinSum + df.at[index,binName]  # add current repair density to the newly reset sum
        currentLotRepairDensityBin.append(df.at[index,binName]) # add current repair density to the newly reset list that stores the repair densities for this lot
        #print("New SUM", currentLotBinSum)

    elif (i == len(df) - 1): # if current row is the last row
        currentLotBinSum = currentLotBinSum + df.at[index,binName]  # add current repair density to the sum
        currentLotRepairDensityBin.append(df.at[index,binName])     # add current repair density to the list that stores the repair densities for this lot
        #print("CURRENT SUM", currentLotBinSum)
        
        averageRepairDensityPerBin.append((currentLot,(currentLotBinSum/len(currentLotRepairDensityBin))))    # calculate the average repair density for that lot in the chosen bin and add to list
        lotRepairDensity.append((currentLot,(binName,currentLotRepairDensityBin.copy())))

        #print("FOR LOT: ", currentLot)
        #print("AVERAGE IS:", averageRepairDensityPerBin)
        #print("TOTAL SUM IS: ", currentLotBinSum)
        #rint("FINAL AVG IS: ",averageRepairDensityPerBin)

#print(lotRepairDensity)
largestRepairDensityIndex = 0    # variable that stores the index of the LotId that has the highest repair Density in averageRepairDensityPerBin (used to grab the 'LotId')
largestRepairDensity = averageRepairDensityPerBin[0][1] # variable to store the highest repair density (by default is set to first one)

# find the lot that has the highest average repair density
for i in range(1,len(averageRepairDensityPerBin)):
    currentRepairDensity = averageRepairDensityPerBin[i][1]

    if(currentRepairDensity > largestRepairDensity):
        largestRepairDensity = currentRepairDensity
        largestRepairDensityIndex = i

largestRepairDensityLot = averageRepairDensityPerBin[largestRepairDensityIndex][0]  #variable that stores the 'LotId' of the highest average repair density
#print(largestRepairDensityLot)


# need a loop to loop throught all the columns
for column in df:
    #if(column == '3500-4L CONTAINER CARBON DRY ETCH::WaferData::Product::Tool::PROCESS_CHAMBER - WAFER_ATTR'):
        #break

    if(column[0] <= "9"):
        #print(column)

        top1RepairDensityIndex = 0  # variable that stores the index of the top1 repair density in the already selected bin
        top2RepairDensityIndex = 0  # variable that stores the index of the top2 repair density in the already selected bin
        top1RepairDensity = 0       # variable that stores the top1 repair density in the already selected bin
        top2RepairDensity = 0       # variable that stores the top2 repair density in the already selected bin
        top2Found = False           # variable to keep track of when we should stop looking for top2 
        copyOfLotBinRepairDensity = []  # list that is a copy of lotRepairDensity[i][1][1] (used to find second largest because after finding largest we change it to 0, and find next largest)
        difference = 0
        top1Chamber = ' '
        top2Chamber = ' '

        for i in range(len(lotRepairDensity)):
            #print("HIIIII", lotRepairDensity[i][0])
            #print("BYEEE", lotRepairDensity[i][1][0])
            if (lotRepairDensity[i][0] == largestRepairDensityLot) and (lotRepairDensity[i][1][0] == binName):
                top1RepairDensity = max(lotRepairDensity[i][1][1])
                copyOfLotBinRepairDensity = lotRepairDensity[i][1][1].copy()
                copyOfLotBinRepairDensity[lotRepairDensity[i][1][1].index(top1RepairDensity)] = 0   # set top1 to 0 after its found
                #print(lotRepairDensity[i][1][1])
                #print(copyOfLotBinRepairDensity)
                #print(top1RepairDensity)

        # find top1 chamber name (can't use index to find because index gotten from lotRepairDensity is only relative within the lot, have to check if repair density
        # is equal to top1RepairDensity in order to find the row
        for index,row in df.iterrows():
            if row[binName] == top1RepairDensity:
                top1Chamber = row[column]   # need to change column name as variabe later
                top1RepairDensityIndex = index
        #print(top1Chamber)
        #print(top1RepairDensityIndex)


        #find top2 chamber
        while(top2Found == False):
            top2RepairDensity = max(copyOfLotBinRepairDensity)

            #print(top2RepairDensity)
            #print(copyOfLotBinRepairDensity)

            for index, row in df.iterrows():    # find the corresponding chamber that is in the same row as top2RepairDensity
                if (row[binName] == top2RepairDensity):
                    
                    if row[column] == top1Chamber:  # if the top2 chamber is the same as top1 chamber, have to find the next top2
                        copyOfLotBinRepairDensity[copyOfLotBinRepairDensity.index(top2RepairDensity)] = 0   # make the current top2 0 in the list, so we can find the next max
                        #print("top2Chamber same as top1: ", row[column])
                    else:   # top2 chamber isn't the same as top1 chamber
                        top2Chamber =  row[column]
                        top2RepairDensityIndex = index
                        top2Found = True
                        #print(top2Chamber)
                        #print(top2RepairDensity)
                        #print(top2RepairDensityIndex)


        difference = top1RepairDensity - top2RepairDensity
        #print("DIFFERENCE: ",difference)
        differenceList.append([[difference,column],[top1Chamber,top2Chamber]])

#print(differenceList)

biggestDifferenceFound = False
differenceListCopy = differenceList.copy()

for x in range(5):
    biggestDifferenceFound = False
    while(biggestDifferenceFound == False):
        
        # find largest difference
        biggestDifference = 0
        biggestDifferenceIndex = 0
        biggestDifferenceTop1ChamberCount = 0
        biggestDifferenceTop2ChamberCount = 0

        for i in range(len(differenceListCopy)):
            currentDifference = differenceListCopy[i][0][0]

            if currentDifference > biggestDifference:
                biggestDifference = currentDifference
                biggestDifferenceIndex = i

        biggestDifferenceStepName = differenceListCopy[biggestDifferenceIndex][0][1]
        biggestDifferenceTop1Chamber = differenceListCopy[biggestDifferenceIndex][1][0]
        biggestDifferenceTop2Chamber = differenceListCopy[biggestDifferenceIndex][1][1]

        

        for index,row in df.iterrows():
            if row[biggestDifferenceStepName] == biggestDifferenceTop1Chamber:
                biggestDifferenceTop1ChamberCount = biggestDifferenceTop1ChamberCount + 1
            elif row[biggestDifferenceStepName] == biggestDifferenceTop2Chamber:
                biggestDifferenceTop2ChamberCount = biggestDifferenceTop2ChamberCount + 1
        
        if (biggestDifferenceTop1ChamberCount - biggestDifferenceTop2ChamberCount) > top1Top2WaferDifference:
            #print("TOP1 HAS WAY MORE WAFER THAN TOP2. MUST FIND NEXT BIGGEST DIFFERENCE!", )
            #print("DIFFERENCE: ", biggestDifference)
            #print("STEP: ",biggestDifferenceStepName)
            #print("TOP1 CHAMBER: ", biggestDifferenceTop1Chamber)
            #print("TOP1 COUNT: ", biggestDifferenceTop1ChamberCount)
            #print("TOP2 COUNT: ", biggestDifferenceTop2ChamberCount)
            #print("\n")
            differenceListCopy[biggestDifferenceIndex][0][0] = 0    # set current biggest difference so we can look for the next biggest in the next loop

        else:
            #print("CURRENT BIGGEST DIFFERENCE IS VALID!")
            print("FOR BIN: ",binName)
            print("#" + str(x+1) + " PROCESS STEP's DIFFERENCE: ",biggestDifference)
            print("PROCESS STEP NAME: ", biggestDifferenceStepName)
            print("TOP1 CHAMBER: ", biggestDifferenceTop1Chamber)
            #print("TOP1 COUNT: ", biggestDifferenceTop1ChamberCount)
            #print("TOP2 COUNT: ", biggestDifferenceTop2ChamberCount)
            print("\n")
            differenceListCopy[biggestDifferenceIndex][0][0] = 0    # set current biggest difference so we can look for the next biggest in the next loop
            biggestDifferenceFound = True
        

    

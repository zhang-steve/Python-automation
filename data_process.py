import csv
import sys
import pandas as pd
import numpy as np

# merge rd and tc file (front-end repair densities with back-end fail rates)
def extract_file():
    # reading two csv files
    data1 = pd.read_csv('rd.csv')
    data2 = pd.read_csv('tc.csv')
    data1.rename(columns={'lotwaf':'WaferId'},inplace=True)
    data1.rename(columns={'fablot':'LotId'},inplace=True)
    data2.rename(columns={'LOT':'LotId'},inplace=True)
    data2.rename(columns={'WAF':'WaferId'},inplace=True)

    #data1.to_csv('246.csv',index=False)
    for col in data1.columns:
        if (col[0] != 'r') and (col != 'WaferId') and (col != 'LotId'):
             data1.drop([col], axis=1, inplace=True)  # Drop the column
    
    
    
    for index, row in data1.iterrows():
        data1.at[index,'WaferId'] = row['WaferId'][-2:]    # only keep the wafer number (i.e we want '01' and not '757L-01')
        data1.at[index,'LotId'] = row['LotId'][0:7]      # don't want last 3 characters of 'LotId" (i.e we want '508757L' and not '508757L.00L')

    for index, row in data1.iterrows():
        if data1.at[index,'WaferId'] != np.nan:
            data1.at[index,'WaferId'] = int(data1.at[index,'WaferId'])

    for index, row in data2.iterrows():        
        if data2.at[index,'WaferId'] != np.nan:
            data2.at[index,'WaferId'] = int(data2.at[index,'WaferId'])

    # using merge function by setting how='inner'
    total_cols=len(data1.axes[1])-2
    print("Number of bin is: ",total_cols)
    output1 = pd.merge(data1, data2, 
                    on=['LotId','WaferId'], 
                    how='inner')
    output1.to_csv('temp.csv', index=False)


# merge Chamber.csv(file created after runnign chamber_delete()) file with rd file
def merge_chamber_data():
    # reading two csv files
    data1 = pd.read_csv('rd.csv')
    data2 = pd.read_csv('chamber.csv')
    df1 = data2.replace(np.nan, ' ', regex=True)
    #data2.dropna(axis=1,how='any',thresh=None,subset=None, inplace=True)

    data1.rename(columns={'lotwaf':'WaferId'},inplace=True)
    data1.rename(columns={'fablot':'LotId'},inplace=True)

    for index, row in data1.iterrows():
        data1.at[index,'WaferId'] = row['WaferId'][-2:]    # only keep the wafer number (i.e we want '01' and not '757L-01')
        data1.at[index,'LotId'] = row['LotId'][0:7]      # don't want last 3 characters of 'LotId" (i.e we want '508757L' and not '508757L.00L')
    
    # drop the columns that are not 'rd..'
    for col in data1.columns:
        if col[0] != 'r' and col != 'WaferId' and col != 'LotId':
             data1.drop([col], axis=1, inplace=True)  # Drop the column


    mergedCol = []  # list that stores the column headers of the columns that have been merged

    for col in df1.columns:
        #print("COL: ", col)
        if col not in mergedCol:
        #print(col.split('::')[0])
            for colInner in df1.columns:
                if (colInner.split('::')[0] == col.split('::')[0]) and (col != colInner):
                    #print("COLINNER: ",colInner)
                    #print("HIIII",df1[col])
                    #print("BYEEE",df1[colInner])
                    df1[col] = df1[col].astype(str)
                    df1[colInner] = df1[colInner].astype(str)
                    df1[col] = df1[col] + ':' + df1[colInner]
                    mergedCol.append(colInner)
                    df1[colInner] = 'A'

    # delete the columns that use the same chamber for all 25 wafers

    for col in df1.columns:  # Loop through columns
        if col != "LotId":  # LotId potentially can be all the same for the entire column, but we don't want to get rid of it
            if len(df1[col].unique()) == 1:  # Find unique values in column along with their length and if len is == 1 then it contains same values
                df1.drop([col], axis=1, inplace=True)  # Drop the column

    # parse 'WaferId' column so that it only contians the Wafer #, and not the LotId as well
    # parse LotID so the last 3 characters are truncated 
    for index, row in df1.iterrows():
        df1.at[index,'WaferId'] = row['WaferId'][-2:]    # only keep the wafer number (i.e we want '01' and not '757L-01')
        df1.at[index,'LotId'] = row['LotId'][0:7]      # don't want last 3 characters of 'LotId" (i.e we want '508757L' and not '508757L.00L') 485470L-01

    data1['WaferId'] = data1['WaferId'].astype(int)
    df1['WaferId'] = df1['WaferId'].astype(int)

    df1.to_csv('123.csv', index=False)
    data1.to_csv('456.csv', index=False)

    

    output1 = pd.merge(data1, df1, 
                    on=['LotId','WaferId'], 
                    how='inner')
    output1.to_csv('chamber_data.csv', index=False)

    

def main():  
    extract_file()
    merge_chamber_data()

if __name__=="__main__":
    main()
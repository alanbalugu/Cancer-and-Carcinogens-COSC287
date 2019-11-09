import numpy as np
import pandas as pd
from pprint import pprint

def main():
    #read data from epa state/chems release cleaned file into pandas dataframe
    myDataFrame = pd.read_csv("epa_data_state_chems_and_releases_cleaned.csv", sep=',', encoding='latin1')
    #run function that sums up avg release estimates per state per year
    stateYearChem(myDataFrame)

def stateYearChem(myDataFrame):
    #create a list of all the state abbreviations and sort alphabetically
    statesList = myDataFrame['STATE_ABBR'].unique()
    statesList.sort()
    #create a list of all the reporting years and sort earliest to latest
    yearsList = myDataFrame['REPORTING_YEAR'].unique()
    yearsList.sort()

    #create a new pandas dataframe
    #stateYearFrame = pd.DataFrame(index=statesList, columns=yearsList)
    stateYearFrame = pd.DataFrame(index=statesList, columns = yearsList)

    #create indexing variable and loop through the states
    index = 0
    for state in myDataFrame['STATE_ABBR']:
        #get the year, whether or not chem is carcinogen, and reporting year
        year = myDataFrame.loc[index, 'REPORTING_YEAR']
        chem = myDataFrame.loc[index, 'CARCINOGEN']
        release = myDataFrame.loc[index, 'AVG_REL_EST']

        if chem is 'Y':
            #if release contained carcinogenic chemicals
            if stateYearFrame.loc[state, year] is np.nan:
                #if the spot in the frame is empty, fill it with release amount
                stateYearFrame.loc[state, year] = release
            else:
                #if the spot in the frame already has a value, add on the release amount
                stateYearFrame.loc[state, year] = stateYearFrame.loc[state, year] + release
        index+=1

    #print to terminal
    print(statesList)
    print(yearsList)
    print(stateYearFrame)
    #write to csv file
    stateYearFrame.to_csv("epa_state_year_frame.csv")

if __name__ == '__main__':
    main()

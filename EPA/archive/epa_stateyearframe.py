import numpy as np
import pandas as pd
from pprint import pprint

def main():
    #read data from other epa file into pandas dataframe
    myDataFrame2 = pd.read_csv("epa_data_state_chems_and_releases_cleaned.csv", sep=',', encoding='latin1')
    #create list of names for columns with numerical data
    numCols2 = ['SUM_REL_EST','AVG_REL_EST','MIN_REL_EST','MAX_REL_EST',
        'STD_REL_EST','VAR_REL_EST','CLEAN_SCORE']
    stateYearChem(myDataFrame2)

def stateYearChem(myDataFrame):
    statesList = myDataFrame['STATE_ABBR'].unique()
    statesList.sort()
    yearsList = myDataFrame['REPORTING_YEAR'].unique()
    yearsList.sort()

    stateYearFrame = pd.DataFrame(index=statesList, columns=yearsList)
    print(stateYearFrame[:10])

    index = 0
    for i in myDataFrame['STATE_ABBR']:
        state = i
        year = myDataFrame.loc[index, 'ACTIVE_DATE']
        chem = myDataFrame.loc[index, 'CARCINOGEN']
        release = myDataFrame.loc[index, 'AVG_REL_EST']
        if chem:
            if stateYearFrame.loc[state, year] is np.nan:
                stateYearFrame.loc[state, year] = release
            else:
                stateYearFrame.loc[state, year] = stateYearFrame.loc[state, year] + release
        index+=1

    print(statesList)
    print(yearsList)
    print(stateYearFrame)
    #write to csv file
    filename="epa_state_year_frame.csv"
    stateYearFrame.to_csv(filename)

main()

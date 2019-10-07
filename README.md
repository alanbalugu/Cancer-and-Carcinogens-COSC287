netids: agb76, pnl8, kie3, akh70

Twitter:

    to download twitter data as csv files:
        run GetTwitterDataByHandle.py
        this file will use twitterhandles.txt to access different handles' data

    to view original twitter data in csv format:
        open UncleanedTwitterData.zip
        contains separate csv files for data from all twitter handles in twitterhandles.txt

    to clean twitter data:
        run twitterDataCleaning.py
        also run combineCSV.py

    to view cleaned epa data in csv format:
        open CleanedTwitterData.zip
        contains separate csv files for cleaned data from all twitter handles in twitterhandles.txt
        also contains one larger csv with compiled cleaned data from all handles

EPA Data:

    to download epa data as csv files:
        run epa_downloader.py

    to view original epa data in csv format:
        open original_epa_data.zip
        contains two files:
            epa_data_state_releases.csv
            epa_data_state_chems_and_releases.csv

    to clean epa data:
        run cleaning_epa.py

    to view cleaned epa data in csv format:
        open cleaned_epa_data.zip
        contains two files:
            epa_data_state_releases_cleaned.csv
            epa_data_state_chems_and_releases_cleaned.csv
            
CDC Data:

    to generate csv files containing data from the CDC
    dataset imported through pandas, run "cdc.py" which
    writes out the original, uncleaned CDC data, as well
    as performs the data cleaning procedures to then create
    a new, cleaned dataset
        this will create "CDC_API.csv" which contains
        the original, uncleaned data from CDC.gov, as
        well as "CDC_API_Clean.csv" which contains the
        dataframe after cleaning, sorting, and removing
        rows containing null data

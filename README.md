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
        open 'original' folder
        contains two files:
            epa_data_state_releases.csv
            epa_data_state_chems_and_releases.csv

    to clean epa data:
        run clean_trim_EPA.py

    to view cleaned epa data in csv format:
        open 'cleaned' folder
        contains two files:
            epa_data_state_releases_cleaned.csv
            epa_data_state_chems_and_releases_cleaned.csv

    to generate a frame of state as rows and years as columns:
        run epa_generate_state_year_frame.py

    to view a frame of state as rows and years as columns:
        open epa_state_year_frame.csv

    to view EPA statistics
        run epa_stats.py

    to view EPA graphs
        open 'epa_graphs' folder

CDC Data:

    to view aggregated cancer stats from 1999-2016 by
    state:
        open USCS_CancerTrends_OverTime_ByState.csv

    to generate csv files containing data from the CDC
        dataset on chronic disease indicators for cancer, 
        run "cdc_download_clean.py" which writes out the original, uncleaned 
        CDC data, as well as performs the data cleaning 
        procedures to then create a new, cleaned dataset
        this will create: 
            "CDC_API.csv" which contains
                the original, uncleaned data from CDC.gov and 
            "CDC_API_Clean.csv" which contains the
                dataframe after cleaning, sorting, and removing
                rows containing null data

    to view CDC API statistics:
        run cdc_api_statistics.py

    to generate NaiveBayes and RandomForest classification and other analysis for USCS:
        run USCS_AlanChau_final.py

    to generate heat maps comparing the regions for USCS:
        run CDC_regions_heatmaps.py

    to run association rule mining for USCS data:
        run AssociationRuleMining.py

    to generate clusters (DBScan, Hierchical and K-means) for USCS data:
        run CdcClustering.py

Merged Data:

    to view our final merged data set contained both CDC and EPA data for every year and state 
        open merged_data.csv
    
    to merge EPA and CDC data into final data set
        run merge_data.py

    to run KNN classification on merged data:
        run kNearestNeighbors.py

    to run decision tree classification on merged data:
        run decision_tree.py

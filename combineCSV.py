#Alan Balu, agb76

import os
import glob
import pandas as pd

#open the directory that the cleaned files are in (created by cleaning script)
os.chdir("DoneCleaning")

#get a list of the files that are .csv
all_cleaned_files = [file for file in glob.glob('*.csv')]

print(all_cleaned_files)


#remove files that are already cleaned
for item in all_cleaned_files:
	if not ("cleaned" in item):
		all_cleaned_files.remove(item)

print(all_cleaned_files)

#combine all the .csv twitter data files by opening as a dataframe and then concatenating
combined_twitter_files = pd.concat([pd.read_csv(file) for file in all_cleaned_files])

#write to a new, compiled csv file
combined_twitter_files.to_csv( "combined_twitter_files.csv", index=False, encoding='utf-8-sig')

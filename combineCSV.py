import os
import glob
import pandas as pd

os.chdir("DoneCleaning")
all_cleaned_files = [file for file in glob.glob('*.csv')]

print(all_cleaned_files)

for item in all_cleaned_files:
	if not ("cleaned" in item):
		all_cleaned_files.remove(item)

print(all_cleaned_files)

combined_twitter_files = pd.concat([pd.read_csv(file) for file in all_cleaned_files])

combined_twitter_files.to_csv( "combined_twitter_files.csv", index=False, encoding='utf-8-sig')

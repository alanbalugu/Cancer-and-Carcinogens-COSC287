import requests, json, csv
from time import sleep

### CONSTANTS
BASE_URL_EPA = "https://enviro.epa.gov/enviro/efservice/"
TABLE_STATE_CHEMS_AND_RELEASES =  "V_TRI_YR_ST_CHEM_SUM_EZ"
TABLE_STATE_RELEASES = "V_TRI_YR_ST_SUM_EZ"

OUTPUT_FILE_NAME_CHEMS_AND_RELEASES = "epa_data_state_chems_and_releases.csv"
OUTPUT_FILE_NAME_RELEASES ="epa_data_state_releases.csv"

START_YEAR = 2006
END_YEAR = 2016
FORMAT = "csv"


def generate_epa_url_year(table_name, year, output_format):
	return BASE_URL_EPA + "/" + table_name + "/reporting_year/" + str(year) + "/" + output_format

# necessary to break download into one request per year to avoid 100,000 row return limit of a single request
def progressive_download(table_name, year_start, year_end, output_format, output_file_name):
	output_file = open(output_file_name, "w")
	first_year = True
	while year_start <= year_end:
		# print("downloading year " + str(year_start))
		full_url = generate_epa_url_year(table_name, year_start, output_format)
		
		# print("requesting: " + full_url)
		response = requests.get(full_url)
		
		decoded_content = response.content.decode('utf-8')	# cleans it to be a nice string in csv format
		
		# writes header to file only the first time
		# also removes table name from column labels for easier reading
		if first_year:
			header = decoded_content[:decoded_content.index('\n')-1]	# get first line, the header
			header = header.replace((table_name + "."),"") 	# remove table name from all labels
			output_file.write(header)
			first_year = False

		# writes actual content to file
		output_file.write(decoded_content[decoded_content.index('\n'):-1])	# removes the header and final new-line char, before appending to database
		
		print("completed year " + str(year_start))

		# if there are more downloads left, pause briefly to give API a break
		if year_start != year_end: 
			# print("pausing 2 secs")
			sleep(2)

		year_start += 1

	output_file.close()

def main():
	print("downloading summary data by state, chemical, and release type")
	progressive_download(TABLE_STATE_CHEMS_AND_RELEASES, START_YEAR, END_YEAR, FORMAT, OUTPUT_FILE_NAME_CHEMS_AND_RELEASES)
	print("downloading summary data by state, and release type")
	progressive_download(TABLE_STATE_RELEASES, START_YEAR, END_YEAR, FORMAT, OUTPUT_FILE_NAME_RELEASES)
		

if __name__ == '__main__':
	main()
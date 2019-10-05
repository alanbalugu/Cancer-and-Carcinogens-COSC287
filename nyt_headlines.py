import requests, json, csv
from time import sleep

### CONSTANTS
# base_url_epa =  "https://enviro.epa.gov/enviro/efservice/tri_facility/"
BASE_URL_NYT = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
# twitter_full_url = 'https://api.twitter.com/1.1/geo/id/fd70c22040963ac7.json'

# Sample call: post_params = generate_params_epa(state="DC",format="EXCEL")
def generate_params_epa(state="VA", date="08/19/2016", format="EXCEL"):
	post_params = {
					'state_abbr':state,
					'rows': "1:10"
			  	  }
	return post_params

def main():
	full_url_nyt = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?q=election&api-key=oQI2zQoo3Zl0ozq9nOKoGYbGOziAI2mp&fq=headline:("Texas") AND headline:("pollution")'
	# response = requests.get(base_url_epa, post_params)	# create request from base_url and params
	response = requests.get(full_url_nyt)	# create request from base_url and params
	response = requests.get(full_url_nyt) # get request from full url
	print("response:\n" + str(json.dumps(response.json(),indent=4)))
	print("sleeping to avoid call limit")
	sleep(6)
	print("done sleeping")

if __name__ == '__main__':
	main()
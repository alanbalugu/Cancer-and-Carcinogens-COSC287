# TO-DOs
# - filter out retweets
# - filter for english tweets only
# - filter for geo-tagged tweets
# 

import json
from pprint import pprint
from glob import glob

### CONSTANTS
single_keywords = ["pollution", "smog"]
# combo_keywords = [['toxic','water'],["unsafe", "water"],['poisoned', 'water']]
pollution_tweets = []
MAX_FILES = 300 # set to 100000 to allow all files to be read

def main():
	file_counter = 0
	# loops all files in directory until it hits max files count.
	for file_name in glob("./*.json"):
		if (file_counter > MAX_FILES):
			break
		
		file = open(file_name, 'r')
		print ('opening '+ str(file_name))

		# reads all line in file, one by one instead of all at once to be memory efficient
		for line in file:
			unknown_error = False
			tweet = {}
			try:
				tweet = json.loads(line)
				# searches for geo-tagged tweets containing a pollution keyword and appends them to pollution_tweets
				if any(x in tweet['text'] for x in single_keywords): # and ((tweet['geo'] is not None) or (tweet['coordinates'] is not None)) :
					pollution_tweets.append(tweet)
					# print('found geo-tagged pollution tweet')
			except Exception as e:
				try:
					if tweet['delete']:  # tests if its deleted tweets, instead of normal tweet. Deleted tweets cause errors but that can be ignored
						pass
				except:
					unknown_error = True

				# Any other exceptions are unexpected and will be raised
				if unknown_error:
					print('this line failed. unknown_error. exiting!')
					print(line)
					exit()
		file_counter += 1

	# outputs pollution tweets to json formattted txt file.
	output_file = open("pollution_tweets.txt",'w')
	output_file.write(json.dumps(pollution_tweets,indent=4))
	output_file.close()
	print("done!")

if __name__ == '__main__':
	main()
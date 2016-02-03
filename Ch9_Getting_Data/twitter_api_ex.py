"""
Twitter API code example

This program demonstrates how to use the twython library to interact with the twitter API. Specifically we will be getting a sample of all (1000) of the latest tweets

"""

import json
from twython import TwythonStreamer

tweets = []

class MyStreamer(TwythonStreamer):
    """ our own sublass that inherits all methods of twython streamer
    speecifying how to interact with the stream"""

    def on_success(self,data):
        """ What to do when we get tweets, here data will be a python dict
        representing a tweet"""
    
        # only want english tweets, not all tweets contain language so check
        if 'lang' in data and data['lang'] == 'en':
            tweets.append(data)
            print "Received tweet #", len(tweets)

        # stop when we have collected 1000
        if len(tweets) >= 100:
            self.disconnect()

    def on_error(self, status_code, data):
        print status_code, data
        self.disconnect()

# Load the credentials.json
with open('twitter_credentials.json') as d:
    credentials = json.load(d)
    
# initialize the stream and start it up
stream = MyStreamer(str(credentials['Consumer Key']), 
                    str(credentials['Consumer Secret']),
                    str(credentials['Access Token']),
                    str(credentials['Access Token Secret']))

stream.statuses.sample()


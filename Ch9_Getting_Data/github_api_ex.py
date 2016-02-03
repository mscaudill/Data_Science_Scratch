"""
github_api_ex.py

In this module, we exlpore the use of application programming interfaces
(APIs). Many websites provide APIs which when queried provide data in a
strucutred format called json. These objects are strings (since HTTP only
transmits strings and look very similar to Pyton Dicts. We use pythons json
module to parse these objects.

"""
# Simple json deserialization ex
import requests
import json
serialized = """{"title":"Data Science Book", 
                 "author":"Joel Grus",
                 "publicationYear":2014,
                 "topics":["data","science", "data science"]}"""
# parse the json object to create a python dict
deserialized = json.loads(serialized)
#print deserialized

# most APIs require you to authenticate but we can do some simple things with the github API without authentication.

# set the location of my github repository
endpt = "https://api.github.com/users/mscaudill/repos"

repos = json.loads(requests.get(endpt).text)

# This returns a list of dictionaries for my repositories on Github
# for example we can access the various pieces of info about the repositories (only one currently)
print (str(repos[0]["name"]).upper() + " : " + \
        str(repos[0]["description"]) + \
        ' is a repository written in ' + \
        str(repos[0]["language"]))

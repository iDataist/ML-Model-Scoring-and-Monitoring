# import subprocess
import requests

#Specify a URL that resolves to your workspace
URL = "http://192.168.0.4:8000/"

#Call each API endpoint and store the responses
response1 = requests.get(URL).text
response2 = requests.get(URL + 'score').text
response3 = requests.get(URL + 'summary_stats').text
response4 = requests.get(URL + 'timing').text
json = {"lastmonth_activity": {"0": 234},"lastyear_activity": {"0": 3},"number_of_employees": {"0": 10}}
response5 = requests.post(URL + 'predict', json = json).text

print(response1, response2, response3, response4, response5)





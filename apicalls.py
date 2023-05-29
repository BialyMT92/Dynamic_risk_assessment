import requests


#Call each API endpoint and store the responses
response1 = requests.post("http://127.0.0.1:8000/prediction?data_location=testdata/testdata.csv").content
response2 = requests.post("http://127.0.0.1:8000/scoring?model_location=production_deployment/trainedmodel.pkl").content
response3 = requests.get('http://127.0.0.1:8000/summarystats').content
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content

response = [response1, response2, response3, response4]

with open ('apireturns.txt', 'w') as file:
    file.write(str(response))


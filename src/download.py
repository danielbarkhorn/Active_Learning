import requests
import gzip

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'
filename = 'data/raw/SUSY.csv'
with open(filename, 'wb') as ofile:
   response = gzip.open(requests.get(url))
   ofile.write(response.content)

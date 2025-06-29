import os
import urllib.request as request
import zipfile

data_url = "https://github.com/entbappy/Branching-tutorial/raw/master/articles.zip"

def download_data():
    filename, headers = request.urlretrieve(
        url=data_url,
        filename="articles.zip",
    )

download_data()

with zipfile.ZipFile("articles.zip", 'r') as zip_ref:
    zip_ref.extractall()

os.remove("articles.zip")

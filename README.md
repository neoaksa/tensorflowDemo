### 1.mnist Demo.py
#### This is a tensorflow demo using the famous mnist data
MLP structure is dynamic
```python
structure = [784,500,500,10]
x = tf.placeholder(dtype='float',shape=[None,structure[0]])
y = tf.placeholder(dtype='float')
batch_size = 100
epoch_max = 10
```
#### The model inculdes two part: feedforward and traning
```python
# feedforward Model
def netural_network_model(x):
#training
def train_model(x):
```

### 2.word sentiment model
#### put the training and test dataset into folder “sentiment140” which you can download from [here](http://help.sentiment140.com/for-students/)
```python
# word2vec.py is used for word to vector using one-hot coding
# ANN.py is ANN structure and training , testing parts
```
wordSentiment.ipynb is jupter version running on google Colaboratory. the majory different is reading datasource from google drive, may other subtle changes.
```python
! pip install pydrive
# these classes allow you to request the Google drive API
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive 
from google.colab import auth 
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def downloadFile(inputfilename,outputfilename):
    downloaded = drive.CreateFile({'id': inputfilename})
    # assume the file is called file.csv and it's located at the root of your drive
    downloaded.GetContentFile(outputfilename)
    
# traning file download
trainingFile = downloadFile("1adyPElLZ118U1aKVEeqrsVNX4b-VoVDm","training.1600000.processed.noemoticon.csv")
# test file download
testingFile = downloadFile("1-6lzGSZ-IkIjYiUULuhpoADPe55aSWcR","testdata.manual.2009.06.14.csv")
```
**Beware that you should keep your brower active, otherwise the process will be stopped if long time computation is taken. Personlly, I installed an app to keep computer waked up.**

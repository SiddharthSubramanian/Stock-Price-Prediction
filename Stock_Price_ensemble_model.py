import requests # interaction with the web
import os  #  file system operations
import yaml # human-friendly data format
import re  # regular expressions
import pandas as pd # pandas... the best time series library out there
import datetime as dt # date and time functions
import io
import csv
import numpy as np
from sklearn import svm 
from sklearn.metrics import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Activation, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import optimizers
from keras.optimizers import SGD
# fix random seed for reproducibility
numpy.random.seed(7)


## data extraction 
url = 'https://uk.finance.yahoo.com/quote/AAPL/history' # url for a ticker symbol, with a download link
r = requests.get(url)  # download page
txt = r.text # extract html
cookie = r.cookies['B'] # the cooke we're looking for is named 'B'
print('Cookie: ', cookie)
pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')
for line in txt.splitlines():
    m = pattern.match(line)
    if m is not None:
        crumb = m.groupdict()['crumb']
# create data directory in the user folder
dataDir = os.path.expanduser('~')+'/twpData'

if not os.path.exists(dataDir):
    os.mkdir(dataDir)
data = {'cookie':cookie,'crumb':crumb}
dataFile = os.path.join(dataDir,'yahoo_cookie.yml')

with open(dataFile,'w') as fid:
    yaml.dump(data,fid)

sDate = (2009,1,1)
eDate = (2017,9,15) 
dt.datetime(*sDate).timestamp() # convert to seconds since epoch

# prepare input data as a tuple
data = (int(dt.datetime(*sDate).timestamp()),
        int(dt.datetime(*eDate).timestamp()), 
        crumb)
url = "https://query1.finance.yahoo.com/v7/finance/download/SBIN.NS?period1={0}&period2={1}&interval=1d&events=history&crumb={2}".format(*data)
data = requests.get(url, cookies={'B':cookie})
buf = io.StringIO(data.text) # create a buffer
df = pd.read_csv(buf,index_col=0) # convert to pandas DataFrame
df2 = df[(df.High != "null")]

def creating_binary_labels(close_list, open_price_list):
    label_list = []
    for i in range(1,len(open_price_list)):
        if(close_list[i -1] - open_price_list[i] >= 0):
            label_list.append(1)
        else:
            label_list.append(0)
    return label_list

def svm_rbf(feature, label_list):

    length_feature = len(feature)
    len_train = int(0.75*length_feature)
    train_feature = feature[0: len_train]
    test_feature = feature[len_train: ]

    train_label = label_list[0:len_train]
    test_label = label_list[len_train:]


    clf = svm.SVC(C=100000,kernel='rbf')
    clf.fit(train_feature, train_label)
    predicted = clf.predict(test_feature)
    print("Accuracy: ", accuracy_score(predicted, test_label)*100, "%")
    print("Precision Score :", precision_score(predicted, test_label)*100, "%")
    print("Recall Score :" ,recall_score(predicted, test_label)*100, "%")
    return predicted, test_label, train_feature, train_label, test_feature

def neural_networks(train_feature, train_label, test_features, test_labels):
    net = buildNetwork(len(train_feature[0]), 30, 1, hiddenclass = TanhLayer, outclass = TanhLayer,recurrent = True)
    ds = ClassificationDataSet(len(train_feature[0]), 1)
    for i, j in zip(train_feature, train_label):
        ds.addSample(i, j)
    trainer = BackpropTrainer(net, ds)
    epochs = 13
    for i in range(epochs):
        trainer.train()
    predicted = list()
    for i in test_features:
        predicted.append(int(net.activate(i)>0.5))
    predicted = np.array(predicted)

    print("Accuracy:", accuracy_score(test_labels, predicted)*100, "%")
    return predicted

all_features = []
timestamp_list =[]
close_list = []
high_list = []
low_list = []
open_price_list =[]
volume_list = []

timestamp_list = df2['timestamp_list']
close_list = df2['close_list']
high_list =  df2['high_list']
open_price_list = df2['open_price_list']
volume_list = df2['volume_list']
low_list = df2['low_list']
high_list = high_list.astype(float)
close_list = close_list.astype(float)
low_list = low_list.astype(float)
open_price_list = open_price_list.astype(float)
volume_list = volume_list.astype(float)

def fearure_creation(timestamp_list, close_list, high_list, low_list, open_price_list, volume_list, x):
    #Initialising
    open_change_percentage_list=[]
    close_change_percentage_list=[]
    low_change_percentage_list=[]
    high_change_percentage_list=[]
    volume_change_percentage_list=[]    
    volume_diff_percentage_list=[]
    open_diff_percentage_list=[]
    Open_price_moving_average_list=[]
    Close_price_moving_average_list=[]
    High_price_moving_average_list=[]
    Low_price_moving_average_list=[]


    highest_open_price = open_price_list[0]
    lowest_open_price = open_price_list[0]
    highest_volume = volume_list[0]
    lowest_volume = volume_list[0]
    if(x>len(open_price_list)):
        x = len(open_price_list)
    for i in range(len(close_list)-x,len(close_list)):
        if(highest_open_price<open_price_list[i]):
            highest_open_price=open_price_list[i]
        if(lowest_open_price>open_price_list[i]):
            lowest_open_price=open_price_list[i]
        if(highest_volume<volume_list[i]):
            highest_volume=volume_list[i]
        if(lowest_volume>volume_list[i]):
            lowest_volume=volume_list[i]


    #Finding change percentage list/difference list
    opensum=open_price_list[0]
    closesum=close_list[0]
    highsum=high_list[0]
    lowsum=low_list[0]
    for i in range(1, len(close_list)-1):
        close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
        close_change_percentage_list.append(close_change_percentage)
        
        open_change_percentage = (open_price_list[i+1] - open_price_list[i])/open_price_list[i]
        open_change_percentage_list.append(open_change_percentage)

        high_change_percentage = (high_list[i] - high_list[i-1])/high_list[i-1]
        high_change_percentage_list.append(high_change_percentage)
        if volume_list[i-1]==0:
            volume_list[i-1] = volume_list[i-2]

        volume_change_percentage = (volume_list[i] - volume_list[i-1])/volume_list[i-1]
        volume_change_percentage_list.append(volume_change_percentage)

        low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
        low_change_percentage_list.append(low_change_percentage)


        volume_diff = (volume_list[i] - volume_list[i-1])/(highest_volume-lowest_volume)
        volume_diff_percentage_list.append( volume_diff)

        open_diff = (open_price_list[i+1] - open_price_list[i])/(highest_open_price - lowest_open_price)
        open_diff_percentage_list.append(open_diff)

        opensum+=open_price_list[i]
        closesum+=close_list[i]
        highsum+=high_list[i]
        lowsum+=low_list[i]

        Open_price_moving_average = float(opensum/i+1) / open_price_list[i+1]
        Open_price_moving_average_list.append(Open_price_moving_average)

        High_price_moving_average = float(highsum/i+1) / high_list[i+1]
        High_price_moving_average_list.append(High_price_moving_average)

        Close_price_moving_average = float(closesum/i+1) / close_list[i+1]
        Close_price_moving_average_list.append(Close_price_moving_average)

        Low_price_moving_average = float(lowsum/i+1) / low_list[i+1]
        Low_price_moving_average_list.append(Low_price_moving_average)

            
    
    #Combining features
    close_change_percentage_list = np.array(close_change_percentage_list)
    high_change_percentage_list = np.array(high_change_percentage_list)
    low_change_percentage_list = np.array(low_change_percentage_list)
    volume_change_percentage_list = np.array(volume_change_percentage_list)
    open_price_list = np.array(open_price_list)
    close_list = np.array(close_list)
    open_diff_percentage_list=np.array(open_diff_percentage_list)
    volume_change_percentage_list=np.array(volume_change_percentage_list)
    
    feature1 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list))  
    feature2 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list, open_diff_percentage_list, volume_diff_percentage_list))  
    feature3 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list, Open_price_moving_average_list, Close_price_moving_average_list, High_price_moving_average_list, Low_price_moving_average_list))  
    feature4 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list, open_diff_percentage_list, volume_diff_percentage_list,Open_price_moving_average_list, Close_price_moving_average_list, High_price_moving_average_list, Low_price_moving_average_list))
    label_list = creating_binary_labels(close_list, open_price_list)
    return feature1, feature2, feature3, feature4, label_list
x = 5
feature1, feature2, feature3, feature4, label_list = fearure_creation(timestamp_list, close_list, high_list, low_list, open_price_list, volume_list, x )
predicted4, test_label4, train_feature4, train_label4, test_feature4 = svm_rbf(feature4, label_list[0:2115])

print("SVM - RBF Kernel with Features : ")
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
predicted1, test_label1, train_feature1, train_label1, test_feature1 = svm_rbf(feature1, label_list)
print("SVM - RBF Kernel with Features : ")
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
print("Open Difference% , Volume Difference%, ")
predicted2, test_label2, train_feature2, train_label2, test_feature2= svm_rbf(feature2, label_list)
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
print("Open MovingAvg, Close MovingAvg, High MovingAvg, Low MovingAvg")
predicted3, test_label3, train_feature3, train_label3, test_feature3= svm_rbf(feature3, label_list)
print("SVM - RBF Kernel with Features : ")
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
print("Open Difference% , Volume Difference%, Open Price Moving Avg")
print("Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg")
predicted4, test_label4, train_feature4, train_label4, test_feature4 =  svm_rbf(feature4, label_list)
print("*******************************RNN*************************************")
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
predicted_NN_1=neural_networks(train_feature1, train_label1, test_feature1, test_label1)
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
#print("RNN with Features : "
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
print("Open Difference% , Volume Difference%, ")
predicted_NN_2=neural_networks(train_feature2, train_label2, test_feature2, test_label2)
print("-----------------------------------------------------------------------")
   
print("-----------------------------------------------------------------------")
#print("RNN with Features : "
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
print("Open MovingAvg, Close MovingAvg, High MovingAvg, Low MovingAvg")
predicted_NN_3=neural_networks(train_feature3, train_label3, test_feature3, test_label3)
print("Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
print("Open Difference% , Volume Difference%, Open Price Moving Avg")
print("Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg")
predicted_NN_4=neural_networks(train_feature4, train_label4, test_feature4, test_label4)

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data
def input_reshape(n_timesteps, train):
    input_num = 0
    output = np.zeros((len(train)-(n_timesteps-1), n_timesteps, train.shape[1]))
    for i in range(n_timesteps-1, len(train)):
        output[input_num] = train[i-(n_timesteps-1):i+1]
        input_num+=1
    
    return output

global_start_time = time.time()
epochs  = 500
seq_len = 50
n_timesteps = 2

print('> Loading data... ')
print('> Data Loaded. Compiling...')
model =  Sequential()
# 	model.add(Dense(11 ,input_dim=11, kernel_initializer='normal', activation='relu'))
model.add(LSTM(32, input_dim=11, input_length=n_timesteps))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    input_reshape(n_timesteps, train_feature4),
    train_label4[n_timesteps-1:],
    batch_size=10,
    nb_epoch=epochs,
    validation_split=0.2)
print('Training duration (s) : ', time.time() - global_start_time)
testdata = input_reshape(n_timesteps, test_feature4)
testdata_label = test_label4[n_timesteps-1:]
output_pred = model.predict_classes(testdata)
output_prob = model.predict(testdata)
model.evaluate(testdata, testdata_label)

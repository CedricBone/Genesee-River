# Genesee River Predictions** (GRP) 1.2
# https://dashboard.waterdata.usgs.gov/app/nwd/?region=lower48&aoi=default

import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

# The function AAAAA changes the values in list to int or float
def AAAAA(list):
  row = 0
  for data in list:
    if(data == ''):
      data = -1
      list[row] = -1
    else:
      try:
        list[row] = int(data)
      except:
        list[row] = float(data)
    row+=1

dates = []           # Date
times = []           # Time (EST)
discharge = []       # Rate of Discharge (FT^3/s)
temperature = []     # Temperature (C)
PH = []              # PH 
dissolved_o2 = []    # Dissolved Oxygen (mg/L)
gauge_height = []    # Gauge Height (Ft)
year = []            # Year
month = []           # Month
day = []             # Day
hour = []            # Hour
minute = []          # Minute


with open('Rochester(11-21).txt', newline = '') as train_riverDataSets:
    riverDataSet_reader = csv.reader(train_riverDataSets, delimiter='\t')
    for train_riverDataSet in riverDataSet_reader:
      
      dates.append(train_riverDataSet[2][0:10])
      times.append(train_riverDataSet[2][11:])
      discharge.append((train_riverDataSet[4]))
      temperature.append((train_riverDataSet[6]))
      PH.append((train_riverDataSet[8]))
      dissolved_o2.append((train_riverDataSet[10]))
      gauge_height.append((train_riverDataSet[12]))

for date in dates:
    year.append(int(date[0:4]))
    month.append(int(date[5:7]))
    day.append(int(date[8:]))
    
for time in times:
    hour.append(int(time[0:2]))
    minute.append(int(time[3:]))

AAAAA(discharge)
AAAAA(temperature)
AAAAA(PH)
AAAAA(dissolved_o2)
AAAAA(gauge_height)
AAAAA(year)
AAAAA(month)
AAAAA(day)
AAAAA(hour)
AAAAA(minute)

train_riverData = []   # List of lists containing [year, month, day, hour, minute, discharge, temperature, PH, dissolved_o2]

for number in range(len(dates)):
    tempList = []
    tempList.append(year[number])
    tempList.append(month[number])
    tempList.append(day[number])
    tempList.append(hour[number])
    tempList.append(minute[number])
    tempList.append(discharge[number])
    tempList.append(temperature[number])
    tempList.append(PH[number])
    tempList.append(dissolved_o2[number])
    train_riverData.append(tempList)

# normalize data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_riverData))

# This function builds and compiles the model
def build_and_compile_model(norm):
  model = keras.Sequential([
      #keras.Input(shape=(9)),
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
      ])
  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)

%%time
history = dnn_model.fit(
    np.array(train_riverData),
    np.array(gauge_height),
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

#!mkdir -p saved_model
#dnn_model.save('/content/drive/MyDrive/Fall 2021/Engineering Intramurals')

dates = []           # Date
times = []           # Time (EST)
discharge = []       # Rate of Discharge (FT^3/s)
temperature = []     # Temperature (C)
PH = []              # PH 
dissolved_o2 = []    # Dissolved Oxygen (mg/L)
gauge_height = []    # Gauge Height (Ft)
year = []            # Year
month = []           # Month
day = []             # Day
hour = []            # Hour
minute = []          # Minute


with open('Rochester(21-21).txt', newline = '') as test_riverDataSets:
    riverDataSet_reader = csv.reader(test_riverDataSets, delimiter='\t')
    for test_riverDataSet in riverDataSet_reader:
      
      dates.append(test_riverDataSet[2][0:10])
      times.append(test_riverDataSet[2][11:])
      discharge.append((test_riverDataSet[4]))
      temperature.append((test_riverDataSet[6]))
      PH.append((test_riverDataSet[8]))
      dissolved_o2.append((test_riverDataSet[10]))
      gauge_height.append((test_riverDataSet[12]))

for date in dates:
    year.append(int(date[0:4]))
    month.append(int(date[5:7]))
    day.append(int(date[8:]))
    
for time in times:
    hour.append(int(time[0:2]))
    minute.append(int(time[3:]))

AAAAA(discharge)
AAAAA(temperature)
AAAAA(PH)
AAAAA(dissolved_o2)
AAAAA(gauge_height)
AAAAA(year)
AAAAA(month)
AAAAA(day)
AAAAA(hour)
AAAAA(minute)

test_riverData = []   # List of lists containing [year, month, day, hour, minute, discharge, temperature, PH, dissolved_o2]

for number in range(len(dates)):
    tempList = []
    tempList.append(year[number])
    tempList.append(month[number])
    tempList.append(day[number])
    tempList.append(hour[number])
    tempList.append(minute[number])
    tempList.append(discharge[number])
    tempList.append(temperature[number])
    tempList.append(PH[number])
    tempList.append(dissolved_o2[number])
    train_riverData.append(tempList)

#test_results = {}
#test_results['dnn_model'] = dnn_model.evaluate(test_riverData, gauge_height, verbose=0)
test_predictions = dnn_model.predict(test_riverData)
print(gauge_height)
print(test_predictions)

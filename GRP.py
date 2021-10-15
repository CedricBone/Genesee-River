# Genesee River Predictions (GRP) 0.0
# https://dashboard.waterdata.usgs.gov/app/nwd/?region=lower48&aoi=default

import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

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
riverData = []       # List of lists of each observation 

with open('Rochester(11-21).txt', newline = '') as riverDataSets:
    riverDataSet_reader = csv.reader(riverDataSets, delimiter='\t')
    for riverDataSet in riverDataSet_reader:
        dates.append(riverDataSet[2][0:10])
        times.append(riverDataSet[2][11:])
        discharge.append((riverDataSet[4]))
        temperature.append((riverDataSet[6]))
        PH.append((riverDataSet[8]))
        dissolved_o2.append((riverDataSet[10]))
        gauge_height.append((riverDataSet[12]))

for date in dates:
    year.append(int(date[0:4]))
    month.append(int(date[5:7]))
    day.append(int(date[8:]))
    
for time in times:
    hour.append(int(time[0:2]))
    minute.append(int(time[3:]))

for number in range(len(dates)):
    tempList = []
    tempList.append(year)
    tempList.append(month)
    tempList.append(day)
    tempList.append(hour)
    tempList.append(minute)
    tempList.append(discharge)
    tempList.append(temperature)
    tempList.append(PH)
    tempList.append(dissolved_o2)
    riverData.append(tempList)

riverData = np.array(riverData)

river_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

river_model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam())
river_model.fit(riverData, gauge_height, epochs=10)
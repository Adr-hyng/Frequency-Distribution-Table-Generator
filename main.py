import pandas as pd
import numpy as np
import math
from os import getcwd
import heapq

DECIMAL_PLACES = 3

df = pd.read_excel('Data.xlsx', sheet_name='Sheet1')

def getClassLimits(interval, size):
    df["LOWER LIMIT"][0] = math.floor(df["DATASET"][0])
    df["UPPER LIMIT"][0] = math.ceil((df["LOWER LIMIT"][0] + interval) - 1)
    
    for i in range(1, size):
        df["UPPER LIMIT"][i] = round(df["UPPER LIMIT"][i - 1] + interval, 0)
        df["LOWER LIMIT"][i] = round(df["LOWER LIMIT"][i - 1] + interval, 0)
        
        if (df["LOWER LIMIT"][i] >= round(df.DATASET.max(), 0)):
            df["LOWER LIMIT"][i] = np.nan
            df["UPPER LIMIT"][i] = np.nan
            break;
        
        
def getClassBoundary():
    dataset = df["LOWER LIMIT"].to_numpy()
    size = dataset[~pd.isnull(dataset)].size
    for i in range(0, size):
        df["LOWER BOUNDARY"][i] = df["LOWER LIMIT"][i] - 0.5
        df["UPPER BOUNDARY"][i] = df["UPPER LIMIT"][i] + 0.5
        

def getFrequency(dataset):
    total = []
    frequency_count = 0
    i = 0
    for (data) in dataset[~pd.isnull(dataset)]:
        if df["LOWER LIMIT"][i] <= round(data, 0) <= df["UPPER LIMIT"][i]:
            frequency_count += 1
        else:
            df["FREQUENCY"][i] = frequency_count
            total.append(frequency_count)
            frequency_count = 1
            i += 1
    total.append(frequency_count)
    df["FREQUENCY"][i] = frequency_count
    df["FREQUENCY"][i+1] = sum(total)
    return total


def getRelativeFrequency(total_occurence, frequency):
    RM = []
    for i, freq in enumerate(frequency):
        df["RELATIVE FREQUENCY"][i] = round(freq / total_occurence, DECIMAL_PLACES)
        RM.append(df["RELATIVE FREQUENCY"][i])
    df["RELATIVE FREQUENCY"][i+1] = sum(RM) # Add Total at last of row.
    return RM


def getPercentage(RM):
    for i, percentage in enumerate(RM):
        df["PERCENTAGE"][i] = f"{round(percentage * 100, DECIMAL_PLACES)} %" 
    df["PERCENTAGE"][i+1] = f"{round(sum(RM), DECIMAL_PLACES)} %" # Add Total at last of row.


def getCummulativeFrequency(frequency):
    df["<CM"][0] = df["FREQUENCY"][0]
    i = 0
    for i in range(1, len(frequency)):
        df["<CM"][i] = frequency[i] + df["<CM"][i-1]
    df[">CM"][0] = df["<CM"][i]
    i = 0
    for i in range(0, len(frequency)):
        df[">CM"][i+1] = df[">CM"][i] - frequency[i]
    df[">CM"][i+1] = "" # Delete Useless Last Row
        

def getMidpoint(frequency):
    midpoints = []
    for i in range(0, len(frequency)):
        df["MIDPOINTS"][i] = (df["LOWER LIMIT"][i] + df["UPPER LIMIT"][i]) / 2
        midpoints.append((df["LOWER LIMIT"][i] + df["UPPER LIMIT"][i]) / 2)
    df["MIDPOINTS"][i+1] = sum(midpoints) # Add Total at last of row.
    return midpoints


def getFx(frequency, midpoint):
    total = []
    for i, (fx, freq) in enumerate(zip(midpoint, frequency)):
        df["FX"][i] = fx * freq
        total.append(fx * freq)
    df["FX"][i+1] = sum(total) # Add Total at last of row.
    return total


def getXBar(fx, frequency):
    return sum(fx) / sum(frequency)


def getFxBar(frequency, midpoint, x_bar):
    df["XBAR"][0] = x_bar
    total_variance = 0
    for i in range(0, len(frequency)):
        df["FXBAR"][i] = round(frequency[i] * np.power((midpoint[i] - x_bar), 2), 2)
        total_variance += df["FXBAR"][i]
    df["FXBAR"][i+1] = total_variance # Add Total at last of row.
    return total_variance


def getMedian(frequency, interval):
    n = sum(frequency)
    targetIndex = 0
    mid = 0
    for i in range(0, len(frequency)):
        mid = n / 2
        if mid <= df["<CM"][i]:
            targetIndex = i
            break
    df[f"MEDIAN"][targetIndex] = round(df["LOWER BOUNDARY"][targetIndex] + ((mid - df["<CM"][targetIndex - 1]) / df["FREQUENCY"][targetIndex]) * interval, 2)
    print(f"MID: {mid} \n LB: {df['LOWER BOUNDARY'][targetIndex]} \n\n <CM: {df['<CM'][targetIndex - 1]}")
        

def getMode():
    pandas_mode = df["DATASET"].mode().to_string().split()[1]
    temp_set = {item:df["DATASET"].to_list().count(item) for item in df["DATASET"].to_list()}
    temp_modals = heapq.nlargest(3, temp_set, key=temp_set.get)
    mode_count = temp_set[max(temp_set, key=temp_set.get)] 
    modals = [temp_modals[i] for i in range(len(temp_modals)) if mode_count == temp_set[temp_modals[i]]]
    modal_kind = "Unimodal" if len(modals) == 1 else "Bimodal" if len(modals) == 2 else "Multimodal"
    mode = max(temp_set, key=temp_set.get)
    print(temp_modals)
    df["MODE"][0] = mode
    df["MODAL_TYPE"][0] = modal_kind


def getMidRange():
    df["MIDRANGE"][0] = (df.DATASET.max() + df.DATASET.min()) / 2
    

def getVariance(total_variance, total_class):
    return (total_variance / (total_class - 1))
    

def getQuartile(frequency, k: int, n: int, interval):
    targetIndex = 0
    kn = 0
    for i in range(0, len(frequency)):
        kn = (k * n) / 4
        if kn <= df["<CM"][i]:
            targetIndex = i
            break
    df[f"QUARTILE{k}"][targetIndex] = round(df["LOWER BOUNDARY"][targetIndex] + ((kn - df["<CM"][targetIndex - 1]) / df["FREQUENCY"][targetIndex]) * interval, 2)
    return df[f"QUARTILE{k}"][targetIndex]
    

def getDecile(frequency, k: int, n: int, interval):
    targetIndex = 0
    kn = 0
    for i in range(0, len(frequency)):
        kn = (k * n) / 10
        if kn <= df["<CM"][i]:
            targetIndex = i
            break
    df[f"DECILE"][targetIndex] = round(df["LOWER BOUNDARY"][targetIndex] + ((kn - df["<CM"][targetIndex - 1]) / df["FREQUENCY"][targetIndex]) * interval, 2)
    return df[f"DECILE"][targetIndex]
        

def getPercentile(frequency, k: int, n: int, interval):
    targetIndex = 0
    kn = 0
    for i in range(0, len(frequency)):
        kn = (k * n) / 100
        if kn <= df["<CM"][i]:
            targetIndex = i
            break
    df["PERCENTILE"][targetIndex] = round(df["LOWER BOUNDARY"][targetIndex] + ((kn - df["<CM"][targetIndex - 1]) / df["FREQUENCY"][targetIndex]) * interval, 2)
    return df["PERCENTILE"][targetIndex]


def getMidHinge(q1, q3):
    df["MIDHINGE"][0] = (q1 + q3) / 2
    

def getInterQuartile(q1, q3):
    df["INTER QUARTILE"][0] = (q3 - q1)
    

def getQuartileDeviation(q1, q3):
    df["QUARTILE DEVIATION"][0] = (q3 - q1) / 2

if __name__ == "__main__":
    ds_range = df["RANGE"][0] = (df.DATASET.max() - df.DATASET.min())
    f = df.DATASET.sum()
    dataset = df.DATASET.to_numpy()
    size = df["POPULATION"][0] = dataset[~pd.isnull(dataset)].size
    interval = df["INTERVAL"][0] = np.round((ds_range / (1 + 3.322 * np.log10(size))), decimals = 0)

    getClassLimits(interval, size) # Creates Table
    getClassBoundary() # Creates Table
    getFrequency = getFrequency(dataset) # Creates Table
    getRelativeFrequency = getRelativeFrequency(size, getFrequency) # Creates Table
    getPercentage(getRelativeFrequency) # Creates Table
    getCummulativeFrequency(getFrequency) # Creates Table
    midpoints = getMidpoint(getFrequency) # Creates Table
    fx = getFx(getFrequency, midpoints) # Creates Table
    fxbar = getFxBar(getFrequency, midpoints, getXBar(fx, getFrequency)) # Creates Table
    getMedian(getFrequency, interval) # In Table
    getMode() # In Table
    variance = df["VARIANCE"][0] = round(getVariance(fxbar, size), 2)
    standard_deviation = df["STANDARD DEVIATION"][0] = round(math.sqrt(variance), 2)
    getMidRange() # In Table
    q1 = getQuartile(getFrequency, 1, size, interval)
    q2 = getQuartile(getFrequency, 2, size, interval)
    q3 = getQuartile(getFrequency, 3, size, interval)
    getDecile(getFrequency, 3, size, interval)
    getPercentile(getFrequency, 96, size, interval)
    getMidHinge(q1, q3)
    getInterQuartile(q1, q3)
    getQuartileDeviation(q1, q3)

    df.to_excel(f"{getcwd()}/Datas.xlsx", index = False)
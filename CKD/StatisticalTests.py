import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import chisquare

# KS test

def compare(column_name):
    path = "Chronic_Kidney_Dsease_data.csv"
    df = pd.read_csv(path)

    zero = df.loc[df['Diagnosis'] == 0]
    one = df.loc[df['Diagnosis'] == 1]

    D0 = zero[[column_name]].to_numpy().reshape(-1)
    D1 = one[[column_name]].to_numpy().reshape(-1)

    k = ks_2samp(D0, D1)

    print(column_name,":", k)
    return [k.statistic, k.pvalue]

features = ["MuscleCramps", "Itching", "SleepQuality", "DietQuality", "GFR", "SerumCreatinine", "ProteinInUrine", "FastingBloodSugar", "CholesterolTriglycerides", "MedicationAdherence"]

com = []

for i in range(len(features)):
    com.append(compare(features[i]))


# Chi Squared test

path = "Chronic_Kidney_Dsease_data.csv"
df = pd.read_csv(path)

FHB_0D0 = df.loc[df['Diagnosis'] == 0]
FHB_0D0 = FHB_0D0[FHB_0D0["FamilyHistoryDiabetes"] == 0]
FHB_0D0 = FHB_0D0.shape[0]

FHB_0D1 = df.loc[df['Diagnosis'] == 1]
FHB_0D1 = FHB_0D1[FHB_0D1["FamilyHistoryDiabetes"] == 0]
FHB_0D1 = FHB_0D1.shape[0]

FHB_1D0 = df.loc[df['Diagnosis'] == 0]
FHB_1D0 = FHB_1D0[FHB_1D0["FamilyHistoryDiabetes"] == 1]
FHB_1D0 = FHB_1D0.shape[0]

FHB_1D1 = df.loc[df['Diagnosis'] == 1]
FHB_1D1 = FHB_1D1[FHB_1D1["FamilyHistoryDiabetes"] == 1]
FHB_1D1 = FHB_1D1.shape[0]

freqs = [FHB_0D0, FHB_0D1, FHB_1D0, FHB_1D1]
totals = np.array([FHB_0D0 + FHB_1D0, FHB_0D1 + FHB_1D1, FHB_0D0 + FHB_0D1, FHB_1D0 + FHB_1D1])
total = totals[0] + totals[1]

proportions = totals / total

expected = np.array([totals[0] * proportions[2], totals[1] * proportions[2], totals[0] * proportions[3], totals[1] * proportions[3]])
observed = np.array(freqs)

c = chisquare(f_obs=observed, f_exp=expected, ddof=1)

print(c)

features = np.array(["MuscleCramps", "Itching", "SleepQuality", "DietQuality", "GFR", "SerumCreatinine", "ProteinInUrine", "FastingBloodSugar", "CholesterolTriglycerides", "MedicationAdherence", "FamilyHistoryDiabetes"])
com.append([c.statistic, c.pvalue])

com = np.array(com)
com = np.c_[features,com]

df = pd.DataFrame(com, columns=['Features','Statistics', 'p-values'])
df['p-values'] = pd.to_numeric(df['p-values'])
df['p < 0.05'] = df['p-values'] < 0.05
df.to_csv('StatisticalTests.csv', index = False) 

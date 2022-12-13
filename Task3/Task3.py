import pandas as pd;
import numpy as np;
import re

def ReadFile():
    File_name = 'Energy Indicators.xls';
    energy = pd.read_excel(File_name, skiprows=16, usecols="C:F", nrows=228);
    energy.head();
    return energy;

def Convert(energy):
    gigajoules = energy['Energy Supply'];
    index = 0;

    for item in gigajoules:
        if(index != 0):

            energy.loc[index, 'Energy Supply'] = item * 1000000;
        index += 1;

    return energy;

def NaNData(energy, column):
    data = energy[column];
    index = 0;

    for item in data:
        if item == '...':
            data[index] = np.NAN;

        index += 1;
    return energy;


def Dictionary():
    dict = {
        "Republic of Korea":"South Korea",
        "United States of America":"United States",
        "United Kingdom of Great Britain and Northern Ireland":"United Kingdom",
        "China, Hong Kong Special Administrative Region":"Hong Kong"
    }

    return dict;

def Rename(data, dict, column):
    index = 0;

    for item in data[column]:
        if item in dict.keys():

            data.loc[index, column] = dict[item];

        index += 1;

    return data;

def DeleteNumbers(energy):
    index = 0;

    for item in energy['Country Name']:
        if type(item) != float:

            newName = ''.join([i for i in item if not i.isdigit()]);
            energy.loc[index, 'Country Name'] = newName;

        index += 1;

    return energy;

def PrintALLColums(energy):
    colums = energy.columns;
    print(colums);

    for j in range(4):
        for i in energy[colums[j]]:
            print(i);

    return energy;

def RemoveBrackets(energy):
    index = 0;

    for name in energy['Country Name']:
        if type(name) != float:

            name = re.sub("\(.*?\)", "", name)
            energy.loc[index, "Country Name"] = name

        index += 1

    return energy


def Start():
    energy = ReadFile();

    #print(energy.columns);

    energy = NaNData(energy, 'Energy Supply');
    energy = NaNData(energy, 'Energy Supply per capita');
    energy = Convert(energy);
    energy = DeleteNumbers(energy);

    dict = Dictionary();
    energy = Rename(energy, dict, 'Country Name');
    energy = RemoveBrackets(energy)
    return energy;

#Start();
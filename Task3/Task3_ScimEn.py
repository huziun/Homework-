import pandas as pd;
import Task3;
import Task3_GDP;

def ScimEn_file():
    file_name = 'scimagojr country rank 1996-2021.xlsx';
    ScimEn = pd.read_excel(file_name);
    ScimEn.head();

    return ScimEn;

def Merge(file_three):
    file_one = Task3.Start();
    file_two = Task3_GDP.Start();
    mergeFirst = pd.merge(file_three, file_one, on='Country Name');
    margeSecond = pd.merge(mergeFirst, file_two);
    del margeSecond['Region'];

    return margeSecond.set_index('Country Name').head(15);

def Start():

    ScimEn = ScimEn_file();
    Merge(ScimEn);

Start();


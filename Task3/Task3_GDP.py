import pandas as pd;
import  Task3;

def GDP_file():
    file_name = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4701247.csv";
    GDP = pd.read_csv(file_name, sep=',', skiprows=4);

    return GDP;

def Dictionary():

    dict = {
    "Korea, Rep.": "South Korea",
    "Iran, Islamic Rep.": "Iran",
    "Hong Kong SAR, China": "Hong Kong"
    }

    return dict;

def Rename(data, dict, column):
    data = Task3.Rename(data, dict, column);
    return data;

def DeleteColumns(data):
    del data['Indicator Name'];
    del data['Country Code'];
    del data['Indicator Code'];

    for i in range(1960, 2023):
        year = i;
        if year < 2006 or year > 2015:

            del data[str(year)];


    return data;

def Start():
    GDP = GDP_file();
    dict = Dictionary();
    #print(GDP.columns);
    GDP = Rename(GDP, dict, 'Country Name');
    GDP = DeleteColumns(GDP);
    return GDP;

Start();
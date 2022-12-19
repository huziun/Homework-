import Task3_ScimEn;
import pandas as pd;
import numpy as np;
import Create_Columns;

def answer_one():
    Frame = Task3_ScimEn.GetMerge();

    return Frame;

def answer_two():
    Frame = answer_one();
    avgGDP = pd.Series();

    for item in range(len(Frame)):
        country = Frame.iloc[item].name;
        country_data = Frame.iloc[item, 10:];
        average = np.average(country_data);
        avgGDP = avgGDP.append(pd.Series({country:average}));

    avgGDP = avgGDP.sort_values(ascending=False);
    #print(avgGDP);
    return avgGDP;

def answer_three():
    Frame = answer_one();
    data_series = answer_two();
    country_six = data_series[5]; #index start 0
    country_sixName = data_series[country_six == data_series].index[0];
    diff = Frame.loc[country_sixName]["2015"] - Frame.loc[country_sixName]["2006"];
    return diff;

def answer_four():
    Frame = answer_one();
    Frame = Create_Columns.NewColumn(Frame, 'Self-citations', 'Citations', 'ratio');
    Frame = Frame.sort_values(by='ratio', ascending=False);
    return Frame.iloc[0].name, Frame.iloc[0]['ratio'];

class PFrame:
    def __init__(self):
        self.pframe = []
pframe = PFrame()

def answer_five():
    Frame = answer_one();
    #Frame = Create_Columns.NewColumn(Frame, 'Energy Supply', 'Energy Supply per capita', 'population');
    Frame = Create_Columns.Five_newColumn(Frame);
    Frame = Frame.sort_values(by='population', ascending=False);
    pframe.pframe = Frame['population']
    return Frame.iloc[2].name;

def answer_six():
    Frame = answer_one();
    Frame = Create_Columns.NewColumn(Frame, 'Citable documents', 'population', 'citable documents per person');

    correlate_df = pd.DataFrame({
        "Energy Supply per capita":Frame['Energy Supply per capita'],
        "citable documents per person":Frame['citable documents per person']
    })

    #print(correlate_df.corr(method='pearson'));
    return correlate_df.corr(method='pearson')['citable documents per person'][0];

def answer_seven():
    ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}

    Frame = answer_five();
    #Frame = Create_Columns.NewColumn('','','')

    Frame = pd.DataFrame(pframe.pframe)

    Continent = []
    frame = []
    for i in range(len(Frame)):
        name = Frame.iloc[i].name
        frame.append(ContinentDict[name])
        frame.append(name)
        frame.append(Frame.iloc[i]["population"])
        Continent.append(frame)
        frame = []


    df = pd.DataFrame(Continent, columns=["Continent", "Country Name", "Populations"])
    size = df.groupby("Continent").count()
    size.rename(columns={"Country Name": "size"}, inplace=True)
    del size['Populations']

    sum = df.groupby("Continent").sum("Populations")
    mean = df.groupby("Continent").mean("Populations")
    std = [
        ['Asia', df.query("Continent == 'Asia'").std()[0]],
        ['Europe', df.query("Continent == 'Europe'").std()[0]],
        ['Australia', df.query("Continent == 'Australia'").std()[0]],
        ['North America', df.query("Continent == 'North America'").std()[0]],
        ['South America', df.query("Continent == 'South America'").std()[0]]
    ]
    std = pd.DataFrame(std, columns=["Continent", "std"])

    def merge( df, df2, newname):
        dataframe = pd.merge(df, df2, on="Continent")
        dataframe.rename(columns={"Populations": newname}, inplace=True)
        return dataframe

    df = merge(size, sum, "sum");
    df = merge(df, mean, "mean")
    df = merge(df, std, "std")
    return df.set_index("Continent")

def Start():
    print("1")
    print(answer_one())
    print("2")
    print(answer_two());
    print("3")
    print(answer_three());
    print("4")
    print(answer_four());
    print("5")
    print(answer_five());
    print("6")
    print(answer_six());
    print("7")
    print(answer_seven())

    return 0;

Start()
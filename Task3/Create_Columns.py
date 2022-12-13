import pandas as pd;

def Five_newColumn(Frame):
    Frame = NewColumn(Frame, 'Energy Supply', 'Energy Supply per capita', 'population');
    return Frame;

def NewColumn(Frame,column1, column2, newcolumn):

    if column2 == 'population':
        Frame = Five_newColumn(Frame);

    new_column = []
    for item in range(len(Frame[column1])):
        new_column.append(Frame.iloc[item][column1] / Frame.iloc[item][column2]);
       # print("Country name: ", Frame.iloc[item].name, Frame.iloc[item][column1], Frame.iloc[item][column2])

    Frame[newcolumn] = new_column;
    return Frame;

def Seven_column(Frame, dict):
    Country = []
    Frame = Frame['populations']
    print(Frame.head(15))
    return Frame;

def NewDF(Frame):

    df = pd.DataFrame({
        #"Country":Frame.iloc.name,
        "Continent":Frame['Continent']
    })

    print(df);
    return 0;
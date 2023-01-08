import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

def GetData():
    data = pd.read_csv('Groceries_dataset.csv')
    print(data)
    #print(data.columns)
    return data

def VisualData(df):
    dist = [len(g) for k, g in df.groupby(['Member_number', 'Date'])]
    plt.hist(dist)
    plt.show();

def GetProductList(df):
    dataframe = df
    dataframe = dataframe.to_frame()
    dataframe = dataframe['itemDescription'].tolist()
    return dataframe

def print_rules(rules):
    count = 0;
    for rule in rules:

        print('rule.items=', list(rule.items))
        print('rule.support=', rule.support)

        for os in rule.ordered_statistics:
            print('\titems_base=', list(os.items_base))
            print('\tlifted_item =', list(os.items_add))
            print('\tlift=', os.lift)
            print('\tconfidence (i.e. cond prob {} if {})='.format(list(os.items_add), list(os.items_base)),
                    os.confidence)
            print('\n')
        count += 1
        if count == 10:
            break

def rules_to_df(rules):
    rules_df = pd.DataFrame()
    for rule in rules:
        support = rule.support
        for os in rule.ordered_statistics:
            item_base = os.items_base
            item_add = os.items_add
            lift = os.lift
            confidence = os.confidence
        rules_df = rules_df.append(pd.DataFrame([[support,item_base,item_add,lift,confidence]], columns = ["Support","Item Base", "Item Add","Lift","Confidence"]))
    return rules_df

df = GetData()
VisualData(df)

df = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)
products_list = GetProductList(df)
print(products_list)

rules = list(apriori(products_list, min_support = 0.001, min_confidence = 0.1, min_lift = 1.1, min_length = 2))
print_rules(rules)
print(f"Len rules: {len(rules)}")
rules_df = rules_to_df(rules)
rules_df.sort_values(by = 'Lift', ascending=False, inplace=True)
print(rules_df.head(10))
rules_df.to_excel('rules.xls')


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

groceries_df = pd.read_csv('groceries - groceries.csv',encoding='latin-1')

groceries_df.head()
groceries_df = groceries_df.drop(['Item(s)'], axis = 1)
num_records = len(groceries_df)
print(num_records)

# Data preprocessing
records =[]
for i in range(0,num_records):
    records.append([str(groceries_df.values[i,j]) for j in range(0, 32)])

association_rules = apriori (records,min_support=0.0045, min_confidence = 0.20, min_lift = 3, min_length =2)
association_results = list(association_rules)

print(association_results[0])

# Display result
results = []
for item in association_results:
    pair = item[0]
    items = [x for x in pair if x != 'nan']
    
    value0 = str(items[0])
    value1 = str(items[1])
    
    value2 = str(item[1])[:7]
    
    value3 = str(item[2][0][2])[:7]
    value4 = str(item[2][0][3])[:7]
    
    rows = (value0,value1,value2,value3,value4)
    results.append(rows)

labels = ['Item 1','Item 2','Support','Confidence','Lift']
item_suggestion = pd.DataFrame.from_records(results,columns = labels)
pd.set_option('display.max_rows', item_suggestion.shape[0]+1)
item_suggestion = item_suggestion.drop_duplicates(subset=['Item 1', 'Item 2'], keep='last')
item_suggestion.style.hide_index()
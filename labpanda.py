import pandas as pd
import numpy as np
df = pd.read_csv("iris.csv")
#task 1
print(df.shape)
print(df.isnull().sum())

numeric_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

#print(df)

if df['Species'].isnull().sum() > 0:
    most_frequent_species = df['Species'].mode()[0]
    df['Species'].fillna(most_frequent_species, inplace=True)
#print(df)

#task 2
print(df.duplicated().sum())
df.drop_duplicates()
df['petal_area'] = df['PetalLengthCm'] * df['PetalWidthCm']
df.dropna(inplace=True)
print(df)

#task3
df['Species']=pd.Categorical(df['Species']).codes
print(df.head())
agg_df=df.groupby('Species')[['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm' , 'SepalWidthCm']].mean()
print(agg_df)



#task4
df1=df.melt(id_vars=['Id'],value_vars=['Species', 'PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm' , 'SepalWidthCm'],
                      var_name='measurement_type', value_name='measurement_value')
#print(df1)







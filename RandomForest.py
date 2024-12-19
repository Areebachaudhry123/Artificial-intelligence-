import pandas as pd
import math
import random

def calculate_entropy(data,target_col):
    class_counts=data[target_col].value_counts()
    total_instances=len(data)
    entropy=0
    for count in class_counts:
        prob=count/total_instances
        entropy-=prob*math.log2(prob)
    return entropy

def calculate_information_gain(data,attribute,target_col):
    dataset_entropy=calculate_entropy(data,target_col)
    attribute_values=data[attribute].unique()
    weighted_entropy=0
    for value in attribute_values:
        subset=data[data[attribute]==value]
        subset_entropy=calculate_entropy(subset,target_col)
        weighted_entropy+=(len(subset)/len(data))*subset_entropy
    information_gain=dataset_entropy-weighted_entropy
    return information_gain

def build_tree(data,attributes,target_col):
    if len(data[target_col].unique())==1:
        return data[target_col].iloc[0]
    if len(attributes)==0:
        return data[target_col].mode()[0]
    gains={attribute:calculate_information_gain(data,attribute,target_col) for attribute in attributes}
    best_attribute=max(gains,key=gains.get)
    tree={best_attribute:{}}
    for value in data[best_attribute].unique():
        subset=data[data[best_attribute]==value]
        subtree=build_tree(subset,[attr for attr in attributes if attr!=best_attribute],target_col)
        tree[best_attribute][value]=subtree
    return tree

def predict(tree,data_point):
    for attribute,sub_tree in tree.items():
        value=data_point[attribute]
        if value in sub_tree:
            subtree=sub_tree[value]
            if isinstance(subtree,str):
                return subtree
            else:
                return predict(subtree,data_point)

def build_random_forest(data,attributes,target_col,n_trees=2):
    trees=[]
    for _ in range(n_trees):
        subset=data.sample(frac=0.7,replace=True)
        random_attributes=random.sample(attributes,k=random.randint(1,len(attributes)))
        tree=build_tree(subset,random_attributes,target_col)
        trees.append(tree)
    return trees

def random_forest_predict(forest,data_point):
    predictions=[predict(tree,data_point) for tree in forest]
    return max(set(predictions),key=predictions.count)

data=pd.DataFrame({'Weather':['Sunny','Sunny','Overcast','Rainy','Sunny','Sunny','Rainy','Rainy','Overcast','Overcast','Sunny','Sunny','Rainy','Overcast'],'Temperature':['Hot','Hot','Hot','Mild','Mild','Cool','Cool','Mild','Cool','Mild','Mild','Hot','Hot','Hot'],'Play?':['No','No','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','No','Yes','No','Yes']})

target_col='Play?'
attributes=['Weather','Temperature']
forest=build_random_forest(data,attributes,target_col,n_trees=2)

for i,tree in enumerate(forest):
    print(f"Decision Tree {i+1}: {tree}")

test_data_point={'Weather':'Sunny','Temperature':'Cool'}
prediction=random_forest_predict(forest,test_data_point)
print(f"Predicted class for the data point {test_data_point}: {prediction}")

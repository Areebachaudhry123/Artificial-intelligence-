import pandas as pd
import math

def calculate_entropy(data, target_col):
    class_counts = data[target_col].value_counts()
    total_instances = len(data)
    entropy = 0
    for count in class_counts:
        prob = count / total_instances
        entropy -= prob * math.log2(prob)
    return entropy

def calculate_information_gain(data, attribute, target_col):
    dataset_entropy = calculate_entropy(data, target_col)
    attribute_values = data[attribute].unique()
    weighted_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value] 
        subset_entropy = calculate_entropy(subset, target_col)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    information_gain = dataset_entropy - weighted_entropy 
    return information_gain

def build_tree(data, attributes, target_col):
    if len(data[target_col].unique()) == 1: 
        return data[target_col].iloc[0]
    if len(attributes) == 0:  
        return data[target_col].mode()[0]
    gains = {}
    for attribute in attributes:
        gain = calculate_information_gain(data, attribute, target_col)
        gains[attribute] = gain
    best_attribute = max(gains, key=gains.get)
    tree = {best_attribute: {}}
    attribute_values = data[best_attribute].unique() 
    for value in attribute_values:
        subset = data[data[best_attribute] == value]
        subtree = build_tree(subset, [attr for attr in attributes if attr != best_attribute], target_col)  #excluding the used attribute 
        tree[best_attribute][value] = subtree
    return tree

def predict(tree, data_point):
    for attribute, sub_tree in tree.items():
        attribute_value = data_point[attribute]
        if attribute_value in sub_tree:
            subtree = sub_tree[attribute_value]
            if isinstance(subtree, str):  
                return subtree
            else:
                return predict(subtree, data_point)


data = pd.DataFrame({
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Rainy', 'Overcast', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Mild', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Hot', 'Hot', 'Hot'],
    'Play?': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
})
target_col = 'Play?'
attributes = ['Weather', 'Temperature']
tree = build_tree(data, attributes, target_col)
print("Decision Tree: ", tree)
data_point = {'Weather': 'Sunny', 'Temperature': 'Cool'}
prediction = predict(tree, data_point)
print(f"Predicted class for the data point {data_point}: {prediction}")

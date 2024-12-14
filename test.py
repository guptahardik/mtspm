#!/bin/python3

import math
import os
import random
import re
import sys
import pandas as pd


#
# Complete the 'valuation' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. LONG_INTEGER reqArea
#  2. LONG_INTEGER_ARRAY area
#  3. LONG_INTEGER_ARRAY price
#

def valuation(reqArea, area, price):
    
   
    data = {
        'Area': area,
        'Price': price
    }
    
    df = pd.DataFrame(data)
    
    # Create compList
    comp_list = []
    for i in range(len(df)):
        same_area_prices = df[(df['Area'] == df.loc[i, 'Area']) & (df.index != i)]['Price']
        comp_list.append(list(same_area_prices))
    
    df['compList'] = comp_list

    # Calculate Mean, StdDev, AbsDiff, and 3xStdDev
    means = []
    std_devs = []
    abs_diff = []
    three_std_devs = []
    
    for i in range(len(df)):
        clist = df.loc[i, 'compList']
    
        if clist:
            mean_val = sum(clist) / len(clist)
            std_dev_val = (sum((x - mean_val) ** 2 for x in clist) / len(clist)) ** 0.5
        else:
            mean_val = float('nan')
            std_dev_val = float('nan')
    
        means.append(mean_val)
        std_devs.append(std_dev_val)
    
        # Calculate |Price[i] - P[m]|
        abs_diff.append(abs(df.loc[i, 'Price'] - mean_val) if clist else float('nan'))
    
        # Calculate 3 * standard deviation
        three_std_devs.append(3 * std_dev_val if clist else float('nan'))

    # Add new columns to DataFrame
    df['Mean'] = means
    df['StdDev'] = std_devs
    df['AbsDiff'] = abs_diff
    df['3xStdDev'] = three_std_devs

    # Determine if an entry is an outlier
    outliers = []
    for i in range(len(df)):
        clist = df.loc[i, 'compList']
        if not clist:
            # If compList is empty, directly set outlier to False
            outliers.append(False)
        else:
            # Check if |Price[i] - P[m]| > 3 * StdDev
            mean_val = df.loc[i, 'Mean']
            std_dev_val = df.loc[i, 'StdDev']
            price_diff = abs(df.loc[i, 'Price'] - mean_val)
            
            if price_diff > 3 * std_dev_val:
                outliers.append(True)
            else:
                outliers.append(False)

    # Add the new 'Is Outlier?' column to DataFrame
    df['Is Outlier?'] = outliers
    
    print(df)
    
    new_df = df[df['Is Outlier?'] == False][['Area', 'Price']].copy()
    #new_df.reset_index(drop=True, inplace=True)
    
    print(new_df)
    
    
    return calculate_price(new_df, reqArea)
    
    
def calculate_price(df, reqArea):
    
    if df.empty:
        return max(1000, min(int(1000*reqArea), 1000000))
        
    if len(df) == 1:
        return max(1000, min(int(df['Price'].mean()),1000000))
        
        
    same_area_houses = df[df['Area'] == reqArea]
    
    
    if not same_area_houses.empty:
        price = int(same_area_houses['Price'].mean())  # Mean price for same area
        return max(1000, min(price, 1000000))  # Apply bounds
        
    if df['Area'].min() <= reqArea <= df['Area'].max():
        
        print(reqArea)
        print("hello")
        
        unique_areas = df['Area'].drop_duplicates().sort_values()
        
        # Find the closest lower unique Area
        lower_areas = unique_areas[unique_areas < reqArea]
        closest_lower_area = lower_areas.max() if not lower_areas.empty else None
        print(closest_lower_area)
        lower_price = df[df['Area'] == closest_lower_area]['Price'].mean() if closest_lower_area is not None else None
        print(lower_price)
        
        # Find the closest higher unique Area
        higher_areas = unique_areas[unique_areas > reqArea]
        closest_higher_area = higher_areas.min() if not higher_areas.empty else None
        print(closest_higher_area)
        upper_price = df[df['Area'] == closest_higher_area]['Price'].mean() if closest_higher_area is not None else None
        print(upper_price)
        
        if lower_price is not None and upper_price is not None:
            # Interpolate the price based on the relative position of reqArea
            price = lower_price + (upper_price - lower_price) * ((reqArea - closest_lower_area) / (closest_higher_area - closest_lower_area))
            return max(1000, min(int(round(price)), 1000000))
    else:
        sorted_unique_areas = df['Area'].unique()
        
        result = {}
        
        if df['Area'].min() < reqArea:
            sorted_unique_areas = sorted(sorted_unique_areas, reverse=True)
        else:
            print("this case")
            sorted_unique_areas = sorted(sorted_unique_areas, reverse=False)
            
        if len(sorted_unique_areas) >= 2:
                max_areas = sorted_unique_areas[:2]  # Two max unique areas
                result = df[df['Area'].isin(max_areas)].groupby('Area', as_index=False).agg({'Price': 'mean'})
        else:
            if len(sorted_unique_areas) == 1:
                price = df[df['Area'] == unique_area]['Price'].mean()
                return max(1000, min(round(price), 1000000))
        area1 = result['Area'].iloc[0]
        mean_price1 = result['Price'].iloc[0]
        area2 = result['Area'].iloc[1]
        mean_price2 = result['Price'].iloc[1]


        extrapolated_price = (mean_price1 + ((mean_price2 - mean_price1) / (area2 - area1)) * (reqArea - area1))
        return max(1000, min(int(round(extrapolated_price)), 1000000))
    

    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    reqArea = int(input().strip())

    area_count = int(input().strip())

    area = []

    for _ in range(area_count):
        area_item = int(input().strip())
        area.append(area_item)

    price_count = int(input().strip())

    price = []

    for _ in range(price_count):
        price_item = int(input().strip())
        price.append(price_item)

    result = valuation(reqArea, area, price)

    fptr.write(str(result) + '\n')

    fptr.close()



#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'maximum_path' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY node_values as parameter.
#

def maximum_path(node_values):
    
    arr = node_values
    
    pyramid = []
    index = 0
    n = 0
    
    # Determine number of layers in the pyramid
    while index < len(arr):
        n += 1
        index += n
    index = 0
    
    # Construct the pyramid
    for i in range(n):
        layer = arr[index:index + i + 1]
        pyramid.append(layer)
        index += i + 1
    node_values = pyramid
    
     # Start from the second last row and move upwards
    for i in range(len(node_values) - 2, -1, -1):
        for j in range(len(node_values[i])):
            # Update each element to be itself plus the maximum of its two children
            node_values[i][j] += max(node_values[i + 1][j], node_values[i + 1][j + 1])

    # The top element now contains the maximum path sum
    return node_values[0][0]
    
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    node_values_count = int(input().strip())

    node_values = []

    for _ in range(node_values_count):
        node_values_item = int(input().strip())
        node_values.append(node_values_item)

    result = maximum_path(node_values)

    fptr.write(str(result) + '\n')

    fptr.close()




#!/bin/python3
import math
import os
import random
import re
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor





#
# Complete the 'calcMissing' function below.
#
# The function accepts STRING_ARRAY readings as parameter.
#

def calcMissing(readings):
    # Write your code her
    
    timestamps = []
    values = []
    for text in readings:
        row = text.split("\t")
        timestamps.append(row[0])
        values.append(row[1])
        
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'value': values
    })
    
    
    n_neighbors = 5
    # Convert 'value' column to numeric, and keep missing values as NaN
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Create a column to identify the rows with missing data
    df['Missing'] = df['value'].isna()

    # Prepare the data for KNN
    # We will use the index (time step) as a feature and 'value' as the target
    df['Index'] = np.arange(len(df))

    # Split the data into known (non-missing) and unknown (missing) sets
    known_data = df[~df['Missing']]
    missing_data = df[df['Missing']]

    # Features: Index (or timestamp could be used as well)
    X_known = known_data[['Index']]
    y_known = known_data['value']

    X_missing = missing_data[['Index']]

    # Fit a KNN model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_known, y_known)

    # Predict the missing values
    predictions = knn.predict(X_missing)

    # Fill the missing values with the predicted values
    df.loc[df['Missing'], 'value'] = predictions
    
    predictions, missing_indices = df['value'], df['Missing']
    
    for value in predictions[missing_indices].values:
        print(value)
    
    
        
    

if __name__ == '__main__':
    readings_count = int(input().strip())

    readings = []

    for _ in range(readings_count):
        readings_item = input()
        readings.append(readings_item)

    calcMissing(readings)

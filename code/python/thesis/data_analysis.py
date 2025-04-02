import matplotlib.pyplot as plt
import numpy as np
#lib to read in csv files 
import pandas as pd

def main(): 
    #read in data from ./data/fresh-leaf-spectra-to-estimate-lma-over-neon-domains-in-eastern-united-states.csv
    path = './data/fresh-leaf-spectra-to-estimate-lma-over-neon-domains-in-eastern-united-states.csv'
    data = pd.read_csv(path)
    #print(data.to_string()) 

    #find index of USDA Symbol header in data
    #print(data.columns.to_list().index('USDA Symbol')) 
    data = data.dropna() 

    #for every row in data, add new USDA Symbol to a dic and count them if they aleard exist
    USDA_Symbol = {}
    for index, row in data.iterrows(): 
        symbol = row['USDA Symbol']
        if symbol in USDA_Symbol:
            USDA_Symbol[symbol] += 1
        else: 
            USDA_Symbol[symbol] = 1
    
    #sort the USDA_Symbol dic by value
    USDA_Symbol = dict(sorted(USDA_Symbol.items(), key=lambda item: item[1], reverse=True))
        #print the USDA_Symbol dic
    #print(USDA_Symbol)

    #if USDA_Symbol value is > 100 add to a new dic exclude nan
    USDA_Symbol_100 = {}
    #make list
    USDA_Symbol_for_analysis = []
    for key, value in USDA_Symbol.items():
        if value > 100:
            USDA_Symbol_100[key] = value 
            #store key in a list
            USDA_Symbol_for_analysis.append(key)

    #print the USDA_Symbol_100 dic
    print(USDA_Symbol_100, "\n",  USDA_Symbol_for_analysis)

    #extract data from the original data frame where USDA Symbol is in USDA_Symbol_for_analysis
    data = data[data['USDA Symbol'].isin(USDA_Symbol_for_analysis)]
    #make a csv file of the data
    path = './data/HS_data_for_analysis.csv'
    data.to_csv(path, index=False)  
    data = pd.read_csv(path)

    # Ensure xpoints and ypoints are 1D arrays for plotting
    xpoints = data.iloc[0, 11:].values 
    #print(xpoints)  
    #convert from np.float to float
    xpoints = np.array(xpoints, dtype=float)
    print(xpoints)

    #the y values are in the header of the data frame
    #ypoints = data.iloc[0, 11:].values 
    ypoints = list(data)
    #remove the first 10 rows
    ypoints = ypoints[11:]
    print(ypoints)

    # Plot the data
    plt.plot(xpoints, ypoints) 
    plt.show()

if __name__ == "__main__":
    main()
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
    #sort data by USDA Symbol
    data = data.sort_values(by=['USDA Symbol'])

   #the x values are in the header of the data frame 
    xpoints = list(data)
    xpoints = xpoints[11:] 

    # Initialize variables for plotting
    last_symbol = None
    plt.figure()

    for index, row in data.iterrows():
        symbol = row['USDA Symbol']
        # Ensure xpoints and ypoints are 1D arrays for plotting
        ypoints = row.iloc[11:].values
        ypoints = np.array(ypoints, dtype=float)

        if symbol != last_symbol and last_symbol is not None:
            # Show the current figure and start a new one
            plt.title(f"USDA Symbol: {last_symbol}")
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            #reduce the number of ticks on the x axis
            plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8)
            plt.show()
            plt.figure()
            #save plot as a png file
            plt.savefig(f'./data/{last_symbol}.png')

        # Plot the data
        plt.plot(xpoints, ypoints, label=f"Sample {index}")
        last_symbol = symbol

    # Show the last figure
    if last_symbol is not None:
        plt.title(f"USDA Symbol: {last_symbol}")
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8) 
        plt.show()
        plt.savefig(f'./data/{last_symbol}.png') 



"""
    

    for row in data.iterrows():
        # Ensure xpoints and ypoints are 1D arrays for plotting
        ypoints = data.iloc[row, 11:].values 
        #convert from np.float to float
        ypoints = np.array(ypoints, dtype=float)
        #print(xpoints)
        # Plot the data
    
    plt.plot(xpoints, ypoints)  

  
    #print(ypoints)

    
    #reduce the number of x ticks to 10
    plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8)

    #set the x and y axis labels    
    plt.xlabel('Wavelength', fontsize=10)
    plt.ylabel('Reflectance', fontsize=10)
    plt.show() """

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
#lib to read in csv files 
import pandas as pd

def token_sample_extract(token_Symbol_dic): 
    token_Symbol_100 = {}
    token_Symbol_for_analysis = []

    #if token_Symbol value is > 100 add to a new dic 
        #ensure that the number of samples is > 100 
    for key, value in token_Symbol_dic.items(): 
        if value > 100:
            token_Symbol_100[key] = value 
            #store key in a list 
            token_Symbol_for_analysis.append(key)
    return token_Symbol_100, token_Symbol_for_analysis

def count_samples_by_symbol(data, sort_term = 'USDA Symbol'):
   
    Symbol_count_dic = {}
    for index, row in data.iterrows():  
        symbol = row[sort_term]
        if symbol in Symbol_count_dic:
            Symbol_count_dic[symbol] += 1
        else:  
            Symbol_count_dic[symbol] = 1 
    return Symbol_count_dic

def return_class_lables(data, sort_term = 'USDA Symbol'):
    token_Symbol_100, token_Symbol_for_analysis = token_sample_extract(count_samples_by_symbol(data)) 
    return token_Symbol_for_analysis


def extract_data(sample_token = 'USDA Symbol', 
                path = './data/fresh-leaf-spectra-to-estimate-lma-over-neon-domains-in-eastern-united-states.csv',  
                final_path='./data/HS_data_for_analysis.csv'):
     
    #read in data from path
    data = pd.read_csv(path)
    #print(data.to_string()) 

    #remove nan values from the data frame 
    data = data.dropna() 

    #for every row in data, add new USDA Symbol to a dic and count them if they aleady exist
    token_Symbol = count_samples_by_symbol(data, sample_token)
    print(token_Symbol)
    #print the length of token_Symbol
    print(len(token_Symbol)) 
    
    #sort the token_Symbol dic by value
    token_Symbol = dict(sorted(token_Symbol.items(), key=lambda item: item[1], reverse=True))
    #print(token_Symbol)

    #extrat the token with the most samples, e.g >100
    token_Symbol_100, token_Symbol_for_analysis  = token_sample_extract(token_Symbol)
   
    #print the token_Symbol_100 dic
    print(token_Symbol_100, "\n",  token_Symbol_for_analysis) 

    #extract data from the original data frame where USDA Symbol is in token_Symbol_for_analysis  
    #excluding data with less than 100 samples
    data = data[data[sample_token].isin(token_Symbol_for_analysis)]
    
    #make a csv file of the data
    path = final_path
    data = data.sort_values(by=[sample_token]) 
    #remove all data colums that are not sample_token or a int 
    data = data.loc[:, data.columns.isin([sample_token]) | data.columns.str.isnumeric()]

    data.to_csv(path, index=False)  
    return data, token_Symbol_for_analysis

def main(): 
    extract_data() 
    



"""
    path = './data/HS_data_for_analysis.csv' 
    #data = pd.read_csv(path)
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
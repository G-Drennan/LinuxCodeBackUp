import matplotlib.pyplot as plt
import numpy as np
#lib to read in csv files 
import pandas as pd
from data_extraction import extract_data

def count_samples_by_symbol(data, sort_term):
   
    Symbol_count = {}
    for index, row in data.iterrows(): 
        symbol = row[sort_term]
        if symbol in Symbol_count:
            Symbol_count[symbol] += 1
        else:  
            Symbol_count[symbol] = 1
    return Symbol_count

def extract_xpoints(data, data_start=11):
    xpoints = list(data)
    xpoints = xpoints[data_start:] 
    return xpoints 

#usage:
#for index, row in data.iterrows():
    #    ypoints = extract_ypoints(row)
def extract_ypoints(data, toggle_float_conversion, data_start=11): 
    ypoints = data.iloc[data_start:].values
    # convert from np.float to float
    if toggle_float_conversion is True:
        ypoints = np.array(ypoints, dtype=float)
    return ypoints

def plot_wavelength(data, sort_term, data_start=11):
    
   #the x values are in the header of the data frame 
    xpoints = extract_xpoints(data,data_start) 

    #for every row in data, add new USDA Symbol to a dic and count them if they aleard exist

    Symbol_sample_count = count_samples_by_symbol(data,sort_term) 

    # Initialize variables for plotting
    last_symbol = None
    plt.figure()

    for index, row in data.iterrows(): 
        symbol = row[sort_term] 
        
        ypoints = extract_ypoints(row, True, data_start)

        if symbol != last_symbol and last_symbol is not None:
            # Show the current figure and start a new one
            plt.title(f"{sort_term}: {last_symbol}")
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            #reduce the number of ticks on the x axis
            plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8)
            
          
            # plt.text(0.5, 0.5, f"Samples: {Symbol_sample_count[last_symbol]}", fontsize=12, ha='left', va='top') 
              # display Symbol_sample_count(last_symbol) on the figure top left
            plt.text(0.75, 0.75, f"Samples: {Symbol_sample_count[last_symbol]}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
            plt.show()
            plt.figure() 
            #save plot as a png file
            plt.savefig(f'./data/{last_symbol}.png')
            
        # Plot the data
        plt.plot(xpoints, ypoints, label=f"Sample {index}")
        last_symbol = symbol

    # Show the last figure
    if last_symbol is not None:
        plt.title(f"{sort_term}: {last_symbol}")
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8) 
        plt.show()
        plt.savefig(f'./data/{last_symbol}.png') 

def main():
    extract_data() 
    path = './data/HS_data_for_analysis.csv' 
    data = pd.read_csv(path) 
    #sort data by USDA Symbol
    data = data.sort_values(by=['USDA Symbol'])
    plot_wavelength(data,'USDA Symbol')



if __name__ == "__main__":
    main() 
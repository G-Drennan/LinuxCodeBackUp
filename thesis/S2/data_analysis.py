#TODO: Extract data from the csv's in the data folder. Stort by Conditions and Genotype. 
#CSV files: data/maize_2018_2019_unl_metadata.csv, data/maize_2018_2019_unl_spectra.csv, data/maize_2018_2019_unl_traits.csv

#TODO: perform wavelength reflectance plot, PCA, correlation analysis

#TODO: sort into test and training data sets 40% - 60% split. Keep consistance for mulit differnt combinations of x and y data. 
    #   where Y data is always the given VI found in data/maize_2018_2019_unl_traits.csv

#TODO: dict = {[ID]: traits = [(header, value)...], wavelenght relfectance [(wavelenght/header, reflectance value)], Genotype}

#class to hold data e.g dict
#import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt

class Data:
    def __init__(self, filenames): 
        self.filenames = filenames
        self.dataDict = {}
        self.firstCsv = True
        self.path = 'data/combined_data.csv'
        self.combined_df = pd.DataFrame() 
        self.wavelenght_min = 450 

    def load_data(self): 
        for filename in self.filenames: 
            if filename.endswith('.csv'):  
               
                self.combine_csvs(filename) 
 
            else:
                print(f"Unsupported file format: {filename}")
        print("ALL files combined into: ", self.path)
        
    def combine_csvs(self, filename):
        new_df = pd.read_csv(filename)
        if self.firstCsv:
            self.firstCsv = False
            self.combined_df = new_df
        else: 
            #ignore the first colum
            new_df = new_df.iloc[:, 1:]
            self.combined_df = pd.concat([self.combined_df, new_df], axis=1)
        
        self.remove_wavelengths()  
        self.combined_df.to_csv(self.path, index=False)
    
    def remove_wavelengths(self):
        headers = self.combined_df.columns.tolist()
        headers_excluding_wavelenght = [header for header in headers if not header.isnumeric()]
        headers_wavelenght = [header for header in headers if header.isnumeric()]
        #print(headers_wavelenght) 
        for wavelenght in headers_wavelenght:
            if int(wavelenght) < self.wavelenght_min:    
                #remove the row from the data frame
                self.combined_df = self.combined_df.drop(columns=[wavelenght])
    
    def fill_dict(self):
        
        headers = self.combined_df.columns.tolist()
        headers_excluding_wavelenght = [header for header in headers if not header.isnumeric()]
        headers_wavelenght = [header for header in headers if header.isnumeric()] 
        headers_traits = headers_excluding_wavelenght[4:]  # Assuming first 4 are ID, Genotype, Year, Conditions

        for index, row in self.combined_df.iterrows():
            id = row['ID']
            if id not in self.dataDict:
                self.dataDict[id] = {
                    'Genotype': row['Genotype'],
                    'Year': row['Year'],
                    'Conditions': row['Conditions'],
                    'Traits': {},
                    'Wavelengths': {}
                }
            # Fill traits
            for trait, value in zip(headers_traits, row[4:4+len(headers_traits)]):
                self.dataDict[id]['Traits'][trait] = value
            
            # Fill wavelengths
            for wavelength, value in zip(headers_wavelenght, row[4+len(headers_traits):]):
                self.dataDict[id]['Wavelengths'][wavelength] = value
        #print("Data dictionary filled with", len(self.dataDict), "entries.")   
        #print only the keys to self.dataDict = {}
        #print(list(self.dataDict.keys()))
        #print(self.dataDict[1])  
        return self.dataDict   

def separate_dict_by_value(value_key, data_dict):
    #break the dict into parts based of value such as genotype or contitions, ensuring each dict entires share the same value
    #the new dict still uses ID as the key, but the value is a list of entries that share the same value for the given key
    #e.g data_dict_1 = {id, genotype: A, ...}, data_dict_2 = {id, genotype: B, ...}
    
    #sort the dict by the value_key 
    sortedDict = {}
    for id, entry in data_dict.items():
        value = entry[value_key]
        if value not in sortedDict: 
            sortedDict[value] = {}
        sortedDict[value][id] = entry
    #write to txt file (sorted_dict) 
    
    write_dict_to_file(sortedDict, value_key)

    return sortedDict 

def write_dict_to_file(dict, lable):
    outputDir = './data/sorted_dicts/'  

    os.makedirs(outputDir, exist_ok=True)  # Ensure the output directory exists
    
    with open(f'{outputDir}sorted_dict_by_{lable}.txt', 'w') as f:
        f.write(str(dict)) 


def plot_wavelengths_reflectance(wavelenghts, reflectance, lable):

    # Ensure the output directory exists
    output_dir = './data/figures/reflectance_v_wavelength/'
    os.makedirs(output_dir, exist_ok=True) 

    plt.figure()
    x_label = 'Wavelength (nm)'
    x_points = wavelenghts
    y_label = 'Reflectance'
    y_points = reflectance

    plt.title(f"{lable}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #reduce the number of ticks on the x axis
    plt.xticks(np.arange(0, len(x_points), len(x_points)/10), fontsize=8)
    
    #save plot as a png file
    plt.savefig(f'{output_dir}{lable}_wavelenght_reflectance_plot.png')
    #plt.show() 
    plt.figure()   
    
    # Plot the data
    plt.plot(x_points, y_points)

if __name__ == '__main__':
    print("GO...")
    filenames = [
        'data/maize_2018_2019_unl_metadata.csv',
        'data/maize_2018_2019_unl_traits.csv', 
        'data/maize_2018_2019_unl_spectra.csv' 
        
    ]
    
    data = Data(filenames) 
    data.load_data()  
    dataDict = data.fill_dict()
    dataDictSort = separate_dict_by_value('Conditions', dataDict) #separate the dict by genotype 
    #for each entry to dataDictSort extract its wavelenght and reflectance to plot


    

#~~~~~~~~~~~~~~~~~~~~~ junk code ~~~~~~~~~~~~~~~~~~~~~~~
#Accessing each entry to the dict 
#print(dataDict[1])      
    #access each entry in the dict
    #for id, entry in dataDict.items():
        #print(f"ID: {id}, Genotype: {entry['Genotype']}, Year: {entry['Year']}, Conditions: {entry['Conditions']}")
        #print("Traits:", entry['Traits'])
        #for vi, value in entry['Traits'].items():
        #    print(f"{vi}:{value}") 
        #print("Wavelengths:", entry['Wavelengths'])

        #break """ 

#self.dataDict
#self.combined_df 
#print(self.combined_df.loc[0]) #counting stars from zero. 
#print(self.combined_df.at[0,'ID'])   
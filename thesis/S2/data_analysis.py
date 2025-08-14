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
import seaborn as sns  

from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler 

class C_Data:
    def __init__(self, filenames, wavelenght_min = 450): 
        self.filenames = filenames
        self.dataDict = {}
        self.firstCsv = True
        self.path = 'data/combined_data.csv' 
        self.combined_df = pd.DataFrame()
        self.all_df = []
        self.wavelenght_min = wavelenght_min 

    def load_data(self): 
        for filename in self.filenames: 
            if filename.endswith('.csv'):  
               
                self.combine_csvs(filename) 
 
            else:
                print(f"Unsupported file format: {filename}")
        print("ALL files combined into: ", self.path)
        return self.combined_df
        
    def combine_csvs(self, filename):
        new_df = pd.read_csv(filename)
        self.all_df.append(new_df)  # Store the dataframe for later use
        if self.firstCsv:
            self.firstCsv = False
            self.combined_df = new_df
        else: 
            #ignore the first colum
            new_df = new_df.iloc[:, 1:]
            self.combined_df = pd.concat([self.combined_df, new_df], axis=1)
        
        self.remove_wavelengths()  
        self.combined_df.drop(columns=['Year']) 
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
    
    #remove year, genotype, all wavelenghts, ect  

    def extract_headers(self):
        """headers = self.combined_df.columns.tolist()
        headers_excluding_wavelenght = [header for header in headers if not header.isnumeric()]
        headers_wavelenght = [header for header in headers if header.isnumeric()] """
        #do this headers_traits = headers_excluding_wavelenght[4:]  # Assuming first 4 are ID, Genotype, Year, Conditions but by usning the headers from the data frame
        headers = self.combined_df.columns.tolist()
        headers_excluding_wavelenght = [header for header in headers if not header.isnumeric()]
        headers_wavelenght = [header for header in headers if header.isnumeric()]

        headers_arr = []

        for dfx in self.all_df:
            # if  is not numeric 
            current_headers = dfx.columns.tolist()
            is_numerioc = 0 
            for header in current_headers:
                if header.isnumeric():
                    #headers_arr.append(new_headers)
                    is_numerioc = 1 
                    break 
            if is_numerioc:
                headers_arr.append(headers_wavelenght) 
            else:
                new_headers = [n_h for n_h in headers_excluding_wavelenght if n_h in current_headers and n_h != 'ID'] 
                headers_arr.append(new_headers)

    
        return headers_arr 
    
    def fill_dict(self):
        
        

        headers_all = self.extract_headers() #each entry is a differnt set of headers
        """
        headers = 
        [['Genotype', 'Year', 'Conditions'], 
        ['Nitrogen', 'Phosphorus', 'Potassium', 'Chlorophyll', 'Leaf_Dry_Weight', 'Leaf_Fresh_Weight', 'Leaf_Area'], 
        ['Leaf Structure Index', 'Chlorophyll index', 'Water index', 'Cellulose Absorption Index', 'Nitrogen Related Index NRI1510', 'Nitrogen Related Index NRI850', 'Disease Water Stress Index 1', 'Disease Water Stress Index 2', 'Disease Water Stress Index 3', 'Disease Water Stress Index 4', 'Disease Water Stress Index 5', 'Chlorophyll content'], 
        ['450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', ... ]] 
        """

            # Assume first 4 columns are: ID, Genotype, Year, Conditions
        metadata_headers = headers_all[0]
        trait_headers = headers_all[1]  # define N_traits appropriately
        hs_trait_headers = headers_all[2]
        headers_wavelenght = headers_all[3]  

        for index, row in self.combined_df.iterrows():
            id = row['ID']
            if id not in self.dataDict:
                self.dataDict[id] = {
                    'Metadata': {}, #headers 0, need in form { header_1:value, header_2:value}
                    'Traits': {}, #headers 1
                    'HS_Traits': {}, #headers 2
                    'Wavelengths': {} #headers 3
                }
                # Fill Metadata
            for header in metadata_headers:
                self.dataDict[id]['Metadata'][header] = row[header]

            # Fill Traits
            trait_start = len(metadata_headers) +1  
            trait_end = trait_start + len(trait_headers)
            for trait, value in zip(trait_headers, row[trait_start:trait_end]):
                self.dataDict[id]['Traits'][trait] = value

            # Fill HS_Traits
            hs_start = trait_end
            hs_end = hs_start + len(hs_trait_headers)
            for hs_trait, value in zip(hs_trait_headers, row[hs_start:hs_end]):
                self.dataDict[id]['HS_Traits'][hs_trait] = value

            # Fill Wavelengths
            for wavelength, value in zip(headers_wavelenght, row[hs_end:]):
                self.dataDict[id]['Wavelengths'][wavelength] = value 

        return self.dataDict   

#~~~~~~~~~~~~~~~~~~~~ dict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class C_Dict_manager:
    def __init__(self, inital_dict = {}): 
        self.inital_dict = inital_dict 

    def get_dict(self):
        return self.inital_dict
    
    def separate_dict_by_value(self, value_key, main_key = 'Metadata'):
        #break the dict into parts based of value such as genotype or contitions, ensuring each dict entires share the same value
        #the new dict still uses ID as the key, but the value is a list of entries that share the same value for the given key
        #e.g data_dict_1 = {id, genotype: A, ...}, data_dict_2 = {id, genotype: B, ...}


        #sort the dict by the value_key 
        sortedDict = {}
        for id, entry in self.inital_dict.items():
            value = entry[main_key][value_key]
            if value not in sortedDict: 
                sortedDict[value] = {}
            sortedDict[value][id] = entry
        #write to txt file (sorted_dict) 
        
        self.write_dict_to_file(self.inital_dict, value_key) 

        return sortedDict 

    def write_dict_to_file(self, dict, lable):
        outputDir = './data/sorted_dicts/'  

        os.makedirs(outputDir, exist_ok=True)  # Ensure the output directory exists
        
        with open(f'{outputDir}sorted_dict_by_{lable}.txt', 'w') as f:
            f.write(str(dict)) 

#~~~~~~~~~~~~~~~~~~~~~~~ plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class C_Plot_Wavelenght_reflectance:
    def __init__(self, inital_dict = {}):
        self.wavelengths_arr = []
        self.reflectance_arr = []
        self.lables = [] 
        self.dataDictSort = inital_dict

    def group_lot_wavelengths_reflectance(self, sort_key):
        # plot each entry on the same plot, with different colors for each entry
        # Ensure the output directory exists
        output_dir = './data/figures/reflectance_v_wavelength/'
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        x_label = 'Wavelength (nm)'
        y_label = 'Reflectance'
        
        # Plot each entry with a different color
        for i, (wavelengths_values, reflectance_values) in enumerate(zip(self.wavelengths_arr, self.reflectance_arr)):
            plt.plot(wavelengths_values, reflectance_values, label=self.lables[i]) 
                 
        plt.title("Reflectance vs Wavelength")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        #save to imae
        plt.savefig(f'{output_dir}grouped_wavelengths_reflectance_plot_by_{sort_key}.png')
        plt.show() 
        plt.close()  

    def plot_wavelengths_reflectance(self, wavelenghts, reflectance, lable):

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
        
        # Plot the data
        plt.plot(x_points, y_points)

        #save plot as a png file
        plt.savefig(f'{output_dir}{lable}_wavelenght_reflectance_plot.png')
        plt.show() 
    
        
        plt.close()   

    def plot_dict_wavelenghts(self, sort_key):

        for key, entries in self.dataDictSort.items():
            # Dictionary to accumulate reflectance values per wavelength
            wavelength_accumulator = {}
            count = {} 
            self.lables.append(key) 

            for entry in entries.values(): 
                for wl_str, refl in entry['Wavelengths'].items():
                    wl = int(wl_str)
                    wavelength_accumulator[wl] = wavelength_accumulator.get(wl, 0) + refl
                    count[wl] = count.get(wl, 0) + 1

            # Compute average reflectance per wavelength
            wavelengths = sorted(wavelength_accumulator.keys())
            reflectance = [wavelength_accumulator[wl] / count[wl] for wl in wavelengths]
            self.wavelengths_arr.append(wavelengths)
            self.reflectance_arr.append(reflectance)

            self.plot_wavelengths_reflectance(wavelengths, reflectance, key)
        self.group_lot_wavelengths_reflectance(sort_key)  

 
class C_Covariance_analysis:
    def __init__(self, init_df, init_dict, samples=[], features_names=[]):
        self.init_df = init_df
        self.init_dict = init_dict

        scaler = StandardScaler()
        self.samples = scaler.fit_transform(samples)  # 2D array: samples x features
        self.features_names = features_names  # List of feature names

        self.output_dir = './data/figures/covariance/'
        os.makedirs(self.output_dir, exist_ok=True)  

    def update_samples_features(self, samples, features_names):
        self.samples = samples
        self.features_names = features_names 
    
    def perform_covariance_analysis(self, name): #, samples_name = 'Samples', features_name = 'Features' 
        # Ensure the output directory exists
        cov = EmpiricalCovariance().fit(self.samples)
        sns.heatmap(
            cov.covariance_,
            annot=True,
            cmap='coolwarm',
            square=True,
            xticklabels=self.features_names,
            yticklabels=self.features_names 
        )
        plt.title('Covariance Matrix') 
        
        #save to self.output_dir
        plt.savefig(f'{self.output_dir}covariance_matrix_{name}.png')
        plt.show() 
        plt.close()  # Close the plot to free memory
        print(f"Covariance matrix saved to {self.output_dir}covariance_matrix_{name}.png")

if __name__ == '__main__':  
    print("GO...")
    filenames = [
        'data/maize_2018_2019_unl_metadata.csv',
        'data/maize_2018_2019_unl_traits.csv', 
        'data/maize_2018_2019_unl_additional_traits.csv', 
        'data/maize_2018_2019_unl_spectra.csv'  
        
    ]
    
    data = C_Data(filenames)  
    df = data.load_data()  

    dataDict = data.fill_dict()

    
    
    # Prepare data for covariance analysis on traits
    trait_names = list(next(iter(dataDict.values()))['Traits'].keys())
    samples = []
    for entry in dataDict.values():
        samples.append([entry['Traits'][trait] for trait in trait_names])

    covar_analysis = C_Covariance_analysis(df, dataDict, samples, trait_names) 
    covar_analysis.perform_covariance_analysis('Traits') 

    sort_key = 'Conditions'
    dictManager = C_Dict_manager(dataDict)  
    dataDictSort = dictManager.separate_dict_by_value(sort_key) #separate the dict by genotype 
    #for each entry to dataDictSort extract its wavelenght and reflectance to plot
    plotWR = C_Plot_Wavelenght_reflectance(dataDictSort)  
    plotWR.plot_dict_wavelenghts(sort_key) #group the wavelengths and reflectance by lables 
 

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
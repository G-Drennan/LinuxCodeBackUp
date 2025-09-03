#CSV files: data/maize_2018_2019_unl_metadata.csv, data/maize_2018_2019_unl_spectra.csv, data/maize_2018_2019_unl_traits.csv


#: gen alg feature selection https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection 
#TODO: perform DT on code and save the outputs


#: reduce the spectra measurements to be multiles of 5
#TODO: svr with spectra ~ 

#TODO: expand on HS_traits. ~

#TODO: more plots, better titles, ensure they are saved every time. ~

#TODO:  PCA   

  

import numpy as np
import math 

import os
import pandas as pd
import csv

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns  
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
from collections import defaultdict  
from scipy.spatial import ConvexHull

from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from sklearn.svm import SVR  
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import cross_val_score
from random import randint
 

class C_Data:
    def __init__(self, filenames,filenames_keys, drop_list = [], wavelenght_min = 450, reduce_wavelenghts = False): 
        if len(filenames_keys)-1 != len(filenames):
            print("WARN: key not same lenght as file names")
            
        self.filenames = filenames
        self.dataDict = {}
        self.firstCsv = True
        self.path = 'data/combined_data.csv'  
        self.combined_df = pd.DataFrame()
        self.all_df = []
        self.wavelenght_min = wavelenght_min 
        self.filenames_keys = filenames_keys
        self.drop_list = drop_list  
        self.reduce_wavelenghts = reduce_wavelenghts

    def load_data(self): 
        for filename in self.filenames: 
            if filename.endswith('.csv'):  
               
                self.combine_csvs(filename) 
 
            else:
                print(f"Unsupported file format: {filename}")
        print("ALL files combined into: ", self.path) 
        return self.combined_df, self.fill_dict()
        
    def combine_csvs(self, filename):
        new_df = pd.read_csv(filename)
        self.all_df.append(new_df)  # Store the dataframe for later use
        if self.firstCsv:
            self.firstCsv = False
            self.combined_df = new_df
        else: 
            #ignore the first colum this is the same col for all sheets, the ID col.  
            new_df = new_df.iloc[:, 1:]
            self.combined_df = pd.concat([self.combined_df, new_df], axis=1)
        
        self.remove_cols()
        self.combined_df.to_csv(self.path, index=False)
    
    def remove_cols(self):
        for header in self.drop_list:
            self.combined_df.drop(columns=[header]) 
        self.remove_wavelengths(self.reduce_wavelenghts)  

    def remove_wavelengths(self, reduce_wavelenghts = False):
        headers = self.combined_df.columns.tolist()
        headers_excluding_wavelenght = [header for header in headers if not header.isnumeric()]
        headers_wavelenght = [header for header in headers if header.isnumeric()]
        #print(headers_wavelenght) 
        for wavelenght in headers_wavelenght:
            if int(wavelenght) < self.wavelenght_min:    
                #remove the row from the data frame
                self.combined_df = self.combined_df.drop(columns=[wavelenght], errors='ignore') 
        if reduce_wavelenghts:
            #remove wavelengths that are not a multieple of 5 
            for wavelenght in headers_wavelenght: 
                if int(wavelenght) % 5 != 0:    
                    #remove the row from the data frame
                    self.combined_df = self.combined_df.drop(columns=[wavelenght],  errors='ignore')  
    
    #remove year, genotype, all wavelenghts, ect  

    def extract_headers(self):

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
                new_headers = [n_h for n_h in headers_excluding_wavelenght if n_h in current_headers and n_h != self.filenames_keys[0]]  
                headers_arr.append(new_headers)

    
        return headers_arr 
    
    def fill_dict(self):   
        
        headers_all = self.extract_headers() #each entry is a differnt set of headers

        for index, row in self.combined_df.iterrows():
            id = row[self.filenames_keys[0]]
            if id not in self.dataDict:
                self.dataDict[id] = {key: {} for key in self.filenames_keys[1:]}
                # Fill Metadata
            start = 0
            end = 0
            count = 0
            
            for key, headers in zip(self.filenames_keys[1:], headers_all):
                for header in headers:
                    self.dataDict[id][key][header] = row[header] 
            
        return self.dataDict   

#~~~~~~~~~~~~~~~~~~~~ dict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class C_Dict_manager:
    def __init__(self, inital_dict = {}): 
        self.inital_dict = inital_dict 

    def get_dict(self):
        return self.inital_dict
    
    def separate_dict_by_value(self, value_key, main_key):
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
        lable = f"sorted_by_{value_key}"
        self.write_dict_to_file(sortedDict, lable)   

        return sortedDict 

    def write_dict_to_file(self, dict, lable):
        outputDir = './data/sorted_dicts/'  

        os.makedirs(outputDir, exist_ok=True)  # Ensure the output directory exists
        
        with open(f'{outputDir}dict_{lable}.txt', 'w') as f:
            f.write(str(dict)) 

#~~~~~~~~~~~~~~~~~~~~~~~ plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class C_Plot_Wavelenght_reflectance:
    def __init__(self, inital_dict = {}):
        self.wavelengths_arr = []
        self.reflectance_arr = []
        self.reflectance_arr_std = []
        self.lables = [] 
        
        self.reflectance_min = []
        self.reflectance_max = [] 
        self.dataDictSort = inital_dict
    
    def plot_dict_wavelenghts(self, sort_key, dict_key):

        for key, entries in self.dataDictSort.items():
            # Dictionary to accumulate reflectance values per wavelength
            wavelength_accumulator = {}
            count = {} 
            self.lables.append(key) 

            #data in form of  'Spectra': {'450': 0.055956043, '451': 0.056224021, '452': 0.056229805, '453': 0.056243247, '454': 0.056340493, '455': 0.056314658, '456': 0.056342624, ect
            wavelength_accumulator = defaultdict(list) 
            for entry in entries.values(): 
                for wl_str, refl in entry[dict_key].items():
                    wl = int(wl_str)
                    wavelength_accumulator[wl].append(refl)
            wavelengths = wavelength_accumulator.keys() 
            reflectance = [np.mean(wavelength_accumulator[wl]) for wl in wavelengths]
            reflectance_std = [np.std(wavelength_accumulator[wl]) for wl in wavelengths]
            reflectance_min = [np.min(wavelength_accumulator[wl]) for wl in wavelengths]
            reflectance_max = [np.max(wavelength_accumulator[wl]) for wl in wavelengths]

            self.wavelengths_arr.append(wavelengths)  
            self.reflectance_arr.append(reflectance)
            self.reflectance_arr_std.append(reflectance_std)
            self.reflectance_min.append(reflectance_min)
            self.reflectance_max.append(reflectance_max)


            self.plot_wavelengths_reflectance(wavelengths, reflectance, (reflectance_min, reflectance_max), key)
        self.group_lot_wavelengths_reflectance(sort_key)  

    def group_lot_wavelengths_reflectance(self, sort_key):
        # plot each entry on the same plot, with different colors for each entry
        # Ensure the output directory exists
        output_dir = './data/figures/reflectance_v_wavelength/'
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        x_label = 'Wavelength (nm)'
        y_label = 'Reflectance'
        
        # Plot each entry with a different color
        for i, (wavelengths_values, reflectance_values, std, max_vals, min_vals) in enumerate(zip(self.wavelengths_arr, self.reflectance_arr, self.reflectance_arr_std, self.reflectance_max, self.reflectance_min)):
            line, = plt.plot(wavelengths_values, reflectance_values, label=self.lables[i])
            plt.fill_between(wavelengths_values, min_vals, max_vals, color = line.get_color(), alpha=0.3)   
            
                 
        plt.title("Reflectance vs Wavelength")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)  

        #save to imae
        plt.savefig(f'{output_dir}grouped_wavelengths_reflectance_plot_by_{sort_key}.png')
        plt.show() 
        plt.close()  

    def plot_wavelengths_reflectance(self, wavelenghts, reflectance, min_max, lable):

        # Ensure the output directory exists
        output_dir = './data/figures/reflectance_v_wavelength/'
        os.makedirs(output_dir, exist_ok=True) 

        plt.figure()
        x_label = 'Wavelength (nm)'
        x_points = wavelenghts
        y_label = 'Reflectance'
        y_points = reflectance
        min_vals = np.array(min_max[0])
        max_vals = np.array(min_max[1]) 

        plt.title(f"{lable}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #reduce the number of ticks on the x axis
        plt.xticks(np.arange(0, len(x_points), len(x_points)/10), fontsize=8)
        plt.grid(True) 

        # Plot with std
        print(f"{lable}: min {min_vals} max: {max_vals}") 
        #print(f"Reflectance:{reflectance}")  
        # Plot the data 
        line, = plt.plot(x_points, y_points)  
        plt.fill_between(wavelenghts, min_vals, max_vals, color = line.get_color(), alpha=0.3)         


        #save plot as a png file
        plt.savefig(f'{output_dir}{lable}_wavelenght_reflectance_plot.png')
        plt.show() 
    
        
        plt.close()   

    
 
class C_analysis:
    def __init__(self, init_df, init_dict):
        self.init_df = init_df
        self.init_dict = init_dict
        self.output_dir_covariance = './data/figures/covariance/'
        os.makedirs(self.output_dir_covariance, exist_ok=True)   

    def update_features(self, features_names): 
        self.features_names = features_names 
    
    def perform_covariance_analysis(self, filenames_key, normalise = True): #, samples_name = 'Samples', features_name = 'Features' 
        features_names = list(next(iter(self.init_dict.values()))[filenames_key].keys())
        samples = []
        for entry in self.init_dict.values():
            samples.append([entry[filenames_key][trait] for trait in features_names])
 
        #normalise samples
        if normalise:
            scaler = StandardScaler()
            samples = scaler.fit_transform(samples) 

        cov = EmpiricalCovariance().fit(samples)
        sns.heatmap(
            cov.covariance_,
            annot=True,
            cmap='coolwarm',
            square=True,
            xticklabels=features_names,
            yticklabels=features_names 
        )
        plt.title('Covariance Matrix') 
        
        #save to self.output_dir
        plt.savefig(f'{self.output_dir_covariance}covariance_matrix_{filenames_key}.png')
        plt.show() 
        plt.close()  
        print(f"Covariance matrix saved to {self.output_dir_covariance}covariance_matrix_{filenames_key}.png")

    def perform_covariance_analysis_two_diff_data_sets(self, filenames_key_1, filenames_key_2,  normalise = True):
        filenames_key_1 = list(next(iter(self.init_dict.values()))[filenames_key_1].keys())
        samples_1 = []
        for entry in self.init_dict.values():
            samples_1.append([entry[filenames_key_1][trait] for trait in filenames_key_1])

        filenames_key_2 = list(next(iter(self.init_dict.values()))[filenames_key_2].keys())
        samples_2 = []
        for entry in self.init_dict.values():
            samples_2.append([entry[filenames_key_2][trait] for trait in filenames_key_2])
        
        #One side of the covariance matrix is one data set the other the otehr data set
        combined_samples = np.hstack((samples_1, samples_2))
         #normalise both samples
        if normalise:
            scaler = StandardScaler()
            combined_samples = scaler.fit_transform(combined_samples) 
        
        combined_features = filenames_key_1 + filenames_key_2

        cov = EmpiricalCovariance().fit(combined_samples)
        sns.heatmap(
            cov.covariance_,
            annot=True,
            cmap='coolwarm',
            square=True,
            xticklabels=combined_features,
            yticklabels=combined_features 
        )
        plt.title('Covariance Matrix')
        #save to self.output_dir
        plt.savefig(f'{self.output_dir_covariance}covariance_matrix_{filenames_key_1}_and_{filenames_key_2}.png')
        plt.show()
        plt.close()
        print(f"Covariance matrix saved to {self.output_dir_covariance}covariance_matrix_{filenames_key_1}_and_{filenames_key_2}.png") 





#~~~~~~~~~~~~~~~~~~~~~ Regressors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

class C_Test_train_split:

    def __init__(self, dataDict, rand_state=42):
        self.dataDict = dataDict
        self.training_sets_made = False
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.cond_train = None
        self.cond_test = None 

        self.random_state = rand_state  # For reproducibility 

    def make_training_n_test_sets(self, features_key, class_main_key, class_name, test_size=0.4, conditions_name = 'Conditions', conditions_key = 'Metadata', normalise = True):
        
        x, keys = self.extract_features(features_key)
        y, trait_key = self.extract_class(class_main_key, class_name)
        conditions, _ = self.extract_class(conditions_key, conditions_name)  
        
        self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test = self.stratified_split(x, y, conditions, test_size)
        
        if normalise: 
            scaler = StandardScaler() 
            self.x_train = scaler.fit_transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)

        self.condition_distribution()

        #prove that the split is stratified
        print("Training set size:", len(self.x_train), "Test set size:", len(self.x_test))
        self.training_sets_made = True  

        return self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test, keys, trait_key 

    def stratified_split(self, x, y, conditions, test_size=0.4):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        for train_idx, test_idx in sss.split(x, y=conditions):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            cond_train, cond_test = np.array(conditions)[train_idx], np.array(conditions)[test_idx]
            return x_train, x_test, y_train, y_test, cond_train, cond_test
       
    def condition_distribution(self):
        if self.training_sets_made:
            print("Unique conditions in test set:", np.unique(self.cond_train))
            
            train_counts = Counter(self.cond_train)
            test_counts = Counter(self.cond_test)
            all_conditions = sorted(set(self.cond_train) | set(self.cond_test))

            print("\nCondition distribution (% in test set):")
            for cond in all_conditions:
                train_n = train_counts.get(cond, 0)
                test_n = test_counts.get(cond, 0)
                total = train_n + test_n
                pct_test = 100 * test_n / total if total > 0 else 0
                print(f"  {cond}: {test_n}/{total} ({pct_test:.1f}%) in test") 

    def extract_features(self, main_key): 
        sample_entry = next(iter(self.dataDict.values()))[main_key]
    
        # Check if keys are numeric 
        try:
            num_keys = sorted([int(k) for k in sample_entry.keys()])
            keys = [str(k) for k in num_keys]
            x = [[entry[main_key][str(k)] for k in keys] for entry in self.dataDict.values()]
        except ValueError:
            keys = list(sample_entry.keys())
            x = [[entry[main_key][k] for k in keys] for entry in self.dataDict.values()]
        
        return np.array(x), keys

    def extract_class(self, main_key, trait_name):
        y = []
        for entry in self.dataDict.values(): 
            y.append(entry[main_key][trait_name])

        return np.array(y), trait_name

class C_svr:
    def __init__(self, dataDict): 

        self.tnt = C_Test_train_split(dataDict)

    def get_svr_kernal(self, kernalType = "poly"):
        self.srv_exists = True
        if kernalType == "rbf":
            return SVR(kernel=kernalType, C=100, gamma=0.1, epsilon=0.1)
        elif kernalType == "linear":
            return SVR(kernel=kernalType, C=100, gamma="auto")
        elif kernalType == "poly":
            return SVR(
                    kernel=kernalType, 
                    C=100,
                    gamma="auto",
                    degree=3,
                    epsilon=0.1,
                    coef0=1
                )
        else: 
            self.srv_exists = False
            raise ValueError(f"Unsupported kernel: {kernalType!r}")

    def plot_svr_all_features(self, features_key, class_main_key): 
        print("Start new set of srv.") 

        class_names = list(next(iter(dataDict.values()))[class_main_key].keys())
        # Prepare for combined plotting 
        n_traits = len(class_names) 
        ncols = 2
        nrows = math.ceil(n_traits / ncols)
        output_dir = './data/figures/svr/' 
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes = axes.flatten()

        for i, name in enumerate(class_names):  
            print(name)
            self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test, keys, trait_key    = self.tnt.make_training_n_test_sets(features_key, class_main_key, name) 
            svr = self.get_svr_kernal("linear") 
            svr.fit(self.x_train, self.y_train) 
            y_pred_test = svr.predict(self.x_test)
            ax = axes[i] 

            # Map conditions to colors
            unique_conditions = np.unique(self.cond_test)
            color_map = {cond: cm.tab20(j / len(unique_conditions)) for j, cond in enumerate(unique_conditions)}
            colors = [color_map[cond] for cond in self.cond_test]

            scatter = ax.scatter(self.y_test, y_pred_test, c=colors, alpha=0.7, edgecolor="k", s=50)
            min_val = min(np.min(self.y_test), np.min(y_pred_test))
            max_val = max(np.max(self.y_test), np.max(y_pred_test))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            r2 = r2_score(self.y_test, y_pred_test)
            print(f"{name} score: {svr.score(self.x_test, self.y_test)}")
            #r2_score(self.y_test, y_pred_test) and svr.score(self.x_test, self.y_test) produce smae results 
            mse = mean_squared_error(self.y_test, y_pred_test)
            ax.set_title(f"{name}\n$R^2$={r2:.3f}, MSE={mse:.2e}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted") 

            # Add legend for conditions
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(cond),
                                markerfacecolor=color_map[cond], markersize=10)
                    for cond in unique_conditions]
            ax.legend(handles=handles, title="Condition", loc='best', fontsize=8)

        # Hide unused axes if any
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        lable = features_key 
        fig.suptitle(f"SVR Test Set Performance for All Traits using {lable} as features", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        fig_path = f"{output_dir}svr_all_traits_{lable}.png"
        fig.savefig(fig_path) 
        print(f"Figure saved to: {fig_path}")
        plt.show() 
        plt.close('all') 
        plt.clf()  
 

class C_Dession_trees:
    def __init__(self, dataDict, x_train, x_test, y_train, y_test):
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test 
        

        self.models = ["RF", "AB", "GB", "DT"]
        self.models_full = ["Random Forest", "Ada Boost", "Gradient Boosting", "Decision Tree"] 
        self.models_dict = {"RF":"Random Forest", "AB":"Ada Boost", "GB":"Gradient Boosting", "DT":"Decision Tree"}
        self.model = None #model 
        self.model_exists = False 
        self.y_pred = None 

    def train_model(self, features_key, class_main_key, class_name, model = "RF"): # features_name,
  
        print(f"{class_name}: Training {model} model")  
        if model == "RF": 
            print(model)
            self.random_forest_Regressor()
        elif model == "AB":
            print(model)
            self.Ada_Boost_Regressor()
        elif model == "GB":
            print(model)
            self.Gradient_Boosting_Regressor()
        elif model == "DT": 
            print(model)
            self.desision_tree_regressor() 
        
        
        self.r2_score, self.mse = self.my_accuracy_score()    
        print(f"{self.models_dict[model]} R2: {self.r2_score:.2f}")
        
        scores, cross_val_mean = self.cross_val()  

        print(f"Cross-validation scores of {self.models_dict[model]}: {scores}")
        print(f"Mean cross-validation score: {cross_val_mean:.2f}") 

        return self.model, self.r2_score, cross_val_mean  

    def desision_tree_regressor(self, max_depth=5):
        
        self.model = DecisionTreeRegressor(max_depth=max_depth)
        self.model.fit(self.x_train, self.y_train) 
        self.y_pred = self.model.predict(self.x_test)
        
        self.model_exists = True
        return self.model#, self.accuracy

    def Gradient_Boosting_Regressor(self, n_est = 100, Lr = 0.1): 
        self.model = GradientBoostingRegressor(n_estimators=n_est, learning_rate = Lr)
        self.model.fit(self.x_train, self.y_train) 
        self.y_pred = self.model.predict(self.x_test)

        self.model_exists = True
        return self.model#, self.accuracy

    def Ada_Boost_Regressor(self, n_est = 100, Lr = 1.0):
        self.model = AdaBoostRegressor(n_estimators=100, learning_rate=1.0)
        self.model.fit(self.x_train, self.y_train) 
        self.y_pred = self.model.predict(self.x_test)

        self.model_exists = True
        return self.model#, self.accuracy

    def random_forest_Regressor(self, n_est = 100):   
         #n_est =  number of trees in the forest.
        self.model =  RandomForestRegressor(n_estimators = n_est)
        self.model.fit(self.x_train, self.y_train) 
        self.y_pred = self.model.predict(self.x_test)

        self.model_exists = True
        return self.model#, self.accuracy
    
    def my_accuracy_score(self): 
        #accuracy = np.mean(y_pred == self.y_test)
        #score = self.model.score(self.x_test, self.y_test) 

        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        #F value F1 = 2 * (precision * recall) / (precision + recall
        return r2, mse 
    
    #Cross-validation test the models performance. 
    def cross_val(self, model = None, n_splits=10):
        if self.model_exists and model == None: 
            scores = cross_val_score(self.model, self.x_train, self.y_train, cv=n_splits) #CV no. folds 
            #returnes scores: ndarray of float of shape=(len(list(cv)),)
            #Array of scores of the estimator for each run of the cross validation.
            cross_val_mean = scores.mean()
        elif model!= None:
            scores = cross_val_score(model, self.x_train, self.y_train, cv=n_splits, scoring='r2') 
            cross_val_mean = scores.mean() 

        else:
            print("Model not trained yet.")
            return None 

        return scores, cross_val_mean 
    
    def confusion_matrix(self, model = None):
        if self.model_exists and model == None:
            y_pred = self.model.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
            print("Confusion Matrix:") 
            print(cm)
            return cm
        elif model!= None:
            y_pred = model.predict(self.x_test) 
            cm = confusion_matrix(self.y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            return cm
        else:
            print("Model not trained yet.")
            return None

class C_gen_alg:
    def __init__(self, dataDict): 
        self.model = None
        self.model_exists = False 
        self.dataDict = dataDict 
        self.tnt = C_Test_train_split(dataDict)
        self.outPutPath = './data/ML/'
        self.score_file_path = None 
        #create the output dir if it does not exist
        os.makedirs(self.outPutPath, exist_ok=True) 

        #class_names = list(next(iter(dataDict.values()))[class_main_key].keys())
        #for i, name in enumerate(class_names): #for each class name

    
    def gen_alg_on_best_model(self, features_key, class_main_key, class_name):
        #chromo_df_bc,score_bc=generations(data_bc,label_bc)
        self.score_file_path = os.path.join(self.outPutPath, f'model_scores_{features_key}.txt')        #create the file 
        #whipe the file if it exists
        with open(f'{self.score_file_path}', 'w') as f:
            f.write(f"--- Model scores for features: {features_key} ---\n") 
        self.find_best_model(features_key, class_main_key, class_name)

        n_feat = self.x_train.shape[1] 
        print(n_feat)  
        n_gen= 10
        best_chromo_x, best_score = self.generations(n_feat=n_feat, n_gen=n_gen) #size=5, n_parents=4, 
        self.best_chromo_x_overall = max(zip(best_score, best_chromo_x), key=lambda x: x[0])[1]
 
        print("Best feature subset found:", np.where(self.best_chromo_x_overall)[0])
        #translate the best chromo to the feature names
        feature_names = np.array(self.tnt.extract_features(features_key)[1])
        best_features = feature_names[self.best_chromo_x_overall]
        print("Best feature names:", best_features)
        print("Corresponding R2:", best_score[0]) 

        with open(f'{self.score_file_path}', 'a') as f: 
            f.write(f"\n--- Generation alg ---\n") 
            f.write(f"num genrations: {n_gen}\n")
            f.write(f"Best feature subset found: {np.where(self.best_chromo_x_overall)[0]},\n Best feature names: {best_features},\n Corresponding R2: {best_score[0]}\n") 
        self.run_best_chromo_on_other_ML(features_key, class_main_key, class_name)
        #
    #func with the best_chromo_x_overall, run each ML model using the  best_chromo_x_overall
    def run_best_chromo_on_other_ML(self, features_key, class_main_key, class_name): 
        #write this func 
        full_x, feature_names = self.tnt.extract_features(features_key)
        feature_names = np.array(feature_names) 
        selected_indices = np.where(self.best_chromo_x_overall)[0]
        selected_features = feature_names[self.best_chromo_x_overall]

        # Subset the data to selected features
        x_selected = full_x[:, selected_indices]

        # Re-split using selected features
        self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test, keys, trait_key = \
            self.tnt.make_training_n_test_sets(features_key, class_main_key, class_name)  

        # Train and evaluate all models on selected features
        model_obj = C_Dession_trees(self.dataDict, self.x_train, self.x_test, self.y_train, self.y_test)
        with open(f'{self.score_file_path}', 'a') as f: 
            f.write("\n--- Performance on Best Chromosome Feature Subset ---\n")
            f.write(f"Selected Features: {selected_features.tolist()}\n")

            for i, model in enumerate(model_obj.models):
                model, r2_score, cross_val_mean = model_obj.train_model(features_key, class_main_key, class_name, model)
                f.write(f"{model_obj.models_full[i]}: R2={r2_score:.2f}, Cross-val mean={cross_val_mean:.2f}\n")

        print("Finished evaluating all models on best feature subset.")  

    
    def find_best_model(self, features_key, class_main_key, name): 
        max_cross_val_mean = 0

        self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test, keys, trait_key  = self.tnt.make_training_n_test_sets(features_key, class_main_key, name) 
        model_obj = C_Dession_trees(dataDict, self.x_train, self.x_test, self.y_train, self.y_test) 
        for i, model in enumerate(model_obj.models): #        self.models = ["RF", "AB", "GB", "DT"]
 
            #genetic feature selection, save best features from first run use for other runs for consistencey and comparision. 
            model, r2_score, cross_val_mean  = model_obj.train_model(features_key, class_main_key, name, model)
            #write the model and its score to a file
            with open(f'{self.score_file_path}', 'a') as f:  
                f.write(f"{model_obj.models_full[i]}: R2={r2_score:.2f}, Cross-val mean={cross_val_mean:.2f}\n")

            if cross_val_mean > max_cross_val_mean : 
                max_cross_val_mean = cross_val_mean
                self.model = model
                self.model_exists = True  
                best_model_index = i   
    
        print(f"{model_obj.models_full[best_model_index]} highest accuracy of {max_cross_val_mean}.") 
        
        
    def generations(self,n_feat,size=80,n_parents=64,mutation_rate=0.20,n_gen=5):
        print("Start generations")
        best_chromo_x= []
        best_score= [] 
        population_nextgen= self.initilization_of_population(size,n_feat)
        for i in range(n_gen):
            scores, pop_after_fit = self.fitness_score(population_nextgen)  
            print('Best score in generation',i+1,':',scores[:1])  #2
            pop_after_sel = self.selection(pop_after_fit,n_parents)
            pop_after_cross = self.crossover(pop_after_sel) 
            population_nextgen = self.mutation(pop_after_cross,mutation_rate,n_feat)
            best_chromo_x.append(pop_after_fit[0])
            best_score.append(scores[0])
        return best_chromo_x, best_score 

    def initilization_of_population(self, size,n_feat):
        print("Start initilization of population") 
        population = []
        for i in range(size):
            chromosome = np.ones(n_feat,dtype=np.bool)     
            chromosome[:int(0.3*n_feat)]=False             
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population


    def fitness_score(self, population, use_cross_val = False):
        print("fitness_score")
        scores = []
        for chromosome in population:  
            #model is the curretn model.   
            #model = RandomForestClassifier(n_estimators=200, random_state=0)
            selected_features = np.where(chromosome)[0]
            self.model.fit(self.x_train[:, selected_features], self.y_train)
            scores.append(self.retrive_score(chromosome, use_cross_val))   
        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)                                     
        return list(scores[inds][::-1]), list(population[inds,:][::-1])  
    
    def retrive_score(self, chromosome, use_cross_val = False): 
        #print("retrive_score")   
        if use_cross_val:
            cv_scores = cross_val_score(self.model, self.x_train[:, chromosome], self.y_train, cv=10, scoring='r2') 
            return cv_scores.mean()  
        else:
            predictions = self.model.predict(self.x_test[:, chromosome])    #self.model.fit(self.x_train[:, chromosome], self.y_train).predict(self.x_test[:, chromosome])
            return r2_score(self.y_test, predictions)    

    def selection(self, pop_after_fit,n_parents):
        print("selection")
        population_nextgen = []
        for i in range(n_parents):
            population_nextgen.append(pop_after_fit[i])
        return population_nextgen


    def crossover(self, pop_after_sel):
        print("crossover") 
        pop_nextgen = pop_after_sel
        for i in range(0,len(pop_after_sel),2):
            new_par = []
            child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
            new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
            pop_nextgen.append(new_par)
        return pop_nextgen


    def mutation(self, pop_after_cross,mutation_rate,n_feat):   
        print("mutation")  
        mutation_range = int(mutation_rate*n_feat)
        pop_next_gen = []
        for n in range(0,len(pop_after_cross)):
            chromo = pop_after_cross[n]
            rand_posi = [] 
            for i in range(0,mutation_range):
                pos = randint(0,n_feat-1)
                rand_posi.append(pos)
            for j in rand_posi:
                chromo[j] = not chromo[j]  
            pop_next_gen.append(chromo)
        return pop_next_gen
        
    def plot(score,x,y,c = "b"): #plot the generation accuracies. 
        gen = [1,2,3,4,5]
        plt.figure(figsize=(6,4))
        ax = sns.pointplot(x=gen, y=score,color = c )
        ax.set(xlabel="Generation", ylabel="R2") 
        ax.set(ylim=(x,y)) 

class C_PCA: 
    def __init__(self, dataDict, sort_key,  feature_key='Hs_Traits'): 
        dictManager = C_Dict_manager(dataDict) 
        self.sortedDataDict = dictManager.separate_dict_by_value(sort_key, filenames_keys[1]) #separate the dict by genotype 
        self.feature_key = feature_key 
        self.sort_term_name = sort_key
        self.output_dir = "./data/pca/"  
        
        self.sorted_dict_to_dataframe()
        

    def sorted_dict_to_dataframe(self):
        rows = []
        for condition, samples in self.sortedDataDict.items(): 
            for sample_id, sample_data in samples.items():
                features = sample_data.get(self.feature_key, {})
                row = {'ID': sample_id, self.sort_term_name: condition}  
                row.update(features)
                rows.append(row)
        self.df =  pd.DataFrame(rows)
        #save the data frame
        output_dir_csv = './data/'  
        os.makedirs(output_dir_csv, exist_ok=True)    
        self.df.to_csv(f'{output_dir_csv}pca_input_dataframe_{self.feature_key}_sort_by_{self.sort_term_name}.csv', index=False)

    def plot_pca_clusters(self):

        #assume teh feature cols always exclude the first 2 and the rest are included
        feature_cols = self.df.columns[2:].tolist() 
        label_col = self.sort_term_name  
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.df[feature_cols])
        self.df['PC1'], self.df['PC2'] = X_pca[:, 0], X_pca[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.colormaps['tab10'] 

        for i, condition in enumerate(self.df[label_col].unique()):
            subset = self.df[self.df[label_col] == condition]
            ax.scatter(subset['PC1'], subset['PC2'], label=condition, color=colors(i), alpha=0.6)

            if len(subset) >= 3:
                points = subset[['PC1', 'PC2']].values
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                ax.plot(*zip(*np.append(hull_pts, [hull_pts[0]], axis=0)), color=colors(i), lw=2)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        ax.set_title(f'PCA Clusters by {label_col}')
        #plt.tight_layout()  
        plt.savefig(f'{self.output_dir}pca_clusters_{self.feature_key}_sort_by_{self.sort_term_name}.png')
        plt.show()
        plt.close() 

    
    



if __name__ == '__main__':    
    print("GO...")
    filenames = [
        'data/maize_2018_2019_unl_metadata.csv',
        'data/maize_2018_2019_unl_traits.csv', 
        'data/maize_2018_2019_unl_additional_traits.csv', 
        'data/maize_2018_2019_unl_spectra.csv'  
        
    ]
    filenames_keys = [
        'ID', 
        'Metadata', 
        'Traits',
        'Hs_Traits',
        'Spectra'
    ] 
     
    data = C_Data(filenames, filenames_keys, reduce_wavelenghts = True)     
    df, dataDict = data.load_data()
    dictManager = C_Dict_manager(dataDict)

    # Ploting spectra based off conditions
    sort_key = 'Conditions' 
    dictManager.write_dict_to_file(dataDict, "original") 

    ga = C_gen_alg(dataDict)
    class_names = list(next(iter(dataDict.values()))[filenames_keys[2]].keys())  
    ga.gen_alg_on_best_model(filenames_keys[3], filenames_keys[2], class_names[0]) 

    """
    pca = C_PCA(dataDict, sort_key) 
    pca.plot_pca_clusters() 

    pca = C_PCA(dataDict, sort_key, feature_key=filenames_keys[4])  
    pca.plot_pca_clusters() 
    """ 
        



    """
    reg = C_svr(dataDict)  
    reg.plot_svr_all_features(filenames_keys[3], filenames_keys[2]) 
    #reg.plot_svr_all_features(filenames_keys[4], filenames_keys[2])  #spectra takes too long
    """   
   
    # Prepare data for covariance analysis on trait
    """
    Analysis = C_analysis(df, dataDict)   
    Analysis.perform_covariance_analysis(filenames_keys[3])  
    Analysis.perform_covariance_analysis(filenames_keys[2])
    """  

    """
    

    #for each entry to dataDictSort extract its wavelenght and reflectance to plot 
    plotWR = C_Plot_Wavelenght_reflectance(dataDictSort)     
    plotWR.plot_dict_wavelenghts(sort_key, filenames_keys[4]) #group the wavelengths and reflectance by lables
    
    """


   
     

#~~~~~~~~~~~~~~~~~~~~~ junk code ~~~~~~~~~~~~~~~~~~~~~~~

 

    """
    model = C_PCA(dataDict) 
    model.extract_features(keys=('Traits', 'Hs_Traits', 'Spectra')) 
    pca_output = model.run_pca(n_components=3)
    print("Explained variance:", model.explained_variance())
    model.plot_pca(metadata_key='Conditions') 
    """  

    #print("Trait name: ", trait_keys)  
    """
    pca = C_PCA(dataDict)
    x, keys = pca.extract_features(filenames_keys[4])
    y, trait_key = pca.extract_class(filenames_keys[2], trait_keys[2])
    print(x) 
    print(y)  
    pca.plot_pca(x,y) """
    #print(keys, x)   



    """
    output_dir = './data/pca/'
    os.makedirs(output_dir, exist_ok=True)  
    np.savetxt(f'{output_dir}pca_features.txt', x, delimiter=',', header=','.join(keys), comments='') 
    print(keys, x) """ 
    

    """
    dictManager = C_Dict_manager(dataDict)  
    dictManager.write_dict_to_file(dataDict, 'data') 
    """


  
    """
    
    def plot_svr_basic(self, svr, lable):
        # Fit on training data
        svr.fit(self.x_train, self.y_train) 

        # Generate predictions
        y_pred_train = svr.predict(self.x_train)
        y_pred_test  = svr.predict(self.x_test)

         # Prepare figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

        # Loop over train/test splits
        for ax, y_true, y_pred, split in zip(
                axes,
                (self.y_train,    self.y_test),
                (y_pred_train, y_pred_test),
                ("Train",    "Test")
        ):
                # Scatter actual vs. predicted
            ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="k", s=50)
                
                # Plot 1:1 line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)

            # Compute metrics
            r2  = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

                # Annotate
            ax.text(
                    0.05, 0.95,
                    f"$R^2$ = {r2:.3f}\nMSE = {mse:.3e}",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

            # Labels & title
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{split} Set {lable}") 

        fig.suptitle("SVR Performance", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


     
    #"""   

"""
class C_PCA:
    
    def plot_pca(self, X, y=None, n_components=2):
        # Standardize the data
        X_scaled = StandardScaler().fit_transform(X)

        # Run PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Plot the first two principal components
        plt.figure(figsize=(8, 6))
        if y is not None:
            for label in set(y):
                idx = y == label
                plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", alpha=0.6)
            plt.legend()
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.title("PCA: First Two Principal Components")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
"""
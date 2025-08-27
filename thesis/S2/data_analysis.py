#CSV files: data/maize_2018_2019_unl_metadata.csv, data/maize_2018_2019_unl_spectra.csv, data/maize_2018_2019_unl_traits.csv


#TODO: gen alg feature selection https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection 
#TODO: perform DT on code and save the outputs


#TODO: reduce the spectra measurements to be multiles of 5
#TODO: svr with spectra

#TODO: expand on HS_traits. 

#TODO: more plots, better titles, ensure they are saved every time.
#TODO:  PCA  

  

import numpy as np
import math 

import os
import pandas as pd
import csv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns  

from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from sklearn.svm import SVR 
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import cross_val_score

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
                self.combined_df = self.combined_df.drop(columns=[wavelenght])
        if reduce_wavelenghts:
            #remove wavelengths that are not a multieple of 5
            for wavelenght in headers_wavelenght:
                if int(wavelenght) % 5 != 0:    
                    #remove the row from the data frame
                    self.combined_df = self.combined_df.drop(columns=[wavelenght])  
    
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

    def plot_dict_wavelenghts(self, sort_key, dict_key):

        for key, entries in self.dataDictSort.items():
            # Dictionary to accumulate reflectance values per wavelength
            wavelength_accumulator = {}
            count = {} 
            self.lables.append(key) 

            for entry in entries.values(): 
                for wl_str, refl in entry[dict_key].items():
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
    
 
class C_analysis:
    def __init__(self, init_df, init_dict):
        self.init_df = init_df
        self.init_dict = init_dict
        self.output_dir_covariance = './data/figures/covariance/'
        os.makedirs(self.output_dir_covariance, exist_ok=True)   

    def update_samples_features(self, samples, features_names):
        self.samples = samples
        self.features_names = features_names 
    
    def perform_covariance_analysis(self, filenames_key): #, samples_name = 'Samples', features_name = 'Features' 
        features_names = list(next(iter(self.init_dict.values()))[filenames_key].keys())
        samples = []
        for entry in self.init_dict.values():
            samples.append([entry[filenames_key][trait] for trait in features_names])
 
        #normalise samples 
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples) 
        
        
        # Ensure the output directory exists
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
        plt.close()  # Close the plot to free memory 
        print(f"Covariance matrix saved to {self.output_dir_covariance}covariance_matrix_{filenames_key}.png")

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

    def make_training_n_test_sets(self, features_key, class_main_key, class_name, test_size=0.4, conditions_name = 'Conditions', conditions_key = 'Metadata'):
        
        x, keys = self.extract_features(features_key)
        y, trait_key = self.extract_class(class_main_key, class_name)
        conditions, _ = self.extract_class(conditions_key, conditions_name)  
        
        self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test = self.stratified_split(x, y, conditions, test_size)
        
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

        fig.suptitle("SVR Test Set Performance for All Traits", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f"{output_dir}svr_all_traits.png")  # Save figure
        plt.show() 

class C_Dession_trees:
    def __init__(self, dataDict):
        self.tnt = C_Test_train_split(dataDict)
        self.models = ["RF", "AB", "GB", "DT"]
        self.models_full = ["Random Forest", "Ada Boost", "Gradient Boosting", "Decision Tree"] 
        self.clf = None
        self.model_exists = False

    def feature_selection_genetic_alg(self, features_key, class_main_key):
        class_names = list(next(iter(dataDict.values()))[class_main_key].keys())

        for i, name in enumerate(class_names): #for each class name
            for i, model in enumerate(self.models):
                #genetic feature selection, save best features from first run use for other runs for consistencey and comparision. 
                self.train_models(features_key, class_main_key, name, model) 

    def train_models(self, features_key, class_main_key, class_name, model = "RF"): # features_name,
  
        print(f"{class_name}: Training {model} model") 
        self.x_train, self.x_test, self.y_train, self.y_test, self.cond_train, self.cond_test, keys, trait_key    = self.tnt.make_training_n_test_sets(features_key, class_main_key, class_name)
        if model == "RF": 
            self.random_forest_Regressor()
        elif model == "AB":
            self.Ada_Boost_Regressor()
        elif model == "GB":
            self.Gradient_Boosting_Regressor()
        elif model == "DT": 
            self.desision_tree_regressor()
        self.cross_val()

    def desision_tree_regressor(self, max_depth=5):
        
        self.clf = DecisionTreeRegressor(max_depth=max_depth)
        self.clf.fit(self.x_train, self.y_train) 
        y_pred = self.clf.predict(self.x_test)
        accuracy = np.mean(y_pred == self.y_test)
        score = self.clf.score(self.x_test, self.y_test) 
        print(f"Decision Tree Accuracy: {accuracy:.2f}")
        print(f"Decision Tree Score: {score:.2f}") 
        self.model_exists = True
        return self.clf, accuracy

    def Gradient_Boosting_Regressor(self, n_est = 100, Lr = 0.1): 
        self.clf = GradientBoostingRegressor(n_estimators=n_est, learning_rate = Lr)
        self.clf.fit(self.x_train, self.y_train) 
        y_pred = self.clf.predict(self.x_test)
        accuracy = np.mean(y_pred == self.y_test)
        score = self.clf.score(self.x_test, self.y_test) 
        print(f"Gradient Boosting Accuracy: {accuracy:.2f}")
        print(f"Gradient Boosting Score: {score:.2f}") 
        self.model_exists = True
        return self.clf, accuracy

    def Ada_Boost_Regressor(self, n_est = 100, Lr = 1.0):
        self.clf = AdaBoostRegressor(n_estimators=100, learning_rate=1.0)
        self.clf.fit(self.x_train, self.y_train) 
        y_pred = self.clf.predict(self.x_test)
        accuracy = np.mean(y_pred == self.y_test)
        score = self.clf.score(self.x_test, self.y_test) 
        print(f"Ada Boost Accuracy: {accuracy:.2f}")
        print(f"Ada Boost Score: {score:.2f}") 
        self.model_exists = True
        return self.clf, accuracy

    def random_forest_Regressor(self, n_est = 100):   
         #n_est =  number of trees in the forest.
        self.clf =  RandomForestRegressor(n_estimators = n_est)
        self.clf.fit(self.x_train, self.y_train) 
        y_pred = self.clf.predict(self.x_test)
        accuracy = np.mean(y_pred == self.y_test)
        score = self.clf.score(self.x_test, self.y_test) 
        #F value F1 = 2 * (precision * recall) / (precision + recall
        print(f"Random Forest Accuracy: {accuracy:.2f}")
        print(f"Random Forest Score: {score:.2f}") 
        self.model_exists = True
        return self.clf, accuracy
    
    #Cross-validation test the models performance. 
    def cross_val(self, n_splits=10):
        if self.model_exists: 
            scores = cross_val_score(self.clf, self.x_train, self.y_train, cv=n_splits) #CV no. folds 
            #returnes scores: ndarray of float of shape=(len(list(cv)),)
            #Array of scores of the estimator for each run of the cross validation.
            
            print(f"Cross-validation scores: {scores}")
            print(f"Mean cross-validation score: {np.mean(scores):.2f}")
        return scores 
    
    def confusion_matrix(self):
        if self.model_exists:
            y_pred = self.clf.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            return cm
        else:
            print("Model not trained yet.")
            return None

class C_PCA: 
    def __init__(self,df, dataDict): 
        self.dataDict = dataDict
        #datadict in form {ID: {Metadata: {}, Traits: {}, Hs_Traits: {}, Spectra: {}}}
        self.df = df
    
    



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



    reg = C_svr(dataDict)  
    reg.plot_svr_all_features(filenames_keys[3], filenames_keys[2])  
    
    
    
    """     
    # Prepare data for covariance analysis on trait
    Analysis = C_analysis(df, dataDict)  
    Analysis.perform_covariance_analysis(filenames_keys[3])  

    # Ploting spectra based off conditions
    sort_key = 'Conditions'
     
    dataDictSort = dictManager.separate_dict_by_value(sort_key, filenames_keys[1]) #separate the dict by genotype 
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
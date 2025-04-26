import matplotlib.pyplot as plt
import numpy as np
import os
#lib to read in csv files  
import pandas as pd
from PIL import Image  # Add this import for image processing
import math
import cv2 as cv
from scipy.stats import f_oneway

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_extraction import extract_data
from data_extraction import token_sample_extract
from data_extraction import count_samples_by_symbol 

# Set NumPy print options to display higher precision
np.set_printoptions(precision=10, suppress=False) 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Collect data from the CSV file~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def extract_reflectance_from_row(data, data_start, data_end, toggle_float_conversion=True): 
    ypoints = data.iloc[data_start:data_end].values   
    # convert from np.float to float
    if toggle_float_conversion is True:
        ypoints = np.array(ypoints, dtype=float) 
    return ypoints

def extract_wavelength(data, min_wavelength = 0, max_wavelength = 2500, data_start=1): 
    xpoints_data = list(data)
    xpoints = xpoints_data[data_start:]   
    if min_wavelength == 0: #all the data is used 
        min_wavelength_index = xpoints_data.index(xpoints[0]) #data_start 
        max_wavelength_index = xpoints_data.index(xpoints[-1])+1#len(xpoints_data) 
        return xpoints, min_wavelength_index, max_wavelength_index #-1 throws an error as their is no min_wave_index  
    else: #ensure the xpoints are above or equal to min_wavelength 
        xpoints_restricted_wavelength = [x for x in xpoints if x.isnumeric() and int(x) >= min_wavelength and int(x) <= max_wavelength]
        #count how far along data the wavelength is 
        min_wavelength_index = xpoints_data.index(xpoints_restricted_wavelength[0]) 
        max_wavelength_index = xpoints_data.index(xpoints_restricted_wavelength[-1])+1 
        return xpoints_restricted_wavelength, min_wavelength_index, max_wavelength_index 

def extract_class_coloum(data, sort_term='USDA Symbol', data_start=1):
    #extract the class coloum from the data frame 
    #and return it as a list
    class_coloum = data[sort_term].values.tolist()
    return class_coloum

def extract_reflectance(data, data_start, data_end):

    data = data.iloc[:, data_start:data_end]   

    return data 

def extract_class_coloum(data, sort_term='USDA Symbol'):
    if sort_term in data.columns:
        return data[sort_term].tolist() 
    else:
        raise ValueError(f"Column '{sort_term}' not found in the DataFrame.")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_wavelength(data, sort_term, min_wavelength = 0, max_wavelength = 2500, data_start=1):
    data = data.sort_values(by=[sort_term])
    #ensu the data is sorted by sort_term 

   #the x values are in the header of the data frame 
    xpoints, data_start, data_end = extract_wavelength(data, min_wavelength, max_wavelength, data_start) 
    #print(xpoints, data_start) 
    #for every row in data, add new USDA Symbol to a dic and count them if they aleard exist

    Symbol_sample_count = count_samples_by_symbol(data,sort_term) 


    # Ensure the output directory exists
    output_dir = './data/reflectance_v_wavelength/'
    os.makedirs(output_dir, exist_ok=True) 

    # Initialize variables for plotting
    last_symbol = None
    plt.figure()
    x_label = 'Wavelength (nm)'
    y_label = 'Reflectance'

    for index, row in data.iterrows(): 
        symbol = row[sort_term] 
        
        ypoints = extract_reflectance_from_row(row, data_start, data_end) 
 
        if symbol != last_symbol and last_symbol is not None:
            # Show the current figure and start a new one
            plt.title(f"{sort_term}: {last_symbol}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            #reduce the number of ticks on the x axis
            plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8)
            # display Symbol_sample_count(last_symbol) on the figure top left
            plt.text(0.75, 0.75, f"Samples: {Symbol_sample_count[last_symbol]}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
            
            #save plot as a png file
            plt.savefig(f'{output_dir}{last_symbol}_wavelenght_reflectance_plot.png')
            #plt.show() 
            plt.figure()   
            
        # Plot the data
        plt.plot(xpoints, ypoints, label=f"Sample {index}")
        last_symbol = symbol

    # Show the last figure
    if last_symbol is not None:
        plt.title(f"{sort_term}: {last_symbol}")
        plt.xlabel(x_label)
        plt.ylabel(y_label) 
        plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8) 
        plt.text(0.75, 0.75, f"Samples: {Symbol_sample_count[last_symbol]}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
        
        plt.savefig(f'{output_dir}{last_symbol}_wavelenght_reflectance_plot.png')
        #plt.show()  

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Data set generation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def matrix_to_image(matrix, output_path, mode='L'): #, mode_type='L'
    #reshape matrix data into 2D grey image     
    # Normalize the matrix values to the range [0, 255]
    normalized_matrix = cv.normalize(matrix, None, 0, 255, cv.NORM_MINMAX) 
    normalized_matrix = normalized_matrix.astype(np.uint8)  # Convert to unsigned 8-bit integer
    # Create an image from the normalized matrix
    image = Image.fromarray(normalized_matrix, mode)  # 'L' mode is for grayscale images
    # Save the image as a PNG file
    image.save(output_path)
    #print("Image saved at", output_path)
    return output_path

#ID allows multiple images to be created with different names loop over create_image_from_data 
def convert_1d_arr_to_2d_matrix(arr_np):

    len_arr = arr_np.size
    #print("len_arr", len_arr)  
    # check if the len is a perfrect square, if it is store the square root in a variable
    if math.sqrt(len_arr) % 1 == 0:
        square_root_len = int(math.sqrt(len_arr))
    else:
        print("len_arr is not a perfect square, taking lengths away from the end of the array until it is a perfect square")
        #takes lenghts away from the end of the array until it is a perfect square 
        count_len_taken = 0
        while math.sqrt(len_arr) % 1 != 0: 
            arr_np = arr_np[:-1]
            len_arr = arr_np.size
            count_len_taken += 1
        square_root_len = int(math.sqrt(len_arr)) 
        print("len_arr is a perfect square, after taking away", count_len_taken, "lengths from the end of the array")
        print("square of len_arr is", square_root_len) 
        len_arr = arr_np.size
        print("len_arr", len_arr)  

    # Reshape the 1D data into a 2D square array  
    reshaped_matrix = arr_np.reshape((square_root_len, square_root_len))

    return reshaped_matrix  


def dataset_genration(data, min_wavelength, max_wavelength,  data_start = 1,  sort_term = 'USDA Symbol'): 
    xpoints, data_start, data_end = extract_wavelength(data, min_wavelength, max_wavelength, data_start) 

    sample_counter = 0 
    last_symbol = None

    for index, row in data.iterrows():
        symbol = row[sort_term]   
        if index is 0:
            output_dir = f'./data/{symbol}_dataset/'
            os.makedirs(output_dir, exist_ok=True) 
        
        ypoints_1d = extract_reflectance_from_row(row, data_start, data_end) 
        
        #convert ypoints to float
        ypoints_1d_refined_np = np.array(ypoints_1d, dtype=float)
        #convert to 2D matrix
        ypoints_2d_refined_np = convert_1d_arr_to_2d_matrix(ypoints_1d_refined_np) 


        #if the symbol changes from the last reset sample counter
        if symbol != last_symbol and last_symbol is not None:
            sample_counter = 0
            output_dir = f'./data/{symbol}_dataset/'
            os.makedirs(output_dir, exist_ok=True)
         
        #conver matrix to image
        #save a a png file at data/{symbol}/{symbol}_{sample_counter}.png
        matrix_to_image(ypoints_2d_refined_np, f"{output_dir}{symbol}_{sample_counter}.png") 


        last_symbol = symbol
        sample_counter += 1
        #break #used to test and create only 1 image  
    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ statstic analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def class_data_dict_gen(data,  min_wavelength = 350, max_wavelength = 2500,  data_start = 1,  sort_term = 'USDA Symbol', toggle_float_conversion = True, toggle_norm = False):
     
    data = data.sort_values(by=[sort_term])
    token_Symbol_sample_no_dic =  count_samples_by_symbol(data, sort_term) 
    #print(token_Symbol_sample_no_dic)   

    x_points, data_start, data_end = extract_wavelength(data, min_wavelength, max_wavelength, data_start)

    data_len = data_end - data_start 
     
    data_matrix = []

    #make a dic for each token_Symbol_for_analysis
    token_symbol_of_matixs = {}
    last_symbol = None
    overall_index = 0
    for index, row in data.iterrows():
        symbol = row[sort_term] 
        if symbol != last_symbol and last_symbol is not None:

            #add matrix to dic
            token_symbol_of_matixs[last_symbol] = data_matrix 

            #reset matrix
            data_matrix = np.zeros((token_Symbol_sample_no_dic[symbol],  data_len))  
            overall_index = index  

        if last_symbol is None:
            #reduce the  matrix to the size of token_Symbol_sample_no_dic[last_symbol]
            data_matrix = np.zeros((token_Symbol_sample_no_dic[symbol], data_len)) 

        # Ensure xpoints and ypoints are 1D arrays for plotting
        y_points = extract_reflectance_from_row(row, data_start, data_end)  

        # Ensure ypoints has exactly data_len and evenly spaced the data points thru the data set 
        y_points = np.linspace(y_points[0], y_points[-1], data_len)
        
        #convert ypoints to float
        #if toggle_float_conversion is True:
            #ypoints = np.array(ypoints, dtype=float)
        
        #normalize ypoints
        #if toggle_norm is True:
            #ypoints = ((ypoints - np.min(ypoints)) / (np.max(ypoints) - np.min(ypoints)))
        
        #add ypoints to matrix
        data_matrix[index-overall_index] = y_points 
        
        last_symbol = symbol
    return token_symbol_of_matixs, np.array(x_points, dtype=float)   
        

def find_mean_of_matrix_row(matrix):
    matrix = matrix.transpose() 
    #find the mean of each row, then add to new arr  
    mean_arr = []
    for row in matrix: 
        #calc the mean
        mean_arr.append(np.mean(row))
    mean_arr = np.array(mean_arr, dtype=float)   
    return mean_arr

def find_std_of_matrix_row(matrix):
    matrix = matrix.transpose() 
    #std = standard deviation 
    #find the std of each row, then add to new arr
    std_arr = []
    for row in matrix:  
        #calc std
        std_arr(np.std(row)) 

    return std_arr
def find_anova_1_way_p_f(matrix):
    #dont transposet the matrix
    #f_value, p_value = f_oneway()

     
    return None

def statsitic_analysis(data,  min_wavelength = 350, max_wavelength = 2500,  data_start = 1,  sort_term = 'USDA Symbol'):
    class_data_dict, wavelengths = class_data_dict_gen(data, min_wavelength, max_wavelength, data_start, sort_term)

    #transpose data
    #print(wavelengths.transpose()) 
    #print("Shape of the matrix:", wavelengths.shape) 
    matrix_output = {}

    
    # Ensure the output directory exists
    output_dir = './data/analysis/'
    os.makedirs(output_dir, exist_ok=True) 

    for key, matrix in class_data_dict.items():
        
        
        #print(f"{class_lable}:", matrix) 
        #print("Shape of the matrix:", matrix.shape) 
        #do the action
        #arr_f, arr_p = find_anova_1_way_p_f_value_of_row(matrix)
        #revert back add to new dict
        #print("arr p:",arr_p)
        #matrix_output[key] = arr_p
        """
        #plot
        plt.figure()  
        plt.title(f"{sort_term}: {key}") 
        plt.xlabel("wavelengths")
        plt.ylabel("P_vlaue")
            #reduce the number of ticks on the x axis
        plt.xticks(np.arange(0, len(wavelengths), len(wavelengths)/10), fontsize=8)
        plt.plot(wavelengths, matrix_output[key] , label=f"Sample {key}")
        plt.show() 
        plt.savefig(f'{output_dir}{key}_mean_wavelenght_reflectance_plot.png') 
        """
        break  


    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~miscellaneous~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_class_from_image_name(image_name):
    # Extract the class from the image name
    # Assuming the format is like 'class_name_random_dataset_sample_size_45_square_map_ID1.png'
    parts = image_name.split('_')
    if len(parts) > 0:
        class_name = parts[0]
        return class_name 
    return None

def main(): 
    #extract_data()    
    path = './data/HS_data_for_analysis.csv'  
    data = pd.read_csv(path)  
    #sort data by USDA Symbol
    data = data.sort_values(by=['USDA Symbol'])      
    #plot_wavelength(data,'USDA Symbol', 400, 2424) 
    dataset_genration(data, 400, 2424)

    #statsitic_analysis(data, 400, 2424)     

     
    

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~old code~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""




def find_mean_of_matrix_col(matrix):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")

    # Calculate the mean of each column
    column_means = np.mean(matrix, axis=0)
    return column_means

def find_std_of_matrix_col(matrix):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")

    # Calculate the standard deviation of each column
    column_stds = np.std(matrix, axis=0) 
    return column_stds

def data_analysis(data, wavelength_min = 300, wavelenght_max = 2500, sort_term='USDA Symbol', data_start=1):
    class_data_dict = create_data_dict_by_class(data, wavelength_min, wavelenght_max, sort_term, data_start)

    class_data_dict_mean_std = {}
    for each_class_key, value in class_data_dict.items(): 
        mean = find_mean_of_matrix_col(value) 
        std = find_std_of_matrix_col(value)
        class_data_dict_mean_std[each_class_key] = (mean, std)  
    #print(class_data_dict_mean_std) 
    xpoints = np.linspace(wavelength_min, wavelenght_max, len(next(iter(class_data_dict_mean_std.values()))[0]))
  
     # Plot the mean reflectance for each class
    for class_label, (mean, std) in class_data_dict_mean_std.items():
        plt.plot(xpoints, mean, label=f"Class: {class_label}")

    # Configure the plot
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Mean Reflectance')
    plt.title('Mean Reflectance vs Wavelength for Each Class')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    output_dir = './data/mean_reflectance_plots/'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f'{output_dir}mean_reflectance_plot.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Mean reflectance plot saved at {plot_path}")





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Anova~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Anova_feature_selection(data): 
    #data: header, wavelenght 300-2500
    #      class , reflectance for each wavelength

    x_points, data_start, data_end = extract_wavelength(data, 400, 2424) 
    y_points = extract_reflectance(data, data_start, data_end)
    class_col = extract_class_coloum(data) 

     # Group reflectance data by class
    grouped_data_dict = {}
    for label, row in zip(class_col, y_points.values):
        if label not in grouped_data_dict:
            grouped_data_dict[label] = []
        grouped_data_dict[label].append(row)

    # Perform ANOVA for each wavelength
    significant_features_dict = {}
    for i, wavelength in enumerate(x_points):
        # Collect reflectance values for this wavelength across all classes
        wavelength_data = [np.array(group)[:, i] for group in grouped_data_dict.values()]
        
        # Perform one-way ANOVA
        f_value, p_value = f_oneway(*wavelength_data)
        
        # Store results for each class
        for class_label in grouped_data_dict.keys(): 
            if class_label not in significant_features_dict:
                significant_features_dict[class_label] = []
            significant_features_dict[class_label].append((wavelength, p_value, f_value))

    # Sort results by p-value for each class
    for class_label in significant_features_dict:
        significant_features_dict[class_label].sort(key=lambda x: x[1])  # Sort by p-value (ascending)

    return significant_features_dict

def plot_Anova(significant_features):
    # Create an output directory for the plots
    output_dir = './data/anova_plots/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot p-values for each class
    for class_label, features in significant_features.items():
        wavelengths = [f[0] for f in features]
        p_values = [f[1] for f in features]
        plt.plot(wavelengths, p_values, label=f"Class: {class_label}")

    # Configure the plot
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('p-value')
    plt.title('ANOVA p-values for Wavelengths by Class')
    plt.yscale('log')  # Use a logarithmic scale for better visualization of small p-values
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_path = f'{output_dir}anova_p_values_line_plot.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Line plot saved at {plot_path}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PCA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def principal_component_analysis(xpoints, ypoints, n_components = 45):
 
    # if we want to know if the wavelegnth is significant in the data set
    # then we need to do PCA on the data set where in  
    # xpoints are reflectance values and ypoints are the class labels

    x_train, x_test, y_train, y_test = train_test_split(xpoints, ypoints, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train) 

    pca = PCA(n_components)  
    x_pca = pca.fit_transform(x_train) 
    #print(pca1.explained_variance_ratio_())   
    plt.plot(pca.explained_variance_ratio_) 
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance') 
    plt.savefig(f'./data/PCA_cumulative_explained_variance.png') 
    plt.show() 

    plt.bar(range(1, len(pca.explained_variance_)+1), pca.explained_variance_)
    plt.ylabel('explained variance')
    plt.xlabel('number of components')
    plt.plot(range(1,len(pca.explained_variance_)+1), 
             np.cumsum(pca.explained_variance_), c='red', label='cumulative explained variance')
    plt.legend(loc = 'upper left')
    plt.title('PCA explained variance') 
    #save plot as a png file
    plt.savefig(f'./data/PCA_explained_variance.png')
    plt.show() 

    # Analyze loadings
    loadings = pca.components_  # Shape: (n_components, n_features)
    wavelengths = extract_wavelength(pd.DataFrame(xpoints))  # Extract wavelength headers
    #print("Loadings shape:", loadings.shape)
    #print("loadings:", loadings) 
    # CV has 2096 rows inlcuding the header, and 2151 colums including the class row, 
    # Loadings shape: (n_components, 2151) (PC, features)   
    # thus the colums/wavelengths are the features. 

    #find the PC wavelengths that have the highest loadings



def create_images_from_data(data, ID, all = False, squre_size = 45, sort_term='USDA Symbol', data_start=1):
    
    class_samples = dataset_genration(data, all, squre_size, sort_term, data_start)

    for key, value in class_samples.items(): 
        #print("HERE")
        # Convert the matrix to an image and save it 
        if all is True:
            path = f"./data/{key}_whole_dataset_map.png"  
        else:
            path = f"./data/{key}_random_dataset_sample_size_{squre_size}_square_map_ID{ID}.png" 
        matrix_to_image(value, path)  
        print(f"Image saved for {key} at {path}")  
        #break #used to test and create only 1 image 
    return None

def create_even_spaced_data_set(data, sort_term='USDA Symbol', data_start=1, all = False, data_len=45, toggle_norm = False, toggle_float_conversion=True ): 
    data = data.sort_values(by=[sort_term])
    token_Symbol_sample_no_dic =  count_samples_by_symbol(data, sort_term) 
    print(token_Symbol_sample_no_dic)  
    #find len of data from data_start to end of data 
    if all is True:
        data_len = len(data.iloc[data_start:])
    else:
        if data_len > len(data.iloc[data_start:]):
            raise ValueError("data_len is greater than the number of data points") 
            return None
    
    #create a matrix of zeros with the same dimenatsion as data
    data_matrix = np.zeros((len(data), data_len)) 
     
    #make a dic for each token_Symbol_for_analysis
    token_symbol_of_matixs = {}
    last_symbol = None
    overall_index = 0
    for index, row in data.iterrows():

        symbol = row[sort_term] 
        if symbol != last_symbol and last_symbol is not None:
            
            
            #add matrix to dic
            token_symbol_of_matixs[last_symbol] = data_matrix 

            #reset matrix
            data_matrix = np.zeros((token_Symbol_sample_no_dic[symbol],  data_len))  
            overall_index = index  

        if last_symbol is None:
            #reduce the  matrix to the size of token_Symbol_sample_no_dic[last_symbol]
            data_matrix = np.zeros((token_Symbol_sample_no_dic[symbol], data_len)) 

        # Ensure xpoints and ypoints are 1D arrays for plotting
        ypoints = extract_reflectance_from_row(row, data_start) 

        # Ensure ypoints has exactly data_len and evenly spaced the data points thru the data set 
        ypoints = np.linspace(ypoints[0], ypoints[-1], data_len)
        
        #convert ypoints to float
        if toggle_float_conversion is True:
            ypoints = np.array(ypoints, dtype=float)
        
        #normalize ypoints
        if toggle_norm is True:
            ypoints = ((ypoints - np.min(ypoints)) / (np.max(ypoints) - np.min(ypoints)))
        
        #add ypoints to matrix
        data_matrix[index-overall_index] = ypoints 
        
        last_symbol = symbol

    return token_symbol_of_matixs 


def random_matrix_rows(class_matrix_dic, num_rows=45):
    #randomly select num_rows from each matrix in the dic and return a new dic
    random_matrix_dic = {}
    for key, value in class_matrix_dic.items():
        #randomly select num_rows from each matrix in the dic
        random_rows = np.random.choice(value.shape[0], num_rows, replace=False)
        #create a new matrix with the random rows 
        random_matrix_dic[key] = value[random_rows]
    return random_matrix_dic


def dataset_genration_2d(data, all = False, squre_size = 45, sort_term='USDA Symbol', data_start = 1):
    
    if all is True: 
        squre_size = len(data.iloc[data_start:])
    
    class_matrix_dic = create_even_spaced_data_set(data, sort_term, data_start, all, squre_size, toggle_norm = False)
    if all is False:
        random_matrix_rows_dic = random_matrix_rows(class_matrix_dic, num_rows = squre_size) 
        return random_matrix_rows_dic
    else:
        return class_matrix_dic    
    return None   

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~junk code~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #print( "[data_matrix]",data_matrix)   
            #print the last row of  data_matrix 

'''def select_points(ypoints, squre_size):
    # Ensure ypoints has exactly squre_size and evenly spaced the data points thru the data set 
    if len(ypoints) > squre_size:
        ypoints = np.linspace(ypoints[0], ypoints[-1], squre_size)
    elif len(ypoints) < squre_size:
        raise ValueError("ypoints is less than squre_size")
    return ypoints'''

"""
            print( "[data_matrix_LAST]",data_matrix[-1,:]) 
            print("LAST == last entered", ypoints == data_matrix[-1,:]) 
            print( "[data_matrix_FIRST]",data_matrix[0,:]) 
            print("FIRST == first entered", ypoints == data_matrix[0,:])
            print("[token_symbol_of_matixs]", token_symbol_of_matixs[last_symbol]) """


"""
def matrix_to_image(matrix, output_path): #, mode_type='L'
    #reshape matrix data into 2D grey image
     # Normalize the matrix values to the range [0, 255]
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255
    normalized_matrix = normalized_matrix.astype(np.uint8)  # Convert to unsigned 8-bit integer

    # Create an image from the normalized matrix
    image = Image.fromarray(normalized_matrix, mode='L')  # 'L' mode is for grayscale images

    # Save the image as a PNG file
    image.save(output_path)
    
   

def create_image_from_data(data, sort_term='USDA Symbol', data_start=11):
    data = data.sort_values(by=[sort_term])
    class_matrix_dic = create_rand_data_set(data, sort_term, data_start)
    for key, value in class_matrix_dic.items():
        # Convert the matrix to an image and save it 
        path = f"./data/{key}_whole_dataset_map.png"  
        matrix_to_image(value, path) 
        print(f"Image saved for {key} at {path}")  
        #display image
        """ 
        #img = cv2.imread(path) 
        #cv2.imshow(f"Image for {key}", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
""" 
        break  """



"""
    # Scale values from 0-1 to 0-255 for grayscale image
    scaled_matrix = (matrix * 255).astype(np.uint8)
    
    # Save the grayscale image
    save_img(output_path, scaled_matrix[..., np.newaxis]) """
    
    # Normalize the matrix values to the range [0, 255]
"""normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255
    normalized_matrix = normalized_matrix.astype(np.uint8)  # Convert to unsigned 8-bit integer

    if mode_type == 'RGB':
        # Convert 2D matrix to 3D by duplicating the matrix across 3 channels 
        wr = 0.9
        wg = 0.5
        wb = 0.1 
        r_matrix = normalized_matrix * wr
        b_matrix    = normalized_matrix * wb
        g_matrix = normalized_matrix * wg
        normalized_matrix = np.stack([r_matrix, g_matrix, b_matrix], axis=-1)  # Shape: (height, width, 3)
        #normalized_matrix = np.stack([normalized_matrix] * 3, axis=-1)  # Shape: (height, width, 3)

    # Create an image from the normalized matrix
    image = Image.fromarray(normalized_matrix, mode=mode_type)

    # Save the image as a PNG file
    image.save(output_path)"""
"""   
    #Convert a 1D reflectance matrix to a 2D grayscale image and save it.
    #:param matrix: The input 1D matrix (e.g., reflectance data).
    #:param output_path: The path to save the image.
   
    # Flatten the matrix to ensure it's 1D
    flattened_matrix = matrix.flatten()

    # Reshape the 1D data into a 45 × 45 2D array 
    reshaped_matrix = flattened_matrix[:2025].reshape((45, 45))

    # Scale the values from 0-1 to 0-255
    scaled_matrix = (reshaped_matrix * 255).astype(np.uint8)

    # Create a grayscale image from the scaled matrix
    image = Image.fromarray(scaled_matrix, mode='L')

    # Save the image as a PNG file
    image.save(output_path)"""
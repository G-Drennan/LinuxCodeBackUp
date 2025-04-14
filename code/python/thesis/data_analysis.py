import matplotlib.pyplot as plt
import numpy as np
import os
#lib to read in csv files 
import pandas as pd
from PIL import Image  # Add this import for image processing

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#import cv2

from data_extraction import extract_data
from data_extraction import token_sample_extract
from data_extraction import count_samples_by_symbol 

# Set NumPy print options to display higher precision
np.set_printoptions(precision=10, suppress=False) 

#gives the wavelength data from the header of the data frame
def extract_wavelength(data, data_start=1): 
    xpoints = list(data)
    xpoints = xpoints[data_start:] 
    return xpoints 

#usage:
#for index, row in data.iterrows():
    #    ypoints = extract_reflectance_from_row(row) 
    # gives the reflectance data from the row
def extract_reflectance_from_row(data, data_start=1, toggle_float_conversion=True): 
    ypoints = data.iloc[data_start:].values
    # convert from np.float to float
    if toggle_float_conversion is True:
        ypoints = np.array(ypoints, dtype=float)
    return ypoints

def extract_class_coloum(data, sort_term='USDA Symbol', data_start=1):
    #extract the class coloum from the data frame 
    #and return it as a list
    class_coloum = data[sort_term].values.tolist()
    return class_coloum

def extract_reflectance(data, data_start=1):
    #extract the data from the data frame 
    data = data.iloc[:, data_start:]  
    return data 

def plot_wavelength(data, sort_term, data_start=1):
    data = data.sort_values(by=[sort_term])
    #ensu the data is sorted by sort_term

   #the x values are in the header of the data frame 
    xpoints = extract_wavelength(data,data_start) 

    #for every row in data, add new USDA Symbol to a dic and count them if they aleard exist

    Symbol_sample_count = count_samples_by_symbol(data,sort_term) 

    # Initialize variables for plotting
    last_symbol = None
    plt.figure()

    for index, row in data.iterrows(): 
        symbol = row[sort_term] 
        
        ypoints = extract_reflectance_from_row(row, data_start) 

        if symbol != last_symbol and last_symbol is not None:
            # Show the current figure and start a new one
            plt.title(f"{sort_term}: {last_symbol}")
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            #reduce the number of ticks on the x axis
            plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8)
            # display Symbol_sample_count(last_symbol) on the figure top left
            plt.text(0.75, 0.75, f"Samples: {Symbol_sample_count[last_symbol]}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
            
            #save plot as a png file
            plt.savefig(f'./data/{last_symbol}_wavelenght_reflectance_plot.png')
            plt.show()
            plt.figure()  
            
        # Plot the data
        plt.plot(xpoints, ypoints, label=f"Sample {index}")
        last_symbol = symbol

    # Show the last figure
    if last_symbol is not None:
        plt.title(f"{sort_term}: {last_symbol}")
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.xticks(np.arange(0, len(xpoints), len(xpoints)/10), fontsize=8) 
        plt.text(0.75, 0.75, f"Samples: {Symbol_sample_count[last_symbol]}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
        
        plt.savefig(f'./data/{last_symbol}_wavelenght_reflectance_plot.png')
        plt.show() 


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

def matrix_to_image(matrix, output_path): #, mode_type='L'
    #reshape matrix data into 2D grey image     
    # Normalize the matrix values to the range [0, 255]
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255
    normalized_matrix = normalized_matrix.astype(np.uint8)  # Convert to unsigned 8-bit integer
    # Create an image from the normalized matrix
    image = Image.fromarray(normalized_matrix, mode='L')  # 'L' mode is for grayscale images
    # Save the image as a PNG file
    image.save(output_path)
    #print("Image saved at", output_path)

def create_image_from_data(data, all = False, squre_size = 45, sort_term='USDA Symbol', data_start=1):
    
    random_matrix_rows_dic = dataset_genration_2d(data, all, squre_size, sort_term, data_start)
    for key, value in random_matrix_rows_dic.items(): 
        #print("HERE")
        # Convert the matrix to an image and save it 
        if all is True:
            path = f"./data/{key}_whole_dataset_map.png"  
        else:
            path = f"./data/{key}_random_dataset_sample_size_{squre_size}_square_map.png" 
        matrix_to_image(value, path)  
        print(f"Image saved for {key} at {path}")  
        #break #used to test and create only 1 image 

def main():
    #extract_data()   
    path = './data/HS_data_for_analysis.csv' 
    data = pd.read_csv(path)  
    #sort data by USDA Symbol
    data = data.sort_values(by=['USDA Symbol'])      

    principal_component_analysis(extract_reflectance(data).to_numpy(), extract_class_coloum(data), 20)     

    #create_image_from_data(data, True)   
    #plot_wavelength(data,'USDA Symbol')    
    #dataset_genration(data)  



if __name__ == "__main__":
    main()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~junk code~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #print( "[data_matrix]",data_matrix)   
            #print the last row of  data_matrix 
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
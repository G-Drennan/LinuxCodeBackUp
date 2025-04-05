import matplotlib.pyplot as plt
import numpy as np
#lib to read in csv files 
import pandas as pd
from PIL import Image  # Add this import for image processing

#import cv2

from data_extraction import extract_data
from data_extraction import token_sample_extract
from data_extraction import count_samples_by_symbol 


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

def create_data_set(data, sort_term = 'USDA Symbol', data_start=11):
    
    token_Symbol_100, token_Symbol_for_analysis = token_sample_extract(count_samples_by_symbol(data, sort_term))
    print("[TOKEN SYMBOL 100]", token_Symbol_for_analysis)
    
    #find len of data from data_start to end of data
    data_len = len(data.columns)-data_start  
    #print("[DATA LEN]", data_len)
    
    #create a matrix of zeros with the same dimenatsion as data
    data_matrix = np.zeros((len(data), data_len))
    
    #make a dic for each token_Symbol_for_analysis
    token_symbol_of_matixs = {}
    last_symbol = None
    overall_index = 0
    for index, row in data.iterrows():
        symbol = row[sort_term] 
        # Ensure xpoints and ypoints are 1D arrays for plotting
        ypoints = extract_ypoints(row, True, data_start)
        #add ypoints to a matrix
        data_matrix[index-overall_index] = ypoints
        #print lenfghyt ot ypoints
        if symbol != last_symbol and last_symbol is not None:
            #add matrix to dic
            token_symbol_of_matixs[last_symbol] = data_matrix

            #reset matrix
            data_matrix = np.zeros((len(data), data_len))  
            overall_index = index
        last_symbol = symbol
    
    #print dic
    #print("[TOKEN SYMBOL DIC]", _of_mat)

    return token_symbol_of_matixs 


def matrix_to_image(matrix, output_path, mode_type='L'):
    # Normalize the matrix values to the range [0, 255]
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255
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
    image.save(output_path)

def create_image_from_data(data, sort_term='USDA Symbol', data_start=11):
    class_matrix_dic = create_data_set(data, sort_term, data_start)
    for key, value in class_matrix_dic.items():
        # Convert the matrix to an image and save it 
        path = f"./data/{key}_whole_dataset_map.png"  
        matrix_to_image(value, path, 'RGB')  # Save as RGB image 
        print(f"Image saved for {key} at {path}")  
        #display image
        """ 
        img = cv2.imread(path) 
        cv2.imshow(f"Image for {key}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        """ 
        break  

def main():
    #extract_data() 
    path = './data/HS_data_for_analysis.csv' 
    data = pd.read_csv(path) 
    #sort data by USDA Symbol
    data = data.sort_values(by=['USDA Symbol'])
   #plot_wavelength(data,'USDA Symbol') 
    create_image_from_data(data, 'USDA Symbol') 



if __name__ == "__main__":
    main()
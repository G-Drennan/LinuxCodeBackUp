import numpy as np
from sklearn.covariance import EmpiricalCovariance

        #make a matrix plot of the covariance matrix
import matplotlib.pyplot as plt
import seaborn as sns

#TODO: sort into test and training data sets 40% - 60% split. Keep consistance for mulit differnt combinations of x and y data. 


class C_Regression:
    def __init__(self):
        pass

    def 
    def make_training_n_test_sets():


#~~~~~~~~~~~~~~~~~~~~~ junk code ~~~~~~~~~~~~~~~~~~~~~~~ 
"""
class C_Covariance_analysis:
    def __init__(self, init_df, init_dict, samples=[], features=[]):
        self.init_df = init_df
        self.init_dict = init_dict
        self.samples_features = (samples, features)
    def update_samples_features(self, samples, features):
        self.samples_features = (samples, features)
    
    def perform_covariance_analysis(self, samples_name = 'Samples', features_name = 'Features'):
        cov = EmpiricalCovariance().fit(self.samples_features)

        sns.heatmap(cov.covariance_, annot=True, cmap='coolwarm', square=True)
        plt.title('Covariance Matrix')
        plt.x_label(samples_name)
        plt.y_label(features_name) 
        plt.show()
"""
"""
samples = [1,2,3,4,5,6]
features = [2,4,6,8, 10, 12]

X = (samples,features)

cov = EmpiricalCovariance().fit(X)
print(cov.covariance_)
print(cov.location_) 

#make a matrix plot of the covariance matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cov.covariance_, annot=True, cmap='coolwarm', square=True)
plt.title('Covariance Matrix')
plt.show() """
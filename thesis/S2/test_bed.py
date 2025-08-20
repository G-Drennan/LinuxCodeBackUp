import numpy as np
from sklearn.covariance import EmpiricalCovariance

        #make a matrix plot of the covariance matrix
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["m", "c", "g"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="{} support vectors".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()

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
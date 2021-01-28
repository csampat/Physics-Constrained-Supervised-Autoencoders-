from numpy import unique
from numpy import where
from numpy import multiply, divide
from numpy import array
from numpy import isnan, logical_not
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


def main():
    #import data
    data = pd.read_csv('dataFiles/test_data1_WithOutputs_withCalculatedValues.csv')
    ls = data['L/S Ratio']
    torque = data['DetTorque']
    MRT = data['DetMRT']
    extent = data['Extent of gran']
    relvis = data['Binder Viscosity (mPa.s)'] / 0.001
    fill = data['Calc Fill level']
    nKE = data['n KE']
    torque = data['DetTorque']
    fillVol = data['Calc Fill volume'] / 1e9
    plotArray = array([multiply(ls,relvis), fill]).T
    plotArray_stress = array([multiply(ls,relvis), divide(torque,fillVol)]).T
    #clustering 1
    X1 = array([ls,relvis, fill]).T
    cluster1 = 'L/S, Vis, Fill'
    yhat1,model1  = kmeansClustering(5,X1,X1)
    plot_cluster_kmeans(yhat1,plotArray,cluster1)

    #clustering 2
    X2 = array([torque,MRT,extent,nKE]).T
    cluster2 = 'Torque,MRT,extent,nKE'
    yhat2,model2  = kmeansClustering(5,X2,X2)
    plot_cluster_kmeans(yhat2,plotArray,cluster2)

    #clustering 3
    fines = data['Fines %'] / 100
    X3 = array([fines,extent]).T
    X3na = X3[logical_not(isnan(X3))].reshape(-1,2)
    print(X3na.shape)
    cluster3 = 'fines,extent'
    yhat3,model3  = kmeansClustering(5,X3na,X3na)
    plot_cluster_kmeans(yhat3,plotArray,cluster3)

    plt.show()

# K-means clustering model
def kmeansClustering(n_clusters,InputArray,PredictArray):
# create model
    model = KMeans(n_clusters=n_clusters)
    model.fit(InputArray)
    yhat = model.predict(PredictArray)
    return yhat, model

def plot_cluster_kmeans(yhat,plotArray,title):
    clusters = unique(yhat)
    plt.figure()
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(plotArray[row_ix, 0], plotArray[row_ix, 1], label=cluster)
    plt.legend()
    # show the plot
    #plt.ylim(0,1)
    #plt.ylim([1e-4,1])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([1e-1,1000])
    plt.xticks([1e-1,1e0,1e1,1e2])
    plt.title(title)





if __name__ =="__main__":
    main()

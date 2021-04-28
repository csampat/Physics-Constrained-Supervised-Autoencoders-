import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from .DataCompletionMethods import DataCompletionMethods
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, PCA


""""
This class may be used to create PLS and PCA models for the reduced order
latent variable plotting
""""

class ReducedorderModels:
    #defining init method

    def __init__(self,dataFile, completedFileFlag, fractionsplit=0.8, ran_state=42):
        self.dataFile = dataFile
        if not completedFileFlag:
            self.dataFile = self.completeFile(self.dataFile)

    
    # PCA

    def simplePCA(self,n_comps,train_dataset,exp_traindataset,plot_flag=True):
        pca_obj = PCA(n_components=n_comps)
        prinComps = pca_obj.fit_transform(train_dataset)

        if(plot_flag):
            fig,ax = plt.subplots(1,2,subplot_kw=dict(box_aspect=1),
                sharey=True)
            exps = exp_traindataset
            for g in np.unique(exps):
                i = np.where(exps==g)
                ax[0].scatter(prinComps[i,0],prinComps[i,1],label=g)
            ax[0].set_xlabel('LV 1')
            ax[0].set_ylabel('LV 2')
            ax[0].set_title('According to exp.')
            ax[0].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                    ncol=2, mode="expand", borderaxespad=0.)
            ranges = [0,15,20,25,30,35,50]
            train_dataset_pca = train_dataset.copy()
            train_dataset_pca['Extent of gran'] = extent
            train_dataset_pca['LV 1'] = prinComps[:,0]
            train_dataset_pca['LV 2'] = prinComps[:,1]
            groups = train_dataset_pca.groupby(pd.cut(train_dataset_pca['Extent of gran'],ranges))
            for val, group in groups:
                ax[1].scatter(group['LV 1'],group['LV 2'],label=val)
            ax[1].set_xlabel('LV 1')
            ax[1].set_ylabel('LV 2')
            plt.title('Extent of granulation')
            fig.set_size_inches(10,5)
            fig.suptitle('PCA latent variables',fontsize=16)
            ax[1].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0)




    # PLS 

    def simplePLS(self,n_comps,train_data,train_label,test_data,test_label,exp_traindataset,scale_flag=True,plot_flag=False):
        pls_obj = PLSRegression(n_components=n_comps,scale=scale_flag)
        pls_obj.fit(train_data,train_label)
        print(np.round(pls_obj.coef_,3))
        pls_obj.predict(test_data)
        pls_obj.score(test_data,test_label)

        latent_variables = pls_obj.transform(train_data)

        if(plot_flag):
            extent = np.divide(train_label['final d50'],train_data['Initial d50'])
            exps = exp_traindataset
            fig,ax = plt.subplots(1,2,subplot_kw=dict(box_aspect=1),
                sharey=True)
            for g in np.unique(exps):
                i = np.where(exps==g)
                ax[0].scatter(latent_variables[i,0],latent_variables[i,1],label=g)
            ax[0].set_xlabel('LV 1')
            ax[0].set_ylabel('LV 2')
            ax[0].set_title('According to exp.')
            ax[0].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                    ncol=2, mode="expand", borderaxespad=0.)
            ranges = [0,15,20,25,30,35,50]
            train_dataset_pls = train_data.copy()
            train_dataset_pls['Extent of gran'] = extent
            train_dataset_pls['LV 1'] = latent_variables[:,0]
            train_dataset_pls['LV 2'] = latent_variables[:,1]
            groups = train_dataset_pls.groupby(pd.cut(train_dataset_pls['Extent of gran'],ranges))
            for val, group in groups:
                ax[1].scatter(group['LV 1'],group['LV 2'],label=val)
            ax[1].set_xlabel('LV 1')
            ax[1].set_ylabel('LV 2')
            ax[1].set_xlim([-5,5])
            ax[1].set_ylim([-5,5])
            plt.title('Extent of granulation')
            fig.set_size_inches(10,5)
            fig.suptitle('PLS latent variables',fontsize=16)
            ax[1].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                    ncol=2, mode="expand", borderaxespad=0)



    
    
    
    
    # data completion definition

    def completeFile(self, uncompletedFile):
        dcm_object = DataCompletionMethods(uncompletedFile)
        dataFile_DettorqueMRT = dcm_object.lineaeRegressionModel(['Torque','MRT'],True)
        uncompletedFile[['DetTorque','DetMRT']] = dataFile_DettorqueMRT
        dataFile_RantorqueMRT = dcm_object.lineaeRegressionModel(['Torque','MRT'],True)
        uncompletedFile[['RanTorque','RanMRT']] = dataFile_RantorqueMRT
        ranMRT = uncompletedFile['RanMRT']
        FillLevel = dcm_object.fillLevel_osorio(ranMRT)
        dataFile_complete = uncompletedFile
        return dataFile_complete
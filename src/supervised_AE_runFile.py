import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HelperTools.DataCompletionMethods import DataCompletionMethods
from HelperTools.SupervisedAutoencoder import SupervisedAutoencoder
from HelperTools.plotterClass import PlotterClass

from SALib.sample import saltelli
from SALib.analyze import sobol

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, DBSCAN
from keras.optimizers import Adam, Adadelta, SGD, Nadam, RMSprop, Adagrad
import os
import json


def main():
    # get datafile
    fileLoc = 'dataFiles/pls_200points_MRT_torqueAdded.csv'
    cwd = os.getcwd()
    dataFile_original = pd.read_csv(os.path.join(cwd,fileLoc))

    #  define hyperparameters for the Autoencoder
    hyperparameters = {
    "input_layer_nodes"      : 16,
    "out_layer_nodes_unsup"  : 16,
    "out_layer_nodes_sup"    : 3,
    "nodes_predec_layer"     : 4,
    "nodes_lv_layer"         : 1, 
    "nodes_enc_layer"        : 4,
    "nodes_pp_enc_layer"     : 4,
    "nodes_geo_enc_layer"    : 4,
    "nodes_mat_enc_layer"    : 4,
    "nodes_dec_layers"       : 8,
    "nodes_predec_layer_1"   : 5,
    "nodes_predec_layer_2"   : 4,
    "nodes_predec_layer_3"   : 4,
    "nodes_predec_layer_4"   : 4,
    "inner_layer_actFcn"     : 'tanh',
    "encod_layer_actFcn"     : 'tanh',
    "decod_layer_actFcn"     : 'tanh',
    "decod_layer_actFcn_pre" : 'tanh',
    "output_layer_actFcn"    : 'relu',
    "sup_layer_actFcn"       : 'linear',
    "optimizer_sup"          : Adam,
    "optimizer_unsup"        : Adadelta,
    "learning_rate"          : 0.01,
    "loss_sup"               : 'mse',
    "loss_unsup"             : 'mse',
    "n_epochs"               : 2500,
    "shuffle_flag"           : False,
    "val_split"              : 0.33,
    "verbose"                : 2
    }

    #creating class objects
    sae_obj = SupervisedAutoencoder(hyperparameters,dataFile_original)
    plt_obj = PlotterClass()

    train_datafile = dataFile_original.sample(frac=0.8,random_state=42)
    test_datafile = dataFile_original.drop(train_datafile.index)
    sae1_name = 'Supervised Autoencoder'
    sae2_pc_name = 'Supervised Physics Constrained Autoencoder'

    # mrt_lim = np.abs(sae_obj.calculatingLimits())
    # print(mrt_lim)
    labels = ['DetTorque','RanMRT','final d50']
    x_headers = ['RPM','L/S Ratio','FlowRate (kg/hr)', 'Temperature','Initial d50','Binder Viscosity (mPa.s)','Flowability (HR)','Bulk Density','nCE','Granulator diameter (mm)','L/D Ratio','SA of KE','nKE','Liq add position','nKZ','dKZ']

    history_3lv_unc, sae_3lv_unc, hr_3lv_unc = sae_obj.sup_3lv_ae_2hiddenpre(train_datafile)
    
    sae_obj.parameters['learning_rate'] = 0.0075

    history_3lv_sae, sae_3lv, hr_3lv = sae_obj.sup_3lv_ae_2hiddenpre_physicsConstrained(train_datafile)
    
    plt_obj.history_plotter(history_3lv_unc,'loss','val_loss',sae1_name)

    plt_obj.history_plotter(history_3lv_sae,'loss','val_loss',sae2_pc_name)

    test_mat, test_pp, test_geo, test_all, test_labels, mms_pp_test, \
        mms_geo_test, mms_mat_test, mms_all_test, mrt_lim, torque_lim, d50_lim \
             = sae_obj.dataPreprocessing_3lv(test_datafile)

    # getting prediction for reconstruction and prediction
    predictions_y_unc = sae_3lv_unc.predict([test_pp,test_mat,test_geo])

    predictions_y_AE = sae_3lv.predict([test_pp,test_mat,test_geo,mrt_lim, torque_lim, d50_lim])

    pred_con_y_unc = np.reshape(np.ravel(np.array(predictions_y_unc[1])),(len(test_datafile),3))
    pred_con_recons_unc = np.reshape(np.ravel(np.array(predictions_y_unc[0])),(len(test_datafile),16))

    pred_con_y = np.reshape(np.ravel(np.array(predictions_y_AE[1])),(len(test_datafile),3))
    pred_con_recons = np.reshape(np.ravel(np.array(predictions_y_AE[0])),(len(test_datafile),16))
    
    # parity plots for prediction

    plt.figure()
    parityPlot(np.ravel(predictions_y_unc[1]),test_labels.values,'SAE prediction')
    plt.figure()
    parityPlot(np.ravel(predictions_y_unc[0]),test_all,'SAE reconstruction')


    plt.figure()
    parityPlot(np.ravel(predictions_y_AE[1]),test_labels.values,'PCSAE prediction')
    plt.figure()
    parityPlot(np.ravel(predictions_y_AE[0]),test_all,'PCSAE reconstruction')

    r2_ypre_unc = sae_obj.calculateR2(np.ravel(predictions_y_unc[1]),test_labels.values,labels)
    r2_recons_unc = sae_obj.calculateR2(np.ravel(predictions_y_unc[0]),test_all,x_headers)

    r2_ypre = sae_obj.calculateR2(np.ravel(predictions_y_AE[1]),test_labels.values,labels)
    r2_recons = sae_obj.calculateR2(np.ravel(predictions_y_AE[0]),test_all,x_headers)

    hrall_mat, hrall_pp, hrall_geo, hrall_all, hrall_labels, mms_pp_hrall, \
        mms_geo_hrall, mms_mat_hrall, mms_all_hrall, mrt_lim_all, torque_lim_all, d50_lim_all \
             = sae_obj.dataPreprocessing_3lv(dataFile_original)

    latent_rep = np.reshape(np.array(hr_3lv.predict([hrall_pp,hrall_mat,hrall_geo,mrt_lim_all, torque_lim_all, d50_lim_all])).T,(len(dataFile_original),3))

    latent_rep_unc = np.reshape(np.array(hr_3lv_unc.predict([hrall_pp,hrall_mat,hrall_geo])).T,(len(dataFile_original),3))

    extent = np.divide(dataFile_original['final d50'],dataFile_original['Initial d50'])
    dataFile_original['Extent of Granulation'] = extent

    ranges = [0,15,20,25,30,35,50]
    titles = ["Acc to Exp","Acc to Regime","Acc to extent"]
    i_pp = 0
    i_mat = 1
    i_geo = 2
    label_pp = 'PP'
    label_mat = 'Mat'
    label_geo = 'Geo'

    # sae_3lv_unc.save('./SAEmodel')
    # sae_3lv.save('./PCSAEmodel')

    plot_all = sae_obj.plot_all3_range1_un2(latent_rep,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_pp,label_mat,i_pp,i_mat)
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_geo,label_mat,i_geo,i_mat)
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_pp,label_geo,i_pp,i_geo)


    plot_all = sae_obj.plot_all3_range1_un2(latent_rep_unc,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_pp,label_mat,i_pp,i_mat)
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep_unc,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_geo,label_mat,i_geo,i_mat)
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep_unc,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_pp,label_geo,i_pp,i_geo)
    
    print("\n--------- Performance--------\n")
    print("R^2 for ",sae1_name ," prediction     = ",r2_ypre_unc)
    print("R^2 for ",sae1_name ," reconstruction = ",r2_recons_unc)

    print("R^2 for ",sae2_pc_name ," prediction     = ",r2_ypre)
    print("R^2 for ",sae2_pc_name ," reconstruction = ",r2_recons)



    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    # print(latent_rep.shape)
    regs = dataFile_original['Regime']
    # print(len_TrainData)
    for g in np.unique(regs):
        i = np.where(regs==g)
        ax.scatter3D(latent_rep[i,0],latent_rep[i,1],latent_rep[i,2],label=g)
    ax.set_xlabel('PP')
    ax.set_ylabel('Mat.')
    ax.set_zlabel('Geo')
    ax.set_title('According to Regime')
    ax.legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
    ax.view_init(-45, 30)
    headers_pp  = ['RPM','L/S Ratio','FlowRate', 'Temperature']
    headers_mat = ['Initial d50','Binder Viscosity','Flowability','Bulk Density']
    headers_geo = ['nCE','G. diameter','L/D Ratio','SA of KE','nKE','Liq add','nKZ','dKZ']
    
    cluster_kmeans_3d(dataFile_original,latent_rep,'Physics constrained model')
    
    samplesize_pp = {
    'num_vars': 4,
    'names': headers_pp,
    'bounds': [[50, 1000],
               [0, 1.22],
               [0.04, 25],
               [25, 45]
            ]}

    samplesize_mat = {
    'num_vars': 4,
    'names': headers_mat,
    'bounds': [[27, 170],
               [0.001, 0.565],
               [1.2, 1.46],
               [290, 1602]
            ]}

    samplesize_geo = {
    'num_vars': 8,
    'names': headers_geo,
    'bounds': [[8, 38.5],
               [11, 25],
               [20, 40],
               [0, 90],
               [0, 16],
               [1, 3],
               [0, 2],
               [0, 24]
            ]}
    
    input_parameter_sensitivity_lv(hr_3lv_unc,[samplesize_pp,samplesize_mat,samplesize_geo],[headers_pp,headers_mat,headers_geo],False)

    input_parameter_sensitivity_lv_phycon(hr_3lv,[samplesize_pp,samplesize_mat,samplesize_geo],[headers_pp,headers_mat,headers_geo], mrt_lim_all, torque_lim_all, d50_lim_all,False)
    plt.show()


def cluster_kmeans_3d(datafile,latent_vars,title):
    clus_regimes_exp = np.unique(datafile['Regime'])
    n_clus_regimes = len(clus_regimes_exp)
    kmeans_cluster = KMeans(n_clusters=n_clus_regimes)
    kmeans_cluster.fit(latent_vars)
    y_pred = kmeans_cluster.predict(latent_vars)
    fitted_clusters = np.unique(y_pred)
    labels = kmeans_cluster.labels_
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig, rect=[0,0,1,1], elev=48,azim=134)
    for cluster in fitted_clusters:
	# get row indexes for samples with this cluster
        row_ix = np.where(y_pred == cluster)
	# create scatter of these samples
        ax.scatter(latent_vars[row_ix,0],latent_vars[row_ix,1],latent_vars[row_ix,2],label=cluster)
    ax.set_xlabel('PP LS')
    ax.set_ylabel('MAT LS')
    ax.set_zlabel('GEo LS')
    ax.set_title(title)
    ax.legend()



    



def input_parameter_sensitivity_lv(model,samplesize,dictName,dispflag):
    # value of N was decided due the output of the sample method, return N*(2D+2) array
    parameter_vals_pp = saltelli.sample(samplesize[0], 1800)
    parameter_vals_mat = saltelli.sample(samplesize[1], 1800)
    parameter_vals_geo = saltelli.sample(samplesize[2], 1000)
    minVals_pp  = parameter_vals_pp.min(axis=0)
    minVals_mat = parameter_vals_mat.min(axis=0)
    minVals_geo = parameter_vals_geo.min(axis=0)
    maxVals_pp  = parameter_vals_pp.max(axis=0)
    maxVals_mat = parameter_vals_mat.max(axis=0)
    maxVals_geo = parameter_vals_geo.max(axis=0)
    norm_paramVals_pp = np.zeros(parameter_vals_pp.shape)
    norm_paramVals_mat = np.zeros(parameter_vals_mat.shape)
    norm_paramVals_geo = np.zeros(parameter_vals_geo.shape)

    # normalizing inputs since the PCNN model takes only normalized inputs

    for i in range(samplesize[0]['num_vars']):
        for n in range(len(parameter_vals_pp)):
            norm_paramVals_pp[n,i] = (parameter_vals_pp[n,i] - samplesize[0]['bounds'][i][0]) /(samplesize[0]['bounds'][i][1] - samplesize[0]['bounds'][i][0])

    for i in range(samplesize[1]['num_vars']):
        for n in range(len(parameter_vals_mat)):
            norm_paramVals_mat[n,i] = (parameter_vals_mat[n,i] - samplesize[1]['bounds'][i][0]) /(samplesize[1]['bounds'][i][1] - samplesize[1]['bounds'][i][0])

    for i in range(samplesize[2]['num_vars']):
        for n in range(len(parameter_vals_geo)):
            norm_paramVals_geo[n,i] = (parameter_vals_geo[n,i] - samplesize[2]['bounds'][i][0]) /(samplesize[2]['bounds'][i][1] - samplesize[2]['bounds'][i][0])

    prediction_model = model.predict([norm_paramVals_pp,norm_paramVals_mat,norm_paramVals_geo])

    # if (index==0):
    #     prediction_model = model.predict([norm_paramVals,samplesize[1],samplesize[2]])
    # elif (index==1):
    #     prediction_model = model.predict([samplesize[0],norm_paramVals,samplesize[2]])
    # elif (index==2):
    #     prediction_model = model.predict([samplesize[0],samplesize[1],norm_paramVals])

    #Creating a sobol dict for Process parameters
    S_pp = sobol.analyze(samplesize[0],prediction_model[0].flatten())

    #Creating a sobol dict for Material properties
    S_mat = sobol.analyze(samplesize[1],prediction_model[1].flatten())

    #Creating a sobol dict for Geometry
    S_geo = sobol.analyze(samplesize[2],prediction_model[2].flatten())


    # y = json.dumps(S_pp)
    # y = json.dumps(S_mat)
    # y = json.dumps(S_geo)
    if(dispflag):
        print("\n-------S_pp------------\n",S_pp['S2'])
        print("\n-------S_mat------------\n",S_mat['S2'])
        print("\n-------S_geo------------\n",S_geo['S2'])

    


    plot_obj = PlotterClass()
    # Read on sobol indices here : https://uncertainpy.readthedocs.io/en/latest/theory/sa.html
    plot_obj.sensPlot_2(S_pp,samplesize[0]['names'],'Process Parameters Latent Space')
    plot_obj.sensPlot_2(S_mat,samplesize[1]['names'],'Material Properties Latent Space')
    plot_obj.sensPlot_2(S_geo,samplesize[2]['names'],'Geometry Latent Space')

def input_parameter_sensitivity_lv_phycon(model,samplesize,dictName,mrt_lim_all, torque_lim_all, d50_lim_all,dispflag):
    # value of N was decided due the output of the sample method, return N*(2D+2) array
    parameter_vals_pp = saltelli.sample(samplesize[0], 1800)
    parameter_vals_mat = saltelli.sample(samplesize[1], 1800)
    parameter_vals_geo = saltelli.sample(samplesize[2], 1000)
    minVals_pp  = parameter_vals_pp.min(axis=0)
    minVals_mat = parameter_vals_mat.min(axis=0)
    minVals_geo = parameter_vals_geo.min(axis=0)
    maxVals_pp  = parameter_vals_pp.max(axis=0)
    maxVals_mat = parameter_vals_mat.max(axis=0)
    maxVals_geo = parameter_vals_geo.max(axis=0)
    norm_paramVals_pp = np.zeros(parameter_vals_pp.shape)
    norm_paramVals_mat = np.zeros(parameter_vals_mat.shape)
    norm_paramVals_geo = np.zeros(parameter_vals_geo.shape)
    mrt_lim_all = np.zeros(parameter_vals_pp.shape[0])    
    torque_lim_all = np.zeros(parameter_vals_pp.shape[0])
    d50_lim_all = np.zeros(parameter_vals_pp.shape[0])

    # normalizing inputs since the PCNN model takes only normalized inputs

    for i in range(samplesize[0]['num_vars']):
        for n in range(len(parameter_vals_pp)):
            norm_paramVals_pp[n,i] = (parameter_vals_pp[n,i] - samplesize[0]['bounds'][i][0]) /(samplesize[0]['bounds'][i][1] - samplesize[0]['bounds'][i][0])

    for i in range(samplesize[1]['num_vars']):
        for n in range(len(parameter_vals_mat)):
            norm_paramVals_mat[n,i] = (parameter_vals_mat[n,i] - samplesize[1]['bounds'][i][0]) /(samplesize[1]['bounds'][i][1] - samplesize[1]['bounds'][i][0])

    for i in range(samplesize[2]['num_vars']):
        for n in range(len(parameter_vals_geo)):
            norm_paramVals_geo[n,i] = (parameter_vals_geo[n,i] - samplesize[2]['bounds'][i][0]) /(samplesize[2]['bounds'][i][1] - samplesize[2]['bounds'][i][0])

    prediction_model = model.predict([norm_paramVals_pp,norm_paramVals_mat,norm_paramVals_geo,mrt_lim_all, torque_lim_all, d50_lim_all])

    # if (index==0):
    #     prediction_model = model.predict([norm_paramVals,samplesize[1],samplesize[2]])
    # elif (index==1):
    #     prediction_model = model.predict([samplesize[0],norm_paramVals,samplesize[2]])
    # elif (index==2):
    #     prediction_model = model.predict([samplesize[0],samplesize[1],norm_paramVals])

    #Creating a sobol dict for Process parameters
    S_pp = sobol.analyze(samplesize[0],prediction_model[0].flatten())

    #Creating a sobol dict for Material properties
    S_mat = sobol.analyze(samplesize[1],prediction_model[1].flatten())

    #Creating a sobol dict for Geometry
    S_geo = sobol.analyze(samplesize[2],prediction_model[2].flatten())


    # y = json.dumps(S_pp)
    # y = json.dumps(S_mat)
    # y = json.dumps(S_geo)
    if(dispflag):
        print("\n-------S_pp------------\n",S_pp['S2'])
        print("\n-------S_mat------------\n",S_mat['S2'])
        print("\n-------S_geo------------\n",S_geo['S2'])

    


    plot_obj = PlotterClass()
    # Read on sobol indices here : https://uncertainpy.readthedocs.io/en/latest/theory/sa.html
    plot_obj.sensPlot_2(S_pp,samplesize[0]['names'],'Process Parameters Latent Space for PCSAE')
    plot_obj.sensPlot_2(S_mat,samplesize[1]['names'],'Material Properties Latent Space for PCSAE')
    plot_obj.sensPlot_2(S_geo,samplesize[2]['names'],'Geometry Latent Space for PCSAE')


def parityPlot(test_data,test_predictions,title):
    plt.axes(aspect='equal')
    plt.scatter(test_data,test_predictions)
    plt.title(title + ' Parity Plot')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.rcParams.update({'font.size': 16})
    # lims = [np.floor(min(test_data)), np.round(max(test_data),decimals=0)]
    lims = [0,1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    # plt.savefig(title+'_parityPlot.png')



if __name__ =="__main__":
    main()


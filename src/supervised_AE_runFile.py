import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HelperTools.DataCompletionMethods import DataCompletionMethods
from HelperTools.SupervisedAutoencoder import SupervisedAutoencoder
from HelperTools.plotterClass import PlotterClass

from keras.optimizers import Adam, Adadelta
import os


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
    "nodes_pp_enc_layer"     : 2,
    "nodes_geo_enc_layer"    : 5,
    "nodes_mat_enc_layer"    : 2,
    "nodes_dec_layers"       : 8,
    "nodes_predec_layers"    : 6,
    "inner_layer_actFcn"     : 'tanh',
    "encod_layer_actFcn"     : 'tanh',
    "decod_layer_actFcn"     : 'tanh',
    "output_layer_actFcn"    : 'sigmoid',
    "sup_layer_actFcn"       : 'tanh',
    "optimizer_sup"          : Adam,
    "optimizer_unsup"        : Adadelta,
    "learning_rate"          : 0.008,
    "loss_sup"               : 'mse',
    "loss_unsup"             : 'mse',
    "n_epochs"               : 600,
    "shuffle_flag"           : False,
    "val_split"              : 0.33,
    "verbose"                : 0
    }

    #creating class objects
    sae_obj = SupervisedAutoencoder(hyperparameters)
    plt_obj = PlotterClass()

    train_datafile = dataFile_original.sample(frac=0.8,random_state=42)
    test_datafile = dataFile_original.drop(train_datafile.index)
    sae1_name = 'Supervised Autoencoder'
    history_3lv_sae, sae_3lv, hr_3lv = sae_obj.sup_3lv_ae(train_datafile)
    
    labels = ['DetTorque','RanMRT','final d50']
    x_headers = ['RPM','L/S Ratio','FlowRate (kg/hr)', 'Temperature','Initial d50','Binder Viscosity (mPa.s)','Flowability (HR)','Bulk Density','nCE','Granulator diameter (mm)','L/D Ratio','SA of KE','nKE','Liq add position','nKZ','dKZ']
    plt_obj.history_plotter(history_3lv_sae,'loss','val_loss',sae1_name)

    test_mat, test_pp, test_geo, test_all, test_labels, mms_pp_test, \
        mms_geo_test, mms_mat_test, mms_all_test = sae_obj.dataPreprocessing_3lv(test_datafile)

    # getting prediction for reconstruction and prediction
    predictions_y_AE = sae_3lv.predict([test_pp,test_mat,test_geo])

    pred_con_y = np.reshape(np.ravel(np.array(predictions_y_AE[1])),(len(test_datafile),3))
    pred_con_recons = np.reshape(np.ravel(np.array(predictions_y_AE[0])),(len(test_datafile),16))
    
    
    r2_ypre = sae_obj.calculateR2(np.ravel(predictions_y_AE[1]),test_labels.values,labels)
    r2_recons = sae_obj.calculateR2(np.ravel(predictions_y_AE[0]),test_all,x_headers)

    print(r2_ypre)

    hrall_mat, hrall_pp, hrall_geo, hrall_all, hrall_labels, mms_pp_hrall, \
        mms_geo_hrall, mms_mat_hrall, mms_all_hrall = sae_obj.dataPreprocessing_3lv(dataFile_original)

    latent_rep = np.reshape(np.array(hr_3lv.predict([hrall_pp,hrall_mat,hrall_geo])).T,(len(dataFile_original),3))
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
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_pp,label_mat,i_pp,i_mat)
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_geo,label_mat,i_geo,i_mat)
    plot_all = sae_obj.plot_all3_range1_un2(latent_rep,dataFile_original,dataFile_original['Experiments'],dataFile_original['Regime'],dataFile_original['Extent of Granulation'],\
        ranges,titles,label_pp,label_geo,i_pp,i_geo)
    
    print("\n--------- Performance--------\n")
    print("R^2 for prediction     = ",r2_ypre)
    print("R^2 for reconstruction = ",r2_recons)

    
    plt.show()

if __name__ =="__main__":
    main()


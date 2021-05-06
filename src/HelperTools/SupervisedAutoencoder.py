# """"
# Takes hyperparameters dictionary as input, hyperparameter is defined as:
# hyperparameters = {}
#     "input_layer_nodes"      : 16,
#     "out_layer_nodes_unsup"  : 16,
#     "out_layer_nodes_sup"    : 3,
#     "nodes_predec_layer"     : 4,
#     "lv_layer_nodes"         : 1, 
#     "nodes_enc_layer"        : 6,
#     "nodes_pp_enc_layer"     : 6,
#     "nodes_geo_enc_layer"    : 6,
#     "nodes_mat_enc_layer"    : 6,
#     "nodes_dec_layers"       : 2,
#     "inner_layer_actFcn"     : 'tanh',
#     "encod_layer_actFcn"     : 'tanh',
#     "decod_layer_actFcn"     : 'tanh',
#     "output_layer_actFcn"    : 'tanh',
#     "sup_layer_actFcn"       : 'tanh',
#     "optimizer_sup"          : "adadelta",
#     "optimizer_unsup"        : "adadelta",
#     "learning_rate"          : 0.03,
#     "loss_sup"               : 'mse',
#     "loss_unsup"             : 'mse',
#     "n_epochs"               : 100,
#     "shuffle_flag"           : False,
#     "val_split"              : 0.25,
#     "verbose"                : 0
# }
# """"

from keras import regularizers
from keras.layers import Dense, Input, Concatenate
from keras.models import Sequential, Model

from numpy import unique, where
from .DataCompletionMethods import DataCompletionMethods
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class SupervisedAutoencoder:

    # define init 
    def __init__(self,parameters):
        self.parameters=parameters

    # single unsupervised layer autoencoder

    def unsup_one_ae(self,train_dataset,fit_flag):
        # singlelayer encoder -- Input --> bottleneck --> output
        params = self.parameters
        # defining input layer
        input_layer = Input(shape=(params["input_layer_nodes"],))

        # bottleneck layer
        encoder = Dense(params["lv_layer_nodes"],activation=params["inner_layer_actFcn"],name='lv_layer')(input_layer)

        # output layer reconstruction
        output = Dense(params["out_layer_nodes_unsup"],activation=params["output_layer_actFcn"],name='recon_out')(encoder)

        autoencoder = Model(input_layer,output)
        autoencoder.compile(optimizer=params["optimizer_unsup"],loss=params["loss_unsup"],metrics=['mse','mae'])
        
        # if fitting is reqd 
        if(fit_flag):
            autoencoder.fit(train_dataset,train_dataset,epochs=params["n_epochs"], \
            shuffle=params["shuffle_flag"],validation_split=params["val_split"],verbose=params["verbose"])

        # Getting the hidden representation of the 
        hidden_representation = Sequential()
        hidden_representation.add(autoencoder.layers[0])
        hidden_representation.add(autoencoder.layers[1])

        return autoencoder,hidden_representation


    def unsup_two_ae(self,train_dataset,fit_flag):
        # singlelayer encoder -- Input --> Hidden layer --> bottleneck --> output
        params = self.parameters
        # defining input layer
        input_layer = Input(shape=(params["input_layer_nodes"],))
        #hidden layer
        hidden_1 = Dense(params["nodes_enc_layer"],activation=params["inner_layer_actFcn"],name='hidden_layer_1')(input_layer)
        # bottleneck layer
        encoder = Dense(params["lv_layer_nodes"],activation=params["inner_layer_actFcn"],name='lv_layer')(hidden_1)

        # output layer reconstruction
        output = Dense(params["out_layer_nodes_unsup"],activation=params["output_layer_actFcn"],name='recon_out')(encoder)

        autoencoder = Model(input_layer,output)
        autoencoder.compile(optimizer=params["optimizer_unsup"],loss=params["loss_unsup"],metrics=['mse','mae'])
        
        # if fitting is reqd 
        if(fit_flag):
            autoencoder.fit(train_dataset,train_dataset,epochs=params["n_epochs"], \
            shuffle=params["shuffle_flag"],validation_split=params["val_split"],verbose=params["verbose"])

        # Getting the hidden representation of the 
        hidden_representation = Sequential()
        hidden_representation.add(autoencoder.layers[0])
        hidden_representation.add(autoencoder.layers[1])
        hidden_representation.add(autoencoder.layers[2])

        return autoencoder,hidden_representation

    def unsup_two_two_ae(self,train_dataset,fit_flag):
            # singlelayer encoder -- Input --> Hidden layer --> bottleneck --> output
            params = self.parameters
            # defining input layer
            input_layer = Input(shape=(params["input_layer_nodes"],))
            #hidden layer
            hidden_1 = Dense(params["nodes_enc_layer"],activation=params["inner_layer_actFcn"],name='hidden_layer_1')(input_layer)
            # bottleneck layer
            encoder = Dense(params["lv_layer_nodes"],activation=params["inner_layer_actFcn"],name='lv_layer')(hidden_1)
            #hidden layer
            hidden_2 = Dense(params["nodes_enc_layer"],activation=params["inner_layer_actFcn"],name='hidden_layer_2')(encoder)

            # output layer reconstruction
            output = Dense(params["out_layer_nodes_unsup"],activation=params["output_layer_actFcn"],name='recon_out')(hidden_2)

            autoencoder = Model(input_layer,output)
            autoencoder.compile(optimizer=params["optimizer_unsup"],loss=params["loss_unsup"],metrics=['mse','mae'])
            
            # if fitting is reqd 
            if(fit_flag):
                autoencoder.fit(train_dataset,train_dataset,epochs=params["n_epochs"], \
                shuffle=params["shuffle_flag"],validation_split=params["val_split"],verbose=params["verbose"])

            # Getting the hidden representation of the 
            hidden_representation = Sequential()
            hidden_representation.add(autoencoder.layers[0])
            hidden_representation.add(autoencoder.layers[1])
            hidden_representation.add(autoencoder.layers[2])

            return autoencoder,hidden_representation

    def sup_two_ae(self,train_dataset,train_labels,fit_flag):
            # singlelayer encoder -- Input --> Hidden layer --> bottleneck -->--> Hidden layer --> recons and  output
            params = self.parameters
            # defining input layer
            input_layer = Input(shape=(params["input_layer_nodes"],))
            #hidden layer
            hidden_1 = Dense(params["nodes_enc_layer"],activation=params["inner_layer_actFcn"],name='hidden_layer_1')(input_layer)
            # bottleneck layer
            encoder = Dense(params["lv_layer_nodes"],activation=params["inner_layer_actFcn"],name='lv_layer')(hidden_1)
            #hidden layer
            hidden_2 = Dense(params["nodes_enc_layer"],activation=params["inner_layer_actFcn"],name='hidden_layer_2')(encoder)

            # hidden layer for exp reg
            hidden_exp = Dense(params["nodes_enc_layer"],activation=params["inner_layer_actFcn"],name='hidden_layer_exp')(encoder)

            # output layers reconstruction
            output = Dense(params["out_layer_nodes_unsup"],activation=params["output_layer_actFcn"],name='recon_out')(hidden_2)
            output_exp = Dense(params["out_layer_nodes_sup"],activation=params["output_layer_actFcn"],name='exp_out')(hidden_2)

            autoencoder = Model(input=input_layer,output=[output,output_exp])
            autoencoder.compile(optimizer=params["optimizer_unsup"],loss=params["loss_unsup"],metrics=['mse','mae'])
            
            # if fitting is reqd 
            if(fit_flag):
                autoencoder.fit(train_dataset,[train_dataset,train_labels],epochs=params["n_epochs"], \
                shuffle=params["shuffle_flag"],validation_split=params["val_split"],verbose=params["verbose"])

            # Getting the hidden representation of the 
            hidden_representation = Sequential()
            hidden_representation.add(autoencoder.layers[0])
            hidden_representation.add(autoencoder.layers[1])
            hidden_representation.add(autoencoder.layers[2])

            return autoencoder,hidden_representation


            
    def sup_one_ae(self,train_dataset,train_labels,fit_flag):
            # singlelayer encoder -- Input --> Hidden layer --> bottleneck --> output
            params = self.parameters
            # defining input layer
            input_layer = Input(shape=(params["input_layer_nodes"],))
            
            # bottleneck layer
            encoder = Dense(params["lv_layer_nodes"],activation=params["inner_layer_actFcn"],name='lv_layer')(input_layer)
            #hidden layer
            
            # output layers reconstruction
            output = Dense(params["out_layer_nodes_unsup"],activation=params["output_layer_actFcn"],name='recon_out')(encoder)
            output_exp = Dense(params["out_layer_nodes_sup"],activation=params["output_layer_actFcn"],name='exp_out')(encoder)

            autoencoder = Model(input=input_layer,output=[output,output_exp])
            autoencoder.compile(optimizer=params["optimizer_unsup"],loss=params["loss_unsup"],metrics=['mse','mae'])
            
            # if fitting is reqd 
            if(fit_flag):
                autoencoder.fit(train_dataset,[train_dataset,train_labels],epochs=params["n_epochs"], \
                shuffle=params["shuffle_flag"],validation_split=params["val_split"],verbose=params["verbose"])

            # Getting the hidden representation of the 
            hidden_representation = Sequential()
            hidden_representation.add(autoencoder.layers[0])
            hidden_representation.add(autoencoder.layers[1])

            return autoencoder,hidden_representation

    def sup_3lv_ae(self,train_datafile):
        # collect preprocessed data
        train_mat, train_pp, train_geo, train_all, train_labels, mms_pp, mms_geo, mms_mat, mms_all = self.dataPreprocessing_3lv(train_datafile)
        # test_mat, test_pp, test_geo, test_all, test_labels, mms_pp_test, mms_geo_test, mms_mat_test, mms_all_test = self.dataPreprocessing_3lv(test_datafile)
        
        params = self.parameters
        
        # start the autoencoder 
        ##LV for Process Parameters
        
        input_pp = Input(shape=(train_pp.shape[1],))
        #hidden layer
        hidden_pp1 = Dense(params["nodes_pp_enc_layer"],activation=params["inner_layer_actFcn"],name='hid_layer_pp1')(input_pp)
        # bottleneck layer
        encoder_pp = Dense(params["nodes_lv_layer"],activation=params["encod_layer_actFcn"],name='lv_layer_pp')(hidden_pp1)


        ##LV for Material Properties
        input_mat = Input(shape=(train_mat.shape[1],))
        #hidden layer
        hidden_mat = Dense(params["nodes_mat_enc_layer"],activation=params["inner_layer_actFcn"],name='hid_layer_mat')(input_mat)
        # bottleneck layer
        encoder_mat = Dense(params["nodes_lv_layer"],activation=params["encod_layer_actFcn"],name='lv_layer_mat')(hidden_mat)

        ##LV for Geometry
        input_geo = Input(shape=(train_geo.shape[1],))
        #hidden layer
        hidden_geo = Dense(params["nodes_geo_enc_layer"],activation=params["decod_layer_actFcn"],name='hid_layer_geo')(input_geo)
        # bottleneck layer
        encoder_geo = Dense(params["nodes_lv_layer"],activation=params["encod_layer_actFcn"],name='lv_layer_geo')(hidden_geo)

        concat_bottleneck = Concatenate()([encoder_pp,encoder_mat,encoder_geo])
        
        # recons decoder layer 1
        decoder_recons = Dense(params["nodes_dec_layers"],activation=params["decod_layer_actFcn"],name='decoder_layer_recons1')(concat_bottleneck)

        # output prediction decoder layer 1 
        decoder_pred = Dense(params["nodes_predec_layer_1"],activation=params["decod_layer_actFcn"],name='decoder_layer_pred1')(concat_bottleneck)

        # output layer reconstruction
        output_recons = Dense(train_all.shape[1],activation=params["output_layer_actFcn"],name='recon_out')(decoder_recons)
        output_pred = Dense(train_labels.shape[1],activation=params["sup_layer_actFcn"],name='pred_out')(decoder_pred)

        autoencoder = Model([input_pp,input_mat,input_geo],[output_recons,output_pred])
        autoencoder.compile(optimizer=params['optimizer_sup'](learning_rate=params['learning_rate']),loss=params['loss_sup'],metrics=['mse','mae'])

        history = autoencoder.fit([train_pp,train_mat,train_geo],[train_all,train_labels],epochs=params['n_epochs'], shuffle=True,validation_split=params['val_split'],verbose=params['verbose'])

        # autoencoder.summary()

        # predictions_y_AE = autoencoder.predict([test_pp,test_mat,test_geo])

        # Getting the hidden representation of the 
        hidden_representation = Model([input_pp,input_mat,input_geo],[encoder_pp,encoder_mat,encoder_geo])
    
        # hidden_representation_3.summary()
        # latent_rep = np.array(hidden_representation.predict([datafile_processparam,datafile_material,datafile_geometry]))
        # hidden_representation_3.summary()

        return history, autoencoder, hidden_representation


    def sup_3lv_ae_2hiddenpre(self,train_datafile):
        # collect preprocessed data
        train_mat, train_pp, train_geo, train_all, train_labels, mms_pp, mms_geo, mms_mat, mms_all = self.dataPreprocessing_3lv(train_datafile)
        # test_mat, test_pp, test_geo, test_all, test_labels, mms_pp_test, mms_geo_test, mms_mat_test, mms_all_test = self.dataPreprocessing_3lv(test_datafile)
        
        params = self.parameters
        
        # start the autoencoder 
        ##LV for Process Parameters
        
        input_pp = Input(shape=(train_pp.shape[1],))
        #hidden layer
        hidden_pp1 = Dense(params["nodes_pp_enc_layer"],activation=params["inner_layer_actFcn"],name='hid_layer_pp1')(input_pp)
        # bottleneck layer
        encoder_pp = Dense(params["nodes_lv_layer"],activation=params["encod_layer_actFcn"],name='lv_layer_pp')(hidden_pp1)


        ##LV for Material Properties
        input_mat = Input(shape=(train_mat.shape[1],))
        #hidden layer
        hidden_mat = Dense(params["nodes_mat_enc_layer"],activation=params["inner_layer_actFcn"],name='hid_layer_mat')(input_mat)
        # bottleneck layer
        encoder_mat = Dense(params["nodes_lv_layer"],activation=params["encod_layer_actFcn"],name='lv_layer_mat')(hidden_mat)

        ##LV for Geometry
        input_geo = Input(shape=(train_geo.shape[1],))
        #hidden layer
        hidden_geo = Dense(params["nodes_geo_enc_layer"],activation=params["decod_layer_actFcn"],name='hid_layer_geo')(input_geo)
        # bottleneck layer
        encoder_geo = Dense(params["nodes_lv_layer"],activation=params["encod_layer_actFcn"],name='lv_layer_geo')(hidden_geo)

        concat_bottleneck = Concatenate()([encoder_pp,encoder_mat,encoder_geo])
        
        # recons decoder layer 1
        decoder_recons_1 = Dense(params["nodes_dec_layers"],activation=params["decod_layer_actFcn"],name='decoder_layer_recons1')(concat_bottleneck)
        decoder_recons_2 = Dense(params["nodes_dec_layers"],activation=params["decod_layer_actFcn"],name='decoder_layer_recons2')(decoder_recons_1)

        # output prediction decoder layer 1 
        decoder_pred_1 = Dense(params["nodes_predec_layer_1"],activation=params["decod_layer_actFcn"],name='decoder_layer_pred1')(concat_bottleneck)
        # output prediction decoder layer 2
        decoder_pred_2 = Dense(params["nodes_predec_layer_2"],activation=params["decod_layer_actFcn"],name='decoder_layer_pred2')(decoder_pred_1)
        # output prediction decoder layer 3
        decoder_pred_2 = Dense(params["nodes_predec_layer_2"],activation=params["decod_layer_actFcn"],name='decoder_layer_pred3')(decoder_pred_2)

        # output layer reconstruction
        output_recons = Dense(train_all.shape[1],activation=params["output_layer_actFcn"],name='recon_out')(decoder_recons_2)
        output_pred = Dense(train_labels.shape[1],activation=params["sup_layer_actFcn"],name='pred_out')(decoder_pred_3)

        autoencoder = Model([input_pp,input_mat,input_geo],[output_recons,output_pred])
        autoencoder.compile(optimizer=params['optimizer_sup'](learning_rate=params['learning_rate']),loss=params['loss_sup'],metrics=['mse','mae'])

        history = autoencoder.fit([train_pp,train_mat,train_geo],[train_all,train_labels],epochs=params['n_epochs'], shuffle=True,validation_split=params['val_split'],verbose=params['verbose'])

        # autoencoder.summary()

        # predictions_y_AE = autoencoder.predict([test_pp,test_mat,test_geo])

        # Getting the hidden representation of the 
        hidden_representation = Model([input_pp,input_mat,input_geo],[encoder_pp,encoder_mat,encoder_geo])
    
        # hidden_representation_3.summary()
        # latent_rep = np.array(hidden_representation.predict([datafile_processparam,datafile_material,datafile_geometry]))
        # hidden_representation_3.summary()

        return history, autoencoder, hidden_representation

    def dataPreprocessing_3lv(self,train_datafile):
        #data preprocessing for training data
        datafile_processparam = train_datafile[['RPM','L/S Ratio','FlowRate (kg/hr)', 'Temperature']]
        datafile_material     = train_datafile[['Initial d50','Binder Viscosity (mPa.s)','Flowability (HR)','Bulk Density']]
        datafile_geometry     = train_datafile[['nCE','Granulator diameter (mm)','L/D Ratio','SA of KE','nKE','Liq add position','nKZ','dKZ']]
        datafile_allinputs    = train_datafile[['RPM','L/S Ratio','FlowRate (kg/hr)', 'Temperature','Initial d50','Binder Viscosity (mPa.s)','Flowability (HR)','Bulk Density','nCE','Granulator diameter (mm)','L/D Ratio','SA of KE','nKE','Liq add position','nKZ','dKZ']]
        
        
        train_labels          = train_datafile[['DetTorque','RanMRT','final d50']]
        train_labels['final d50'] = train_labels['final d50'] / 1e6
        train_labels['DetTorque'] = train_labels['DetTorque'] / 10
        train_labels['RanMRT']    = train_labels['RanMRT'] / 100

        # label encoding for parameters with strings/characters to convert them to float
        le = LabelEncoder()
        datafile_geometry['Liq add position']  = le.fit_transform(datafile_geometry['Liq add position'])
        datafile_allinputs['Liq add position'] = le.fit_transform(datafile_allinputs['Liq add position'])

        # MinMaxScaler for scaling all data files
        mms_pp     = MinMaxScaler()
        mms_geo    = MinMaxScaler()
        mms_mat    = MinMaxScaler()
        mms_all    = MinMaxScaler()

        datafile_material     = mms_mat.fit_transform(datafile_material)
        datafile_processparam = mms_pp.fit_transform(datafile_processparam)
        datafile_geometry     = mms_geo.fit_transform(datafile_geometry)
        datafile_allinputs    = mms_all.fit_transform(datafile_allinputs)

        return datafile_material, datafile_processparam, datafile_geometry, datafile_allinputs, train_labels, mms_pp, mms_geo, mms_mat, mms_all

    def calculateR2(self, y_predicted, y_true, labels):
        # labels = ['Granule_Density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        y_predicted_mean = np.array(y_predicted).mean()
        y_true_reshaped = np.reshape(np.ravel(y_true),(len(y_true)*len(labels),))
        '''
        test['output_pred'] = model.predict(x=np.array(test[inputs]))
        output_mean = test['output'].mean()    # Is this the correct mean value for r2 here?
        test['SSres'] = np.square(test['output']-test['output_pred'])
        test['SStot'] = np.square(test['output']-output_mean)
        r2 = 1-(test['SSres'].sum()/(test['SStot'].sum()))
        '''    
        SSres = np.square(y_true_reshaped - y_predicted)
        SStol = np.square(y_true_reshaped - y_predicted_mean)
        
        r2 = 1 - (SSres.sum() / SStol.sum())          
        return r2


    def plot_LV_unique(self,latent_rep,unique_par,title):
        plt.figure()
        for g in np.unique(unique_par):
            i = np.where(unique_par==g)
            plt.scatter(latent_rep[i,0],latent_rep[i,1],label=g)
        plt.xlabel('LV 1')
        plt.ylabel('LV 2')
        plt.title(title)
        plt.legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0.)


    def plot_LV_range(self,latent_rep,train_dataset,range1,para_cons,title):
        train_dataset_copy = train_dataset.copy()
        train_dataset_copy['Extent of gran'] = para_cons
        train_dataset_copy['LV 1'] = latent_rep[:,0]
        train_dataset_copy['LV 2'] = latent_rep[:,1]
        groups = train_dataset_copy.groupby(pd.cut(train_dataset_copy['Extent of gran'],range1))
        for val, group in groups:
            plt.scatter(group['LV 1'],group['LV 2'],label=val)
        plt.xlabel('LV 1')
        plt.ylabel('LV 2')
        plt.title(title)
        plt.legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=3, mode="expand", borderaxespad=0)

    def plot_all3_range2_un1(self,latent_rep,train_dataset,para1,para2,para3,range1,range2,titles,xlabels,ylabels,i1,i2):
        fig,ax = plt.subplots(1,3,sharey=True)
        for g in np.unique(para1):
            i = np.where(para1==g)
            ax[0].scatter(latent_rep[i,i1],latent_rep[i,i2],label=g)
        ax[0].set_xlabel(xlabels)
        ax[0].set_ylabel(ylabels)
        ax[0].set_title(titles[0])
        ax[0].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0.)
        
        train_dataset_copy = train_dataset.copy()
        train_dataset_copy['Extent of gran'] = para2
        train_dataset_copy['LV 1'] = latent_rep[:,i1]
        train_dataset_copy['LV 2'] = latent_rep[:,i2]
        groups = train_dataset_copy.groupby(pd.cut(train_dataset_copy['Extent of gran'],range1))
        for val, group in groups:
            ax[1].scatter(group['LV 1'],group['LV 2'],label=val)
        ax[1].set_xlabel(xlabels)
        ax[1].set_ylabel(ylabels)
        ax[1].set_title(titles[1])
        fig.set_size_inches(12,8)
        fig.suptitle('3 latent space representation',fontsize=14)
        ax[1].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0)
        
        train_dataset_copy['Calc Fill level'] = para3
        groups = train_dataset_copy.groupby(pd.cut(train_dataset_copy['Calc Fill level'],range2))
        for val, group in groups:
            ax[2].scatter(group['LV 1'],group['LV 2'],label=val)
        ax[1].set_xlabel(xlabels)
        ax[1].set_ylabel(ylabels)
        ax[2].set_title(titles[2])
        ax[2].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0.)
        plt.tight_layout()


    def plot_all3_range1_un2(self,latent_rep,train_dataset,para1,para2,para3,range1,titles,xlabels,ylabels,i1,i2):
        fig,ax = plt.subplots(1,3,sharey=True)
        fig.set_size_inches(12,8)
        fig.suptitle('3 latent space representation',fontsize=14)

        for g in np.unique(para1):
            i = np.where(para1==g)
            ax[0].scatter(latent_rep[i,i1],latent_rep[i,i2],label=g)
        ax[0].set_xlabel(xlabels)
        ax[0].set_ylabel(ylabels)
        ax[0].set_title(titles[0])
        ax[0].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0.)
        
        train_dataset_copy = train_dataset.copy()
        for g in np.unique(para2):
            i = np.where(para2==g)
            ax[1].scatter(latent_rep[i,i1],latent_rep[i,i2],label=g)
        ax[1].set_xlabel(xlabels)
        ax[1].set_ylabel(ylabels)
        ax[1].set_title(titles[1])
        ax[1].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0)
        
        train_dataset_copy['Extent of gran'] = para3
        train_dataset_copy['LV 1'] = latent_rep[:,i1]
        train_dataset_copy['LV 2'] = latent_rep[:,i2]
        groups = train_dataset_copy.groupby(pd.cut(train_dataset_copy['Extent of gran'],range1))        
        for val, group in groups:
            ax[2].scatter(group['LV 1'],group['LV 2'],label=val)
        ax[2].set_xlabel(xlabels)
        ax[2].set_ylabel(ylabels)
        ax[2].set_title(titles[2])
        ax[2].legend(bbox_to_anchor=(0., -0.5, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0.)
        plt.tight_layout()
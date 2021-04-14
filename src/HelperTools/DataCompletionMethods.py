import numpy as np
from numpy.lib.function_base import _place_dispatcher
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model


"""
This class is used to fill data in the collected 
TSG data files 
The methods compelted fill level using Osorio's correlation 
Regression methods for torque and MRT are also present 

"""

class DataCompletionMethods:
    # defining init method
    def __init__(self,uncompletedDatafile):
        self.ucDatafile = uncompletedDatafile # stores the uncompleted combined datafile of exp data
    
####### Fill level prediciton calculations
    def vFreeCalculation(self,dGran,screwConfig):
        l_total = np.multiply(screwConfig["Granulator diameter (mm)"],screwConfig["L/D Ratio"]) 
        l_eff = l_total-2*np.multiply(dGran,self.scalingwithMun(dGran,"Lspce",1))
        At = 2 * self.scalingwithMun(dGran,"Ac",2) - self.scalingwithMun(dGran,"Ae",2)
        v_max = np.multiply(l_eff,At)
        v_shaft = np.multiply(l_eff,self.scalingwithMun(dGran,"As",2))
        v_elem = screwConfig["n KE"] * self.scalingwithMun(dGran,"Vke",3) + screwConfig["nCE"] * self.scalingwithMun(dGran,"Vspce",3)
        v_free = v_max - v_shaft - v_elem

        return v_free, v_max

    def pfnCalc(self):
        dataFile = self.ucDatafile
        denom = np.multiply(dataFile['Bulk Density'],np.multiply(dataFile['RPM']*np.pi/30,np.power(dataFile["Granulator diameter (mm)"]/1000,3)))
        # pfn = np.divide((dataFile["FlowRate (kg/hr)"]*(1+dataFile["L/S Ratio"]))/3600,denom) 
        pfn = np.divide(dataFile["FlowRate (kg/hr)"]/3600,denom) 
        return pfn

    def fillLevel_osorio(self,mrt):
        dataFile = self.ucDatafile
        D = dataFile["Granulator diameter (mm)"]
        F1 = np.divide(self.scalingwithMun(D,"Ae",2)/1e6,np.power(D/1000,2))
        F2 = np.ones((len(F1)))
        # mrt = dataFile["DetMRT"]
        vp = np.divide(np.multiply(D,dataFile["L/D Ratio"])/1000,mrt)
        F3num = 2*np.pi*vp
        F3dem = np.multiply(dataFile["RPM"]*np.pi/30,D/1000)
        F3 = np.divide(F3num,F3dem)
        fill = np.divide(self.pfnCalc(),np.multiply(F1,np.multiply(F2,F3)))
        return fill

    def scalingwithMun(self,dGran,prop,scaleVal):
        munDim = {
        "OD" : 16,
        "L/D": 25,
        "Ae" : 23.56,
        "Ls" : 3,
        "As" : 34.37,
        "Ll" : 10,
        "Tdpm" : 1.18,
        "Bdpm" : 0.65,
        "Tdlm" : 1.52,
        "Bdlm" : 0.57,
        "Lspce" : 1,
        "Ac" : 201,
        "Vspce" : 1,
        "Vlpce" : 2,
        "Vke" : 0.3,
        "Vice" : 0.75,
        }
        finVal = np.empty(len(dGran))

        for i in range(len(finVal)):
            if dGran[i] == munDim["OD"]:
                finVal[i] = (munDim[prop])
            else:
                val1 = np.power(munDim[prop],1/scaleVal)
                val2 = val1 * (dGran[i] / munDim["OD"])
                finVal[i] = (np.power(val2,scaleVal))
        
        return finVal

    
    def fillevel_lalith(self,mrt):
        screwConfig = self.ucDatafile[["Granulator diameter (mm)","L/D Ratio","n KE","nCE"]]
        dGran = screwConfig["Granulator diameter (mm)"]
        freeVolume,maxVol = self.vFreeCalculation(dGran,screwConfig)
        num = np.multiply(freeVolume/1e9,np.multiply(self.ucDatafile["FlowRate (kg/hr)"]/3600,mrt))#self.ucDatafile["DetMRT"]))
        dem = np.multiply(self.ucDatafile["Bulk Density"],np.power(maxVol/1e9,2))

        filllevel = np.divide(num,dem)
        return filllevel
########## Data imputation using regression techniques

    def lineaeRegressionModel(self,missing_columns,plotflag):
        dataFile_emptyOutputs = self.ucDatafile
        for feature in missing_columns:
            dataFile_emptyOutputs[feature+'_imp'] = dataFile_emptyOutputs[feature]
            dataFile_emptyOutputs = self.random_imputation(dataFile_emptyOutputs,feature)
            # mno.matrix(dataFile_emptyOutputs)
            # print(dataFile_emptyOutputs)
            deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])
            # dataFile_emptyOutputs = dataFile_emptyOutputs.drop(['Sr No','Screw Configuration','Experiments','Liq add position','Regime','Beta','d50','Exp Fill level'],axis=1)

        for feature in missing_columns:
                
            deter_data["Det" + feature] = dataFile_emptyOutputs[feature + "_imp"]
            parameters = list(set(dataFile_emptyOutputs.columns) - set(missing_columns) - {feature + '_imp'})
            # print(parameters)
            #Create a Linear Regression model to estimate the missing data
            model = linear_model.LinearRegression(fit_intercept=False,normalize=False,positive=False)
            model.fit(X = dataFile_emptyOutputs[parameters], y = dataFile_emptyOutputs[feature + '_imp'])
            
            #observe that I preserve the index of the missing data from the original dataframe
            deter_data.loc[dataFile_emptyOutputs[feature].isnull(), "Det" + feature] = model.predict(dataFile_emptyOutputs[parameters])[dataFile_emptyOutputs[feature].isnull()]
        
        if(plotflag):
            sns.set()
            fig, axes = plt.subplots(nrows = 2, ncols = 2)

            for index, variable in enumerate(["Torque","MRT"]):
                sns.distplot(dataFile_emptyOutputs[variable].dropna(), kde = False, ax = axes[index, 0])
                sns.distplot(deter_data["Det" + variable], kde = False, ax = axes[index, 0], color = 'red')
                
                sns.boxplot(data = pd.concat([dataFile_emptyOutputs[variable], deter_data["Det" + variable]], axis = 1),
                            ax = axes[index, 1])
                
            plt.tight_layout()

            results1 = deter_data[["DetTorque","DetMRT"]]
            fig.set_size_inches(8, 8)
        return deter_data

##### Stochastic Data completion
    def randomImputationRegression(self,missing_columns,plot_flag):
        dataFile_emptyOutputs = self.ucDatafile
        for feature in missing_columns:
            dataFile_emptyOutputs[feature+'_imp'] = dataFile_emptyOutputs[feature]
            dataFile_emptyOutputs = self.random_imputation(dataFile_emptyOutputs,feature)
            # mno.matrix(dataFile_emptyOutputs)
            # print(dataFile_emptyOutputs)
            deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])
            dataFile_emptyOutputs = dataFile_emptyOutputs.drop(['Sr No','Screw Configuration','Experiments','Liq add position','Regime','Beta','d50','Exp Fill level'],axis=1)
            random_data = pd.DataFrame(columns = ["Ran" + name for name in missing_columns])

        for feature in missing_columns:
                
            random_data["Ran" + feature] = dataFile_emptyOutputs[feature + '_imp']
            parameters = list(set(dataFile_emptyOutputs.columns) - set(missing_columns) - {feature + '_imp'})
            
            model1 = linear_model.LinearRegression(fit_intercept=False,normalize=False,positive=False)
            model1.fit(X = dataFile_emptyOutputs[parameters], y = dataFile_emptyOutputs[feature + '_imp'])
            
            #Standard Error of the regression estimates is equal to std() of the errors of each estimates
            predict = model1.predict(dataFile_emptyOutputs[parameters])
            std_error = (predict[dataFile_emptyOutputs[feature].notnull()] - dataFile_emptyOutputs.loc[dataFile_emptyOutputs[feature].notnull(), feature + '_imp']).std()
            
            #observe that I preserve the index of the missing data from the original dataframe
            random_predict = np.random.normal(size = dataFile_emptyOutputs[feature].shape[0], 
                                            loc = predict, 
                                            scale = std_error)
            random_data.loc[(dataFile_emptyOutputs[feature].isnull()) & (random_predict > 0), "Ran" + feature] = random_predict[(dataFile_emptyOutputs[feature].isnull()) & 
                                                                                    (random_predict > 0)]
        if(plot_flag):    
            sns.set()
            fig, axes = plt.subplots(nrows = 2, ncols = 2)
            fig.set_size_inches(8, 8)

            for index, variable in enumerate(["Torque","MRT"]):
                sns.distplot(dataFile_emptyOutputs[variable].dropna(), kde = False, ax = axes[index, 0])
                sns.distplot(random_data["Ran" + variable], kde = False, ax = axes[index, 0], color = 'red')
                axes[index, 0].set(xlabel = variable + " / " + variable + '_imp')
                
                sns.boxplot(data = pd.concat([dataFile_emptyOutputs[variable], random_data["Ran" + variable]], axis = 1),ax = axes[index, 1])
                
                plt.tight_layout()
            results2 = random_data[["RanTorque","RanMRT"]]
            dataFile_completedTorque = pd.concat([dataFile_emptyOutputs,results2])
            print(dataFile_completedTorque[["Torque","Torque_imp","RanTorque",\
                "MRT","MRT_imp","RanMRT"]].describe().T)
            dataFile_completedTorque.to_csv('completed_random.csv')
            plt.show()
        return random_data
    
    def random_imputation(self,df, feature):
        number_missing = df[feature].isnull().sum()
        observed_values = df.loc[df[feature].notnull(), feature]
        df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
        
        return df
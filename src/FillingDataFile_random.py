import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import missingno as mno

def main():
    dataFile_emptyOutputs = pd.read_csv('test_data1_WithOutputs.csv')
    # print(dataFile_emptyOutputs.describe())
    missing_columns = ['Torque','MRT']
    for feature in missing_columns:
        dataFile_emptyOutputs[feature+'_imp'] = dataFile_emptyOutputs[feature]
        dataFile_emptyOutputs = random_imputation(dataFile_emptyOutputs,feature)
    # mno.matrix(dataFile_emptyOutputs)
    # print(dataFile_emptyOutputs)
    deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])
    dataFile_emptyOutputs = dataFile_emptyOutputs.drop(['Sr No','Screw Configuration','Experiments','Liq add position','Regime','Beta','d50','Exp Fill level'],axis=1)

    # for feature in missing_columns:
            
    #     deter_data["Det" + feature] = dataFile_emptyOutputs[feature + "_imp"]
    #     parameters = list(set(dataFile_emptyOutputs.columns) - set(missing_columns) - {feature + '_imp'})
    #     # print(parameters)
    #     #Create a Linear Regression model to estimate the missing data
    #     model = linear_model.LinearRegression()
    #     model.fit(X = dataFile_emptyOutputs[parameters], y = dataFile_emptyOutputs[feature + '_imp'])
        
    #     #observe that I preserve the index of the missing data from the original dataframe
    #     deter_data.loc[dataFile_emptyOutputs[feature].isnull(), "Det" + feature] = model.predict(dataFile_emptyOutputs[parameters])[dataFile_emptyOutputs[feature].isnull()]
    

    # sns.set()
    # fig, axes = plt.subplots(nrows = 2, ncols = 2)

    # for index, variable in enumerate(["Torque","MRT"]):
    #     sns.distplot(dataFile_emptyOutputs[variable].dropna(), kde = False, ax = axes[index, 0])
    #     sns.distplot(deter_data["Det" + variable], kde = False, ax = axes[index, 0], color = 'red')
        
    #     sns.boxplot(data = pd.concat([dataFile_emptyOutputs[variable], deter_data["Det" + variable]], axis = 1),
    #                 ax = axes[index, 1])
        
    # plt.tight_layout()

    # results1 = deter_data[["DetTorque","DetMRT"]]
    # fig.set_size_inches(8, 8)

    # Stochastic Regression Imputation
    random_data = pd.DataFrame(columns = ["Ran" + name for name in missing_columns])

    for feature in missing_columns:
            
        random_data["Ran" + feature] = dataFile_emptyOutputs[feature + '_imp']
        parameters = list(set(dataFile_emptyOutputs.columns) - set(missing_columns) - {feature + '_imp'})
        
        model1 = linear_model.LinearRegression()
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
    sns.set()
    fig, axes = plt.subplots(nrows = 2, ncols = 2)
    fig.set_size_inches(8, 8)

    for index, variable in enumerate(["Torque","MRT"]):
        sns.distplot(dataFile_emptyOutputs[variable].dropna(), kde = False, ax = axes[index, 0])
        sns.distplot(random_data["Ran" + variable], kde = False, ax = axes[index, 0], color = 'red')
        axes[index, 0].set(xlabel = variable + " / " + variable + '_imp')
        
        sns.boxplot(data = pd.concat([dataFile_emptyOutputs[variable], random_data["Ran" + variable]], axis = 1),
                    ax = axes[index, 1])
        
        plt.tight_layout()
    results2 = random_data[["RanTorque","RanMRT"]]
    dataFile_completedTorque = pd.concat([dataFile_emptyOutputs,results2])
    print(dataFile_completedTorque[["Torque","Torque_imp","RanTorque",\
        "MRT","MRT_imp","RanMRT"]].describe().T)
    dataFile_completedTorque.to_csv('completed_random.csv')
    plt.show()


def random_imputation(df, feature):
    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df


if __name__ =="__main__":
    main()

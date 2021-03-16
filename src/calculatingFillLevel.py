import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt

def main():
    cwd = os.getcwd()
    files = 'dataFiles/test_data1_WithOutputs_withCalculatedValues.csv'
    dataFile = pd.read_csv(os.path.join(cwd,files))
    # dataFile = pd.read_csv('dataFiles/test_data1_WithOutputs.csv')
    dataFile_edit = dataFile.drop(['Sr No','Screw Configuration','Experiments','Liq add position'],axis=1)
    screwConfig = dataFile_edit[["Granulator diameter (mm)","L/D Ratio","n KE","nCE"]]
    dGran = screwConfig["Granulator diameter (mm)"]
    vFree_all,v_max_all = vFreeCalculation(dGran,screwConfig)
    pfnVals = pfnCalc(dataFile_edit)
    calc_fillVals = fillCons(dataFile_edit)
    calc_fillVals = fillevel_lalith(dataFile_edit)
    exp_fillVals = np.array(dataFile_edit["Exp Fill level"].dropna() / 100)
    fill_diff = np.abs(np.array(calc_fillVals[73:115])-exp_fillVals)
    # percent_fillDiff = np.divide(calc_fillVals-exp_fillVals,calc_fillVals)*100
    # avg_diff = np.mean(percent_fillDiff)
    print(np.mean(fill_diff))
    plt.scatter(exp_fillVals[0:13],calc_fillVals[73:86],marker='o')
    plt.scatter(exp_fillVals[14:-1],calc_fillVals[87:114],marker='^')
    plt.legend(['Meier 2017','Mundozah 2020'])
    plt.plot([0,max(calc_fillVals)],[0,max(calc_fillVals)],'k')
    plt.plot([0,max(calc_fillVals)],[0.05,max(calc_fillVals)+0.05],'k--')
    plt.plot([0.05,max(calc_fillVals)],[0,max(calc_fillVals)-0.05],'k--')
    plt.xlim(0,max(exp_fillVals))
    plt.ylim(0,max(exp_fillVals))
    plt.xlabel('Experimental fill level')
    plt.ylabel('Calculated fill level')
    plt.show()
    freeVolume,maxVol = vFreeCalculation(dGran,screwConfig)
    fillVolume = np.multiply(freeVolume, calc_fillVals)
    # print(fillVolume/1e6)
    # print(calc_fillVals)
    granSt = 5e6
    # print(calc_fillVals)
    # dataFile_edit["Lalith fill level"] = fillevel_lalith(dataFile_edit)
    dataFile_edit["Calc Fill level"] = calc_fillVals
    dataFile_edit["Calc Fill volume"] = fillVolume
    dataFile_edit["PFNVals"] = pfnVals
    dataFile_edit["Torque / Fill Volume"] = np.divide(dataFile_edit["DetTorque"],fillVolume/1e9)
    dataFile_edit['Vol free'] = vFree_all
    dataFile_edit['Vol max'] = v_max_all
    beta = np.divide(dataFile_edit["DetTorque"],fillVolume/1e9) / granSt
    lsvis = np.multiply(dataFile_edit["Binder Viscosity (mPa.s)"],dataFile_edit["L/S Ratio"])
    dataFile_edit["Calc Beta"] = beta
    print(dataFile_edit.describe().T)
    dataFile_edit.to_csv('calc_fill.csv')

def vFreeCalculation(dGran,screwConfig):
    l_total = np.multiply(screwConfig["Granulator diameter (mm)"],screwConfig["L/D Ratio"]) 
    l_eff = l_total-2*np.multiply(dGran,scalingwithMun(dGran,"Lspce",1))
    At = 2 * scalingwithMun(dGran,"Ac",2) - scalingwithMun(dGran,"Ae",2)
    v_max = np.multiply(l_eff,At)
    # v_max = np.zeros((len(screwConfig["n KE"])))
    # v_max.fill(v_max_sinVal[0])
    v_shaft = np.multiply(l_eff,scalingwithMun(dGran,"As",2))
    # v_shaft = np.zeros((len(screwConfig["n KE"])))
    # v_shaft.fill(v_shaft_sinVal[0])
    v_elem = screwConfig["n KE"] * scalingwithMun(dGran,"Vke",3) + screwConfig["nCE"] * scalingwithMun(dGran,"Vspce",3)
    v_free = v_max - v_shaft - v_elem

    return v_free, v_max

def pfnCalc(dataFile):
    denom = np.multiply(dataFile['Bulk Density'],np.multiply(dataFile['RPM (1/s)']*np.pi/30,np.power(dataFile["Granulator diameter (mm)"]/1000,3)))
    # pfn = np.divide((dataFile["FlowRate (kg/hr)"]*(1+dataFile["L/S Ratio"]))/3600,denom) 
    pfn = np.divide(dataFile["FlowRate (kg/hr)"]/3600,denom) 
    return pfn

def fillCons(dataFile):
    D = dataFile["Granulator diameter (mm)"]
    F1 = np.divide(scalingwithMun(D,"Ae",2)/1e6,np.power(D/1000,2))
    F2 = np.ones((len(F1)))
    mrt = dataFile["DetMRT"]
    vp = np.divide(np.multiply(D,dataFile["L/D Ratio"])/1000,mrt)
    F3num = 2*np.pi*vp
    F3dem = np.multiply(dataFile["RPM (1/s)"]*np.pi/30,D/1000)
    F3 = np.divide(F3num,F3dem)
    fill = np.divide(pfnCalc(dataFile),np.multiply(F1,np.multiply(F2,F3)))
    return fill

def fillevel_lalith(dataFile):
    screwConfig = dataFile[["Granulator diameter (mm)","L/D Ratio","n KE","nCE"]]
    dGran = screwConfig["Granulator diameter (mm)"]
    freeVolume,maxVol = vFreeCalculation(dGran,screwConfig)
    num = np.multiply(freeVolume/1e9,np.multiply(dataFile["FlowRate (kg/hr)"]/3600,dataFile["DetMRT"]))
    dem = np.multiply(dataFile["Bulk Density"],np.power(maxVol/1e9,2))

    filllevel = np.divide(num,dem)
    return filllevel
    

def scalingwithMun(dGran,prop,scaleVal):
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


if __name__ =="__main__":
    main()


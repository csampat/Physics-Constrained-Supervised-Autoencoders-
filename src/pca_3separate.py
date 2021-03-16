import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize,minmax_scale,scale,OneHotEncoder,maxabs_scale

def main():
    # importing file and sorting data
    dataFile = pd.read_csv('pca_test_dhenge2kumar1.csv')
    y = dataFile['Experiments']
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(dataFile[['Liq add position']]).toarray())
    dataFile_edit = dataFile.drop(['Sr No','Screw Configuration','Experiments','Liq add position'],axis=1)
    # datafile_process = dataFile_edit[['RPM (1/s)','L/S Ratio','FlowRate (kg/hr)']]
    # datafile_geometry = dataFile_edit[['nCE','Granulator diameter (mm)','L/D Ratio','SA of KE', 'n KE']]
    # datafile_geometry = datafile_geometry.join(enc_df)
    # datafile_material = dataFile_edit[['Intial D50','Binder Viscosity (mPa.s)','Flowability (HR)']]

    # scaling data
    datafile_pp = maxabs_scale(dataFile_edit)
    datafile_enc = dataFile_edit.join(enc_df)
    datafile_encpp = maxabs_scale(datafile_enc)
    # datafile_process_pp = minmax_scale(datafile_process)
    # datafile_geometry_pp = minmax_scale(datafile_geometry)
    # datafile_material_pp = minmax_scale(datafile_material)

    # Defining all strings required
    targets = ['kumar','dhenge1','dhenge2']
    colors = ['r','g','y']
    pc_arr1 = ['Principal Component 1','Principal Component 2']
    pc_arr13 = ['Principal Component 1','Principal Component 2','Principal Component 3']
    pc_arr14 = ['Principal Component 1','Principal Component 2','Principal Component 3','Principal Component 4']
    title1 = '2 component PCA'
    title2 = '3 component PCA'
    title3 = '2 component PCA with enc'
    title7 = '3 component PCA with enc'
    title8 = '4 component PCA with enc'
    # title4 = '2 component PCA process'
    # title5 = '2 component PCA geometry'
    # title6 = '2 component PCA material'
    pc_arr2 = ['Principal Component 1','Principal Component 3']
    pc_arr3 = ['Principal Component 2','Principal Component 3']
    labels = np.array(dataFile_edit.keys())
    labelsenc = np.concatenate([np.array(dataFile_edit.keys()),['Liq add position CE','Liq add position KE']])
    # using pca to get results and plots
    [pca2, finDF2] = pca_tsg(datafile_pp,y,2,pc_arr1,title1,labels)
    score_plots(finDF2,targets,colors, pc_arr1,title1)

    [pca2, finDF3] = pca_tsg(datafile_pp,y,3,pc_arr13,title2,labels)
    score_plots(finDF3,targets,colors, pc_arr1,title2)
    score_plots(finDF3,targets,colors, pc_arr2,title2)
    score_plots(finDF3,targets,colors, pc_arr3,title2)
    
    [pcaenc, finDFenc] = pca_tsg(datafile_encpp,y,2,pc_arr1,title3,labelsenc)
    score_plots(finDFenc,targets,colors, pc_arr1,title3)
    
    # [pcaenc, finDFpp] = pca_tsg(datafile_process_pp,y,2,pc_arr1,title4,labels)
    # score_plots(finDFpp,targets,colors, pc_arr1,title4)

    # [pcaenc, finDFgeo] = pca_tsg(datafile_geometry_pp,y,2,pc_arr1,title5,labels)
    # score_plots(finDFgeo,targets,colors, pc_arr1,title5)

    # [pcaenc, finDFmat] = pca_tsg(datafile_material_pp,y,2,pc_arr1,title6,labels)
    # score_plots(finDFmat,targets,colors, pc_arr1,title6)

    [pcaenc3, finDFenc3] = pca_tsg(datafile_encpp,y,3,pc_arr13,title7,labelsenc)
    # [pcaenc3, finDFenc3] = pca_tsg(datafile_encpp,y,4,pc_arr14,title8,labelsenc)

    score_plots(finDFenc3,targets,colors, pc_arr1,title7)
    score_plots(finDFenc3,targets,colors, pc_arr2,title7)
    score_plots(finDFenc3,targets,colors, pc_arr3,title7)
    
    
    # plt.show()


def pca_tsg(datafile,y,n_comps,pc_array,title,labels):
    pca = PCA(n_components=n_comps)
    prinComp = pca.fit_transform(scale(datafile))
    prinDF = pd.DataFrame(data=prinComp, columns=pc_array)
    finDF = pd.concat([prinDF,y],axis=1)
    print("\n",title,"\n")
    print(pca.explained_variance_ratio_)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=pc_array, index=labels)
    print(loading_matrix)
    return [pca, finDF]

def score_plots(final_datafile,targets,colors,pc_array,title):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(pc_array[0], fontsize = 15)
    ax.set_ylabel(pc_array[1], fontsize = 15)
    ax.set_title(title, fontsize = 20)
    for target,color in zip(targets,colors):
        indices = final_datafile['Experiments'] == target
        ax.scatter(final_datafile.loc[indices,pc_array[0]],final_datafile.loc[indices,pc_array[1]],c=color,s=50)
    ax.legend(targets)
    ax.grid()
    




if __name__ == "__main__":
    main()

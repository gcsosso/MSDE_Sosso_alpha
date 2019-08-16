# MSDE Functions

import keras
from keras.layers import Dense, Input
import PYTHON.chemfun as cf
#import PYTHON.chemfun_1 as cf1
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

import datetime

# Tensorflow graph being a nuisance and makes optimisation process slow down as the number of models increases, 
# trying to make function that resets graphs after each model training 
def ResetGraph():
    keras.backend.clear_session()

# Building the model #
def build_model(input,architecture, activation, optimiser):
    ResetGraph()
    input1 = Input(shape = (len(input.columns),))
    
    a = input1
    for layer in architecture:
         a = Dense(layer,activation = activation)(a)
            
    output1 = Dense(1,activation = 'linear')(a)

    model = keras.models.Model(inputs = input1, outputs = output1) 

    model.compile(optimizer = optimiser, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

    return model 
	
def GridSearchCVKeras(xtrain_list, xtest_list, ytrain_list, ytest_list, param_grid):
	results = {'epochs':[],
			   'architecture':[],
			   'activation':[],
			   'optimiser':[]}
			   
	for x in range(1,len(xtrain_list)+1,1):
		results['train_mae_hist_%s'%(x)] = []
		results['train_mse_hist_%s'%(x)] = []
		results['test_mae_hist_%s'%(x)] = []
		results['test_mse_hist_%s'%(x)] = []
	
	num_models = 1
	for key in param_grid.keys():
		num_models*=len(param_grid[key])
	
	counter = 1
	for architecture in param_grid['architecture']: # Grid Search
		for activation in param_grid['activation']:
			for optimiser in param_grid['optimiser']:
				for epoch in param_grid['epochs']:
		
					results['architecture'].append(architecture)
					results['activation'].append(activation)
					results['optimiser'].append(optimiser)
					results['epochs'].append([x for x in range(1,epoch+1,1)])        
				
					print('Model Number:',counter,' | Current time:', datetime.datetime.now())
					index=1
					
					for xtr, xte, ytr, yte in zip(xtrain_list, xtest_list, ytrain_list, ytest_list):
						
						model = build_model(xtr, architecture, activation, optimiser)
						
						history = model.fit(xtr,ytr, validation_data = [xte,yte],epochs = epoch, verbose = 0)
						hist = history.history
						
						results['train_mse_hist_%s'%(index)].append(hist['mean_squared_error'])
						results['train_mae_hist_%s'%(index)].append(hist['mean_absolute_error'])
						results['test_mse_hist_%s'%(index)].append(hist['val_mean_squared_error'])
						results['test_mae_hist_%s'%(index)].append(hist['val_mean_absolute_error'])
						
						ResetGraph()
						index += 1
						
					counter+=1
				
				
	return pd.DataFrame(results)
				
def get_standard_descriptors(mol_smiles, output = None, output_name = 'output'):
	mol_rdkit = cf.SMILES2MOLES(mol_smiles) # Converting smiles to RD-Kit format
	mol_rdkit_H = cf1.ADDH_MOLES(mol_rdkit) # Adding hydrogens to molecules
	
	descriptors = []
	for m in mol_rdkit_H:
		desc = []
		desc.append(Chem.GraphDescriptors.BalabanJ(m))        # Chem. Phys. Lett. 89:399-404 (1982)
		desc.append(Chem.GraphDescriptors.BertzCT(m))         # J. Am. Chem. Soc. 103:3599-601 (1981)
		desc.append(Chem.GraphDescriptors.Ipc(m))             # J. Chem. Phys. 67:4517-33 (1977) # Gives very small numbers #
		desc.append(Chem.GraphDescriptors.HallKierAlpha(m))   # Rev. Comput. Chem. 2:367-422 (1991)
		desc.append(Chem.Crippen.MolLogP(m))                  # Wildman and Crippen JCICS 39:868-73 (1999)
		desc.append(Chem.Crippen.MolMR(m))                    # Wildman and Crippen JCICS 39:868-73 (1999)
		desc.append(Chem.Descriptors.ExactMolWt(m))
		desc.append(Chem.Descriptors.FpDensityMorgan1(m))
		#desc.append(Chem.Descriptors.MaxPartialCharge(m))    # Returns null value for a molecule in lipophilicity dataset #
		desc.append(Chem.Descriptors.NumRadicalElectrons(m))
		desc.append(Chem.Descriptors.NumValenceElectrons(m))
		desc.append(Chem.Lipinski.FractionCSP3(m))
		desc.append(Chem.Lipinski.HeavyAtomCount(m))
		desc.append(Chem.Lipinski.NumHAcceptors(m))
		desc.append(Chem.Lipinski.NumAromaticRings(m))
		desc.append(Chem.Lipinski.NHOHCount(m))
		desc.append(Chem.Lipinski.NOCount(m))
		desc.append(Chem.Lipinski.NumAliphaticCarbocycles(m))
		desc.append(Chem.Lipinski.NumAliphaticHeterocycles(m))
		desc.append(Chem.Lipinski.NumAliphaticRings(m))
		desc.append(Chem.Lipinski.NumAromaticCarbocycles(m))
		desc.append(Chem.Lipinski.NumAromaticHeterocycles(m))
		desc.append(Chem.Lipinski.NumHDonors(m))
		desc.append(Chem.Lipinski.NumHeteroatoms(m))
		desc.append(Chem.Lipinski.NumRotatableBonds(m))
		desc.append(Chem.Lipinski.NumSaturatedCarbocycles(m))
		desc.append(Chem.Lipinski.NumSaturatedHeterocycles(m))
		desc.append(Chem.Lipinski.NumSaturatedRings(m))
		desc.append(Chem.Lipinski.RingCount(m))
		desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
		desc.append(Chem.rdMolDescriptors.CalcTPSA(m))        # J. Med. Chem. 43:3714-7, (2000)
		desc.append(Chem.MolSurf.LabuteASA(m))                # J. Mol. Graph. Mod. 18:464-77 (2000)
		descriptors.append(desc)
		
	descriptors_df = pd.DataFrame(data = descriptors, columns = [x for x in range(len(descriptors[0]))])
	if (outputs != None):
		descriptors_df[output_name] = output
	
	return descriptors_df

# Decomposes a list of molecules in smiles into cliques and returns a clique decompostion dataframe and the list of cliques
def get_clique_decomposition(mol_smiles, outputs= None, output_name = 'output'):
	# Generating Cliques 
	vocab=cf.Vocabulary(mol_smiles)
	size=len(vocab)
	vocabl = list(vocab)
	MolDict=cf.Vocab2Cat(vocab)
	clustersTR=cf.Clusters(mol_smiles)
	catTR=cf.Cluster2Cat(clustersTR,MolDict)
	clique_decomposition=cf.Vectorize(catTR,size)
	
	descriptors_df = pd.DataFrame(data=clique_decomposition, columns = [x for x in range(len(vocabl))])
	
	if (outputs != None):
			descriptors_df[output_name] = output
		
	return descriptors_df, vocabl	
				
def get_train_test_splits(descriptor_df, output,KFold):
	# Generating train test splits 
	splits = {'xtrain':[],
						 'xtest':[],
						 'ytrain':[],
						 'ytest':[]}
	for train_index, test_index in KFold.split(descriptor_df.iloc[:,:],output):
		splits['xtrain'].append(descriptor_df.iloc[train_index,:])
		splits['xtest'].append(descriptor_df.iloc[test_index,:])
		splits['ytrain'].append(output.iloc[train_index])
		splits['ytest'].append(output.iloc[test_index])
				
	return splits			
				
def get_average_history(history_list):
    average_history = np.zeros(len(history_list[0]))

    for kfold in history_list: # loop through KFolds
        for epoch_index, epoch_hist in zip(range(len(kfold)), kfold): # loop through epochs 
            average_history[epoch_index] += epoch_hist
        
    average_history = average_history/5
        
    return average_history
	
def get_std_history(history_list):
	std_history = np.zeros(len(history_list[0]))
	
	for x in range(len(history_list[0])):
		epoch_i_values = [] # list to store loss at epoch i across k folds
		for fold in history_list: # Loop through kfold
			epoch_i_values.append(fold[x])
		std_history[x] = np.std(epoch_i_values)
	
	return std_history
				
def work_up(GridSearchCVKerasResults):
	train_mae_averages = []
	train_mae_stds = []
	train_mae_min = []
	train_mse_averages = []
	train_mse_stds = []
	train_mse_min = []
	test_mae_averages = []
	test_mae_stds = []
	test_mae_min = []
	test_mse_averages = []
	test_mse_min = []
	test_mse_stds =[] 
	
	for row in GridSearchCVKerasResults.index:	
		train_mae_list = GridSearchCVKerasResults.loc[row,['train_mae_hist_1','train_mae_hist_2','train_mae_hist_3','train_mae_hist_4','train_mae_hist_5']]
		train_mse_list = GridSearchCVKerasResults.loc[row,['train_mse_hist_1','train_mse_hist_2','train_mse_hist_3','train_mse_hist_4','train_mse_hist_5']]
		test_mae_list = GridSearchCVKerasResults.loc[row,['test_mae_hist_1','test_mae_hist_2','test_mae_hist_3','test_mae_hist_4','test_mae_hist_5']]
		test_mse_list = GridSearchCVKerasResults.loc[row,['test_mse_hist_1','test_mse_hist_2','test_mse_hist_3','test_mse_hist_4','test_mse_hist_5']]

		train_mae_averages.append(get_average_history(train_mae_list))
		train_mse_averages.append(get_average_history(train_mse_list))
		test_mae_averages.append(get_average_history(test_mae_list))
		test_mse_averages.append(get_average_history(test_mse_list))
		
		train_mae_stds.append(get_std_history(train_mae_list))
		train_mse_stds.append(get_std_history(train_mse_list))
		test_mae_stds.append(get_std_history(test_mae_list))
		test_mse_stds.append(get_std_history(test_mse_list))
		
	for train_mae, train_mse, test_mae, test_mse in zip(train_mae_averages, train_mse_averages, test_mae_averages, test_mse_averages):    
		train_mae_min.append(min(train_mae))
		train_mse_min.append(min(train_mse))
		test_mae_min.append(min(test_mae))
		test_mse_min.append(min(test_mse))


	GridSearchCVKerasResults['train_mae_average'] = train_mae_averages
	GridSearchCVKerasResults['train_mse_average'] = train_mse_averages
	GridSearchCVKerasResults['test_mae_average'] = test_mae_averages
	GridSearchCVKerasResults['test_mse_average'] = test_mse_averages
	
	GridSearchCVKerasResults['train_mae_std'] = train_mae_stds
	GridSearchCVKerasResults['train_mse_std'] = train_mse_stds
	GridSearchCVKerasResults['test_mae_std'] = test_mae_stds
	GridSearchCVKerasResults['test_mse_std'] = test_mse_stds

	GridSearchCVKerasResults['train_mae_min'] = train_mae_min
	GridSearchCVKerasResults['train_mse_min'] = train_mse_min
	GridSearchCVKerasResults['test_mae_min'] = test_mae_min
	GridSearchCVKerasResults['test_mse_min'] = test_mse_min


	return GridSearchCVKerasResults
	
def get_best_model_index_mse(worked_up_results):
    best_ind = worked_up_results.loc[worked_up_results['test_mse_min'] == min(worked_up_results['test_mse_min'])].index[0]
    
    return best_ind

def refit_best_model(worked_up_results, splits, scaler,unscaled_outputs):
    scaler.fit(np.array(unscaled_outputs).reshape(-1,1))
    best_ind = get_best_model_index_mse(worked_up_results)
    best_arch = worked_up_results['architecture'][best_ind]
    best_acti = worked_up_results['activation'][best_ind]
    best_opti = worked_up_results['optimiser'][best_ind]
    
    df = pd.DataFrame(worked_up_results['test_mse_average'][best_ind])
    best_epoch = df.loc[df[0] == min(df[0])].index[0]
    
    tr_pred_list = []
    te_pred_list = []

    for xtr, xte, ytr, yte in zip(splits['xtrain'],splits['xtest'],
                                              splits['ytrain'],splits['ytest']):
        best_model = build_model(xtr, architecture=best_arch, activation=best_acti, optimiser=best_opti)

        best_model.fit(xtr,ytr,validation_data = [xte,yte],epochs = best_epoch,verbose = 0)
        tr_pred_list.append(best_model.predict(xtr))
        te_pred_list.append(best_model.predict(xte))
        
    tr_pred_inv_tr_list = []
    te_pred_inv_tr_list = []
    ytr_inv_tr_list = []
    yte_inv_tr_list = []

    for tr_pred, te_pred, ytr,yte in zip(tr_pred_list,te_pred_list,splits['ytrain'],splits['ytest']):

        tr_pred_inv_tr_list.append(scaler.inverse_transform(tr_pred))
        te_pred_inv_tr_list.append(scaler.inverse_transform(te_pred))
        ytr_inv_tr_list.append(scaler.inverse_transform(ytr))
        yte_inv_tr_list.append(scaler.inverse_transform(yte))

    
    
    return tr_pred_inv_tr_list, te_pred_inv_tr_list, ytr_inv_tr_list, yte_inv_tr_list


def get_GPModel(data_splits,scaler,LS,optimize=0,method='lbfgsb',restarts=0,LS_bounds=[1e-5,1e5],noise=1e-5):
    N=len(data_splits['xtrain'])

    LS_opt=[]
    predTR=[]
    predTE=[]

    for i in range(0,N):
        xtrain=np.array(data_splits['xtrain'][i])
        xtest=np.array(data_splits['xtest'][i])
        ytrain=np.array(data_splits['ytrain'][i]).reshape(-1,1)
        ytest=np.array(data_splits['ytest'][i]).reshape(-1,1)    

        # Dimension of our descriptor
        D=xtrain.shape[1] 

        kernel = GPy.kern.RBF(input_dim=D,ARD=True,variance=1,lengthscale=LS[i])
        m=GPy.models.GPRegression(xtrain,ytrain,kernel)
        m.constrain_bounded(LS_bounds[0],LS_bounds[1]) # bound lengthscales
        m.kern.variance.fix() # fixing the variance 
        
        if noise>0:
            m.likelihood.variance.fix(noise) # fixing the noise param
        
        if optimize==1:
            m.optimize(optimizer=method,messages=False,max_iters=10000)
        
        if restarts>0:
            m.optimize_restarts(optimizer=method,messages=False,max_iters=10000,num_restarts=restarts,verbose=False)
        
        LS_opt.append(np.array(m.rbf.lengthscale)) # Append optimized len
        
        #print(m)
        
        # Appending model predictions
        predTR.append(scaler.inverse_transform(m.predict(xtrain)))
        predTE.append(scaler.inverse_transform(m.predict(xtest)))
        
    return predTR,predTE,LS_opt


def GP_RFplot(data_splits,predTR,predTE,scaler,data_name='Hepatocytes',model='RF',alpha=0.35):
    N=len(data_splits['ytrain'])
    TR_PRED=[]
    TR_EXP=[]
    TE_PRED=[]
    TE_EXP=[]

    fig = plt.figure(figsize=[3,3],edgecolor='k',dpi = 200)
    #sns.set_style("white")
    for i in range(0,N):
        TR_EXP=TR_EXP+list(scaler.inverse_transform(data_splits['ytrain'][i]))
        TE_EXP=TE_EXP+list(scaler.inverse_transform(data_splits['ytest'][i]))
        if model=='GP':
            TR_PRED=TR_PRED+list(predTR[i][0])
            TE_PRED=TE_PRED+list(predTE[i][0])
        else:
            TR_PRED=TR_PRED+list(predTR[i])
            TE_PRED=TE_PRED+list(predTE[i])

    plt.scatter(TR_EXP,TR_PRED,label='Train Set',s=15)
    plt.scatter(TE_EXP,TE_PRED,label='Test Set',s=15,alpha=alpha)
    plt.legend(fontsize=8, prop={'size':6})
    plt.xlabel('Experimental',fontsize=8)
    plt.ylabel('Model Prediction',fontsize=8)
    if model=='GP':
        acc=np.linspace(min(predTR[0][0])-0.15,max(predTR[0][0]+0.15),1000)
        plt.plot(acc,acc,label = 'Accuracy',color='green',linewidth=1) 
        plt.title('%s Gaussian Process' %data_name,fontsize = 8)
    else:
        acc=np.linspace(min(predTR[0])-0.15,max(predTR[0])+0.15,1000)
        plt.plot(acc,acc,label = 'Accuracy',color='green',linewidth=1) 
        plt.title('%s Random Forests' %data_name,fontsize = 8)
                
    return None
    
    
def GP_MSE(data_splits,predTR,predTE,scaler):
    TR_mse=[]
    TE_mse=[]

    for i in range(0,5):
        TR_mse.append(mse(scaler.inverse_transform(data_splits['ytrain'][i]),predTR[i][0,:]))
        TE_mse.append(mse(scaler.inverse_transform(data_splits['ytest'][i]),predTE[i][0,:]))

    mse_df=pd.DataFrame(data=[TR_mse,TE_mse]).T
    mse_df.columns=['train_mse','test_mse']
    print(mse_df) 
    return None


# Borrowed this code from www.geeksforgeeks.org
# Sorts a list based on a second list 
def sort_list(list1,list2):   
    zipped_pairs=zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 


def plot_all_lengthscales(lengthscales,xlim,ylim,title):
    N=len(lengthscales[0])
    fig = plt.figure(figsize=[25,15],edgecolor='k',dpi = 200)
    sns.set_style("white")
    xindex=np.arange(N)
    width=2.5
    plt.bar(xindex,lengthscales[0],width,color='navy',label='Split 1')
    plt.bar(xindex,lengthscales[1],width,color='red',alpha=0.9,label='Split 2')
    plt.bar(xindex,lengthscales[2],width,color='green',alpha=1.0,label='Split 3')
    plt.bar(xindex,lengthscales[3],width,color='dodgerblue',alpha=0.8,label='Split 4')
    plt.bar(xindex,lengthscales[4],width,color='gold',alpha=0.7,label='Split 5')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel('Lengthscale Index',fontsize=20)
    plt.ylabel('Lengthscale Value',fontsize=20)
    plt.legend(loc='best',prop={'size':20})
    plt.title('%s Lengthscale Comparison by Split' %title,fontsize=25)
    return None


def plot_particular_lengthscales(lengthscales,indices,title):
    N=len(lengthscales[0])
    fig=plt.figure(figsize=[20,8],edgecolor='k',dpi = 200)
    ax=plt.axes()
    sns.set_style("white")
    width=2.5
    ax.bar(indices,lengthscales[0][indices],width,color='navy',label='Split 1')
    ax.bar(indices,lengthscales[1][indices],width,color='red',alpha=0.9,label='Split 2')
    ax.bar(indices,lengthscales[2][indices],width,color='green',alpha=1.0,label='Split 3')
    ax.bar(indices,lengthscales[3][indices],width,color='dodgerblue',alpha=0.8,label='Split 4')
    ax.bar(indices,lengthscales[4][indices],width,color='yellow',alpha=0.6,label='Split 5')
    if max(max(lengthscales[0][indices]),max(lengthscales[1][indices]),max(lengthscales[2][indices]),max(lengthscales[3][indices]),max(lengthscales[4][indices]))<800:
        plt.ylim(0,max(max(lengthscales[0][indices]),max(lengthscales[1][indices]),max(lengthscales[2][indices]),max(lengthscales[3][indices]),max(lengthscales[4][indices]))*1.05)
    else:
        plt.ylim(0,max(max(lengthscales[0][indices]),max(lengthscales[1][indices]),max(lengthscales[2][indices]),max(lengthscales[3][indices]),max(lengthscales[4][indices]))*0.25)    
    plt.xlim(-1,max(np.array(indices))+5)
    plt.xlabel('Lengthscale Index',fontsize=20)
    plt.ylabel('Lengthscale Value',fontsize=20)
    plt.legend(loc='best',prop={'size':20})
    plt.title('%s Lengthscale Comparison by Split' %title,fontsize=25)
    ax.set_xticks(indices)
    return None


def mean_lengthscale(lengthscales):
    mean_LS=np.mean(lengthscales,0)
    std_LS=np.std(lengthscales,0)
    return mean_LS,std_LS


def sort_lengthscales(lengthscales,k_splits):
    N=len(lengthscales[0])
    sortLS=[]
    for i in range(0,k_splits):
        split=lengthscales[i]
        indices=split.argsort() # smallest lengthscales (indices in ascending order based on LS value)
        sortLS.append(indices) #Lists the lengthscale ranks

    meanrank=[]
    for j in range(0,N):
        indexsum=0
        for k in range(0,k_splits):
            indexsum+=np.where(sortLS[k]==j)[0][0]
        mean=indexsum/5
        meanrank.append(mean)
    
    meanrank=np.array(meanrank) 
    sortedcliques=meanrank.argsort() #Sorts the indices of ls from smallest to largest
    
    return sortedcliques,meanrank


def get_LS_df(indices,vocab,lengthscales):
    class dictionary(dict): 
        # __init__ function 
        def __init__(self): 
            self=dict() 
        # Function to add key:value 
        def add(self, key, value): 
            self[key]=value 

    smiles=[vocab[i] for i in indices]
    LS_mean=np.take(np.mean(lengthscales,0),indices)
    LS_std=np.take(np.std(lengthscales,0),indices)
            
    LSdict=dictionary()

    LSdict.add('vocab_index',indices)
    LSdict.add('smiles',smiles)
    LSdict.add('mean_LS',LS_mean)
    LSdict.add('std_LS',LS_std)
        
    for j in np.arange(0,len(lengthscales)):
        LSdict.add('split%s'%(j+1),np.take(lengthscales[j],indices))
        
    return pd.DataFrame(LSdict)


def get_RFmodel(data_splits,scaler):
# Instantiate model with 10000 decision trees

    N=len(data_splits['xtrain'])
    rf=RandomForestRegressor(n_estimators=10000,random_state=10)

    RFpredTR=[]
    RFpredTE=[]
    RFgini=[]

    for i in range(0,N):
        xtrain=np.array(data_splits['xtrain'][i])
        xtest=np.array(data_splits['xtest'][i])
        ytrain=np.array(data_splits['ytrain'][i]).reshape(-1,1)
        ytest=np.array(data_splits['ytest'][i]).reshape(-1,1)    

        #Train the model on training data
        rf.fit(xtrain,ytrain)

        #Use the forest's predict method on the test data
        RFpredTR.append(scaler.inverse_transform(rf.predict(xtrain)))
        RFpredTE.append(scaler.inverse_transform(rf.predict(xtest)))
        RFgini.append(rf.feature_importances_)
        
    return RFpredTR,RFpredTE,RFgini


def RF_MSE(data_splits,predTR,predTE,scaler):
    N=len(data_splits['ytrain'])
    # MSE analysis
    TR_mse_RF=[]
    TE_mse_RF=[]

    for i in range(0,N):
        TR_mse_RF.append(mse(scaler.inverse_transform(data_splits['ytrain'][i]),predTR[i]))
        TE_mse_RF.append(mse(scaler.inverse_transform(data_splits['ytest'][i]),predTE[i]))

    mse_RF_df=pd.DataFrame(data=[TR_mse_RF,TE_mse_RF]).T
    mse_RF_df.columns=['train_mse','test_mse']
    print(mse_RF_df)    
    return None


def sort_gini(gini,k_splits):
    N=len(gini[0])

    sorted_gini=[]
    for i in range(0,k_splits):
        split=gini[i]
        indices=split.argsort() #Sorts indices of gini values from smallest to largest
        sorted_gini.append(indices)

    indexmean_gini=[]
    for j in range(0,N):
        indexsum=0
        for k in range(0,k_splits):
            indexsum+=np.where(sorted_gini[k]==j)[0][0]
        mean=indexsum/5
        indexmean_gini.append(mean)
    
    indexmean_gini=np.array(indexmean_gini) # Actually use this list to sort the cliques

    #Sorted array of cliques from smallest gini to largest
    gini_rank=indexmean_gini.argsort()[::-1]
    
    return sorted_gini,gini_rank


def get_gini_df(indices,vocab,gini):
    class dictionary(dict): 
        # __init__ function 
        def __init__(self): 
            self=dict() 
        # Function to add key:value 
        def add(self, key, value): 
            self[key]=value 

    smiles=[vocab[i] for i in indices]
    gini_mean=np.take(np.mean(gini,0),indices)
    gini_std=np.take(np.std(gini,0),indices)
            
    GINIdict=dictionary()
    
    GINIdict.add('vocab_index',indices)
    GINIdict.add('smiles',smiles)
    GINIdict.add('mean_gini',gini_mean)
    GINIdict.add('std_gini',gini_std)

    for j in np.arange(0,len(gini)):
        GINIdict.add('split%s'%(j+1),np.take(gini[j],indices))
    
    return pd.DataFrame(GINIdict)


def plot_gini(gini_df,dataname='Lipo'):
    xlabels=list(gini_df['vocab_index'])
    gini_std=list(gini_df['std_gini'])
    gini_mean=list(gini_df['mean_gini'])
    x_position=np.arange(0,len(xlabels))
    #fig = plt.figure(figsize=[3,3],edgecolor='k',dpi = 200)
    fig,ax=plt.subplots(figsize=[10,8],edgecolor='k',dpi = 200)
    ax.bar(x_position,gini_mean,yerr=gini_std,align='center',alpha=1.0)#,ecolor='black')
    ax.set_ylabel('Mean gini value')
    ax.set_xlabel('Clique Index')
    ax.set_xticks(x_position)
    ax.set_xticklabels(xlabels)
    ax.set_title('Mean/Std of %s RF gini values' %dataname)
    ax.yaxis.grid(True)
    return None


# Lets make some predictions on the various dataset/descriptor combinations using the best models as determined by the 
# optimisation process 

def make_predictions(unscaled_output, splits, best_model_results, scaler):
    results = {}
    results['ytr_scaled'] = [] # place to store scaled train set outputs for splits 1-5
    results['yte_scaled'] = [] # place to store scaled test set outputs for splits 1-5
    results['tr_preds_scaled'] = [] # place to store scaled train predictions for splits 1-5
    results['te_preds_scaled'] = [] # place to store scaled test predictions for splits 1-5
    results['ytr'] = [] # place to store train set outputs for splits 1-5
    results['yte'] = [] # place to store test set outputs for splits 1-5 
    results['tr_preds'] = [] # place to store normal output space train predictions for splits 1-5
    results['te_preds'] = [] # place to store normal output space test predictions for splits 1-5
    
    best_arch = best_model_results['architecture'].values[0] # Getting best architecture
    best_acti = best_model_results['activation'].values[0] # Getting best activation function
    best_opti = best_model_results['optimiser'].values[0] # Getting best optimiser 
    ind = np.argmin(best_model_results['test_mse_average'].values[0])
    best_epoch = best_model_results['epoch'].values[0][ind] # Getting best number of epochs 

    scaler.fit(np.array(unscaled_output).reshape(-1,1)) # To transform scaled splits back to original output space
    
    for xtr, xte, ytr,yte, index in zip(splits['xtrain'],splits['xtest'],splits['ytrain'],splits['ytest'], 
                                       [x for x in range(1,6,1)]):
        model = build_model(xtr, best_arch, best_acti, best_opti)
        
        model.fit(xtr,ytr, validation_data = [xte,yte], epochs = best_epoch, verbose = 0)
        
        tr_pred = model.predict(xtr)
        te_pred = model.predict(xte)
        
        
        results['ytr_scaled'].append(ytr)
        results['yte_scaled'].append(yte)
        results['tr_preds_scaled'].append(tr_pred)
        results['te_preds_scaled'].append(te_pred)
        results['ytr'].append(scaler.inverse_transform(np.array(ytr).reshape(-1,1)))
        results['yte'].append(scaler.inverse_transform(np.array(yte).reshape(-1,1)))
        results['tr_preds'].append(scaler.inverse_transform(tr_pred))
        results['te_preds'].append(scaler.inverse_transform(te_pred))
        
    return pd.DataFrame(results)
	
def plot_results(predictions,fig_title):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=[12,6],dpi=200)
    plt.suptitle(fig_title)
    ax1.set_ylabel('Predicted Value')
    ax1.set_xlabel('Experimental Value')

    # Plotting predicted vs experimental 
    acc = np.linspace(min(predictions['ytr'][0]),max(predictions['ytr'][0]),1000)
    ax1.plot(acc,acc)
    for ytr,tr_pred in zip(predictions['ytr'],predictions['tr_preds']):
        tr_scat = ax1.scatter(ytr,tr_pred,label='Train Set',color='Blue',alpha=1)

    for yte,te_pred in zip(predictions['yte'],predictions['te_preds']):
        te_scat = ax1.scatter(yte,te_pred,label='Test Set',color='orange',alpha=0.5)
    
    ax1.legend([tr_scat,te_scat],['Train Predictions','Test Predictions'])
    
    # Plotting distribution of errors 
    train_errors = []
    test_errors = []
    for ytr_split, tr_pred_split in zip(predictions['ytr'],predictions['tr_preds']):
        for ytr,tr_pred in zip(ytr_split,tr_pred_split):
            train_errors.append(float((ytr-tr_pred)))
    for yte_split, te_pred_split in zip(predictions['yte'],predictions['te_preds']):
        for yte,te_pred in zip(yte_split,te_pred_split):
            test_errors.append(float((yte-te_pred)))
            
    sns.distplot(train_errors, label="Train Errors")
    sns.distplot(test_errors, label="Test Errors")
    
    ax2.legend()
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Counts')
    
# Function that returns average mse, mae, and Pearson's R for a set of predictions, in form of dataframe 
def get_ave_metrics(predictions,name):
    mse_train_list = []
    mae_train_list = []
    r_train_list = []
    
    mse_test_list = []
    mae_test_list = []
    r_test_list = []
    
    for ytr,tr_pred,yte,te_pred in zip(predictions['ytr'],predictions['tr_preds'],
                                       predictions['yte'],predictions['te_preds']):
        mse_train_list.append(mse(ytr,tr_pred))
        mae_train_list.append(mae(ytr,tr_pred))
        r_train_list.append(pearsonr(ytr,tr_pred)[0])
        
        mse_test_list.append(mse(yte,te_pred))
        mae_test_list.append(mae(yte,te_pred))
        r_test_list.append(pearsonr(yte,te_pred)[0])
        
        
    results = {'mse_train_ave':0,
               'mse_train_std':0,
               'mae_train_ave':0,
               'mae_train_std':0,
               'pearsonr_train_ave':0,
               'pearsonr_train_std':0,
               'mse_test_ave':0,
               'mse_test_std':0,
               'mae_test_ave':0,
               'mae_test_std':0,
               'pearsonr_test_ave':0,
               'pearsonr_test_std':0}
    
    results['mse_train_ave']=np.average(mse_train_list)
    results['mse_train_std']=np.std(mse_train_list)
    results['mae_train_ave']=np.average(mae_train_list)
    results['mae_train_std']=np.std(mae_train_list)
    r_train_list = np.array(r_train_list).reshape(-1)
    results['pearsonr_train_ave']=np.average(r_train_list)
    results['pearsonr_train_std']=np.std(r_train_list)
    
    results['mse_test_ave']=np.average(mse_test_list)
    results['mse_test_std']=np.std(mse_test_list)
    results['mae_test_ave']=np.average(mae_test_list)
    results['mae_test_std']=np.std(mae_test_list)
    r_test_list = np.array(r_test_list).reshape(-1)
    results['pearsonr_test_ave']=np.average(r_test_list)
    results['pearsonr_test_std']=np.std(r_test_list)
    
    return pd.DataFrame(results,index=[name]).T
	
def plot_bar(metrics_df,metric):
    fig = plt.figure(figsize=[16,12])
    plt.bar(metrics_df.columns,metrics_df.loc[metric+'_ave',:],edgecolor='k',color='orange',
            yerr=metrics_df.loc[metric+'_std',:])
    plt.ylabel(metric,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=12 )
    plt.title(metric+' across each dataset/descriptor combination',fontsize=20)
    

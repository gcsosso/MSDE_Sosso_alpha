# MSDE Functions

import keras
from keras.layers import Dense, Input
import PYTHON.chemfun as cf
#import PYTHON.chemfun_1 as cf1
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

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

def plot_results(worked_up_results,splits, scaler,unscaled_outputs,fig_title):
    best_ind = get_best_model_index_mse(worked_up_results)
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=[12,16],dpi=200)
    plt.suptitle(fig_title)

    # PLotting Model Accuracies
    ax1.bar([x for x in range(len(worked_up_results['test_mse_min']))],
            worked_up_results['test_mse_min'],color='black',edgecolor='orange')
    #ax1.set_ylim(min(worked_up_results['test_mse_min'])-0.01,max(worked_up_results['test_mse_min'])+0.01)
    ax1.set_title('Model Performance')
    ax1.set_ylabel('Mean Squared Error (scaled units)')
    ax1.set_xlabel('Model Number')
    
    # Plotting loss history 
    ax2.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['test_mse_average'][best_ind], label = 'Test Loss History')
    ax2.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['train_mse_average'][best_ind], label = 'Train Loss History')
    ax2.set_title('Best Model history')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_xlabel('Epoch')

    # Plotting error bars train
    ax3.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['train_mse_average'][best_ind], label = 'Train Loss History',color='k')
    ax3.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['train_mse_average'][best_ind]+worked_up_results['train_mse_std'][best_ind],color='k')
    ax3.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['train_mse_average'][best_ind]-worked_up_results['train_mse_std'][best_ind],color='k')
    ax3.fill_between(worked_up_results['epoch'][best_ind], worked_up_results['train_mse_average'][best_ind]-worked_up_results['train_mse_std'][best_ind],
                     worked_up_results['train_mse_average'][best_ind]+worked_up_results['train_mse_std'][best_ind])
    ax3.set_title('Best Model history Train with Errors')
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_xlabel('Epoch')
    
    # Plotting error bars train
    ax4.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['test_mse_average'][best_ind], label = 'Test Loss History',color='k')
    ax4.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['test_mse_average'][best_ind]+worked_up_results['test_mse_std'][best_ind],color='k')
    ax4.plot(worked_up_results['epoch'][best_ind], 
             worked_up_results['test_mse_average'][best_ind]-worked_up_results['test_mse_std'][best_ind],color='k')
    ax4.fill_between(worked_up_results['epoch'][best_ind], worked_up_results['test_mse_average'][best_ind]-worked_up_results['test_mse_std'][best_ind],
                     worked_up_results['test_mse_average'][best_ind]+worked_up_results['test_mse_std'][best_ind])
    ax4.set_title('Best Model history Test with Errors')
    ax4.set_ylabel('Mean Squared Error')
    ax4.set_xlabel('Epoch')
    
    # Plotting predicted vs experimental 
    tr_pred, te_pred, ytr, yte = refit_best_model(worked_up_results, splits, scaler, unscaled_outputs)
    acc = np.linspace(min(ytr[0]),max(ytr[0]),1000)
    ax5.plot(acc,acc)
    ax5.scatter(ytr[0],tr_pred[0],label='Train Set')
    ax5.scatter(yte[0],te_pred[0],label='Test Set')
    ax5.set_ylabel('Predicted Value')
    ax5.set_xlabel('Experimental Value')
    
    # Plotting distribution of errors
    train_errors = []
    test_errors = []

    for tr_pred, te_pred, ytr, yte in zip(tr_pred[0],te_pred[0],ytr[0],yte[0]):
        train_errors.append(ytr-tr_pred)
        test_errors.append(yte-te_pred)
    train_errors = np.array(train_errors).reshape(-1)
    test_errors = np.array(test_errors).reshape(-1)
    
    #ax6.hist(train_errors)
    #ax6.hist(test_errors)
    
    sns.distplot(train_errors , color="skyblue", label="Train Errors")
    sns.distplot(test_errors , color="red", label="Test Errors")
    ax6.set_title('Distribution of prediction errors')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    

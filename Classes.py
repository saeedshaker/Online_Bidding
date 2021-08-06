import streamlit as st
import itertools
#from itertools import product
import numpy as np
import pandas as pd
import scipy
#from scipy.optimize import minimize
#from scipy.optimize import Bounds
#from scipy import optimize
#from collections import Counter
import collections
import sklearn
from sklearn.preprocessing import OneHotEncoder

#======================================================================================
class MixtureGaussian2():
    def __init__(self):
        return None
    def rvs(self, loc1, scale1, loc2, scale2, weight, size, random_state=None):
        if weight == None: weight = 0.5
        weight = max(min(weight, 1), 0)
        if len(size) == 2: 
            size1 = [int(weight*size[0]), size[1]]
            size2 = [size[0]-size1[0], size[1]]
        rvs1 = np.random.normal(loc=loc1, scale=scale1, size=size1)
        rvs2 = np.random.normal(loc=loc2, scale=scale2, size=size2)
        rvs_concat = np.concatenate((rvs1,rvs2))
        np.random.shuffle(rvs_concat)
        self.rvs_out = rvs_concat
        return self.rvs_out

#======================================================================================
def pick_rv_generator_params(rv_gen='mixture_gaussian'):
    if rv_gen == 'mixture_gaussian':
        rv_gen_func = MixtureGaussian2().rvs
        dist_params_init = {'loc1_init': 8, 'scale1_init': 2,
                            'loc2_init': 12, 'scale2_init': 3, 'weight_init': 0.3}
        dist_params_bounds = {'loc1_bounds': [-5,30], 'scale1_bounds': [.1,15],
                              'loc2_bounds': [-5,30], 'scale2_bounds': [.1,15], 'weight_bounds': [0, 1]}
    elif rv_gen == 'skewnorm':
        rv_gen_func = skewnorm.rvs
        dist_params_init = {'a_init': 0, 'loc_init': 10, 'scale_init': 2}
        dist_params_bounds = {'a_bounds': [-10,10], 'loc_bounds': [1,20], 'scale_bounds': [.1,10]}
    elif rv_gen == 'gengamma':
        rv_gen_func = gengamma.rvs
        dist_params_init = {'a_init': 1, 'c_init': 1,
                            'loc_init': 8, 'scale_init': 2}
        dist_params_bounds = {'a_bounds': [.01,50], 'c_bounds': [.1,50],
                              'loc_bounds': [1,20], 'scale_bounds': [.1,20]}
    return rv_gen_func, dist_params_init, dist_params_bounds    
    
#======================================================================================
class SimulationEstimator():
    def __init__(self, distribution, params):
        self.distribution = distribution
        self.params = params
        
    def run_all(self, bid, rank, num_competitors=None):
        if not num_competitors: num_competitors = max(rank)-1
        self.num_competitors = num_competitors
        self.simulate_competitors(bid, rank)
        self.get_simulated_ranks(bid)
        self.get_estimation_error(rank)        
        
    def simulate_competitors(self, bids, ranks, bid_min=0, bid_max=None):
        if not bid_max: bid_max = max(bids)*3
        n = len(bids)
        sim_bids = self.distribution(*self.params, size=[n, self.num_competitors], random_state=200)
        sim_bids[sim_bids < bid_min] = bid_min
        sim_bids[sim_bids > bid_max] = bid_max
        self.simulated_bids = sim_bids
    
    def get_simulated_ranks(self, bids):
        beaten_bids = [[
            (competitor_bid >= our_bid)*1.0 
            for competitor_bid, our_bid in zip(compet_bids, bids) ] 
            for compet_bids in self.simulated_bids]
        self.simulated_ranks = np.sum(beaten_bids, axis=1)
        self.simulated_ranks += np.ones(len(self.simulated_ranks))
    
    def get_estimation_error(self, ranks, rank_weights = None):
        max_rank = max(ranks)
        rank_hist_errors0 = []
        ranks_count_data = {int(kv[0]):kv[1] for kv in collections.Counter(ranks).items()}
        ranks_count_simu = {int(kv[0]):kv[1] for kv in collections.Counter(self.simulated_ranks).items()}
        self.rank_hist_errors = [np.abs(ranks_count_data.get(k,0)-ranks_count_simu.get(k,0)) for k in range(1,max_rank+1)]
        if not rank_weights: rank_weights = [1/((w+1)**0.8) for w in range(len(self.rank_hist_errors))]
        self.total_error = sum([r*w for r, w in zip(self.rank_hist_errors, rank_weights)])
        
#======================================================================================
class CrossValidator():
    def __init__(self, estimator, rv_generator_name, bid, rank, num_competitors=None):
        self.estimator = estimator
        self.distribution_rvs, self.dist_params_init, self.dist_params_bounds = pick_rv_generator_params(rv_generator_name)
        self.bid = bid
        self.rank = rank
        if not num_competitors: num_competitors = max(rank)-1
        self.num_competitors = num_competitors
    
    def run_estimator(self, current_dist_params):
        est = self.estimator(self.distribution_rvs, current_dist_params)
        est.run_all(self.bid, self.rank, self.num_competitors)
        return est.total_error
    
    def make_params_space(self):
        params_list = [val for key, val in self.dist_params_dict.items()]        
        self.dist_params_space = [param for param in itertools.product(*params_list)]
        
    def grid_search(self, dist_params_dict):
        self.min_error = 10e10
        self.dist_params_dict = dist_params_dict
        self.make_params_space()
        paramspace_indices = range(len(self.dist_params_space))
        self.errorslist = Parallel(n_jobs=-2)(delayed(self.run_estimator)(idx) for idx in tqdm(self.dist_params_space))
        best_inds = np.argmin(self.errorslist)
        if isinstance(best_inds, list):
            best_est_idx = best_inds[0]
        else:
            best_est_idx = best_inds
        self.min_error = self.errorslist[best_est_idx]
        self.best_params = self.dist_params_space[best_est_idx]
        
    def optimized_search(self, optimizer_func, method='trust-constr', max_iter=None):
        #bounds for minimize; bounds_pairs for shgo, etc
        #bounds = Bounds([float(val[0]) for key, val in self.dist_params_bounds.items()], 
        #                [float(val[1]) for key, val in self.dist_params_bounds.items()])
        if not max_iter: options = {'maxiter':max_iter}
        else: options = None
        bounds_pairs = [(float(val[0]),float(val[1])) for key, val in self.dist_params_bounds.items()]
        x0 = np.array([float(val) for key, val in self.dist_params_init.items()])
        if optimizer_func == scipy.optimize.minimize:
            self.best_params = optimizer_func(
                self.run_estimator, x0=x0, method=method, bounds=bounds_pairs, options=options)
        elif self.dist_params_bounds:
            self.best_params = optimizer_func(self.run_estimator, bounds=bounds_pairs) #shgo, ...
        else:
            self.best_params = optimizer_func(F=self.run_estimator, xin=x0) #broyden, ...
        self.best_estimator = self.estimator(self.distribution_rvs, self.best_params.x)
        
#======================================================================================
class CustomerGroupsMaker():
    def __init__(self, df_in, columns_to_ignore=None, columns_to_pass_on=None, columns_bid_rank=None):
        self.N = len(df_in)
        self.df_in = df_in
        self.columns_to_ignore = columns_to_ignore #list of columns that won't be used in estimation like index, id, etc,
        self.columns_to_pass_on = columns_to_pass_on #list of important columns like click, policy sold, etc, not used for categorization
        self.columns_bid_rank = columns_bid_rank # list of bid and rank columns names
        self.df_out = pd.DataFrame(self.df_in[columns_bid_rank + columns_to_pass_on]).reset_index(drop=True)
        self.df_out_column_description = {} #explain the range for numerical columns so that it is easier to 
        #present to user to choose from when entering data for prediction/analysis
        #similarly for categorical variables
        return
    def make_groups(self): #check_all_column_types and run the appropriate function
        for column_name in self.df_in.columns:
            if self.columns_to_ignore and column_name in self.columns_to_ignore: continue
            if self.columns_to_pass_on and column_name in self.columns_to_pass_on: continue
            if self.columns_bid_rank and column_name in self.columns_bid_rank: continue
            column = pd.DataFrame(pd.to_numeric(self.df_in[column_name], errors='ignore')).reset_index(drop=True)
            num_groups_in_column = len(list(column.groupby(column_name)[column_name].count()))
            if num_groups_in_column <= 1: continue
            if pd.api.types.is_numeric_dtype(column[column_name]):
                if num_groups_in_column > 3:
                    self.numerical_continuous(column, quantiles_num=int(np.round(np.log(num_groups_in_column+2))))
                else:
                    self.categorical_onehotenc(column)
            else:
                self.categorical_onehotenc(column)
                
        self.used_column_names = list(self.df_out_column_description.keys())
        used_column_values_list = [list(v['values'].keys()) for k, v in self.df_out_column_description.items() ]
        self.customer_groups = list(itertools.product(*used_column_values_list))
            
    def categorical_onehotenc(self, column_in, column_type='categorical', category_values_dict=None):
        column_name = column_in.columns[0]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        column_encoded = encoder.fit_transform(column_in)
        
        self.df_out_column_description[column_name] = {'type': column_type}
        self.df_out_column_description[column_name].update({'values': {c: None for c in encoder.categories_[0]}})
        if category_values_dict:
            self.df_out_column_description[column_name]['values'] = category_values_dict
            
        if self.df_out.empty:
            self.df_out = column_in.reset_index(drop=True)
        else:
            self.df_out = self.df_out.merge(column_in, left_index=True, right_index=True)

    def numerical_continuous(self, column_in, quantiles_num=2):
        column_name = column_in.columns[0]
        quantiles = np.linspace(0,1,quantiles_num+1)
        quantile_values = list(column_in.quantile(quantiles).iloc[:,0])
        categorized_column = ['Q']*self.N
        for i in range(self.N):
            Qnt_str = ['Q'+str(qnt+1) for qnt in range(quantiles_num) if column_in.iloc[i][0] >= quantile_values[qnt]]
            categorized_column[i] = Qnt_str[-1]
        values_dict = {'Q'+str(qnt+1):[quantile_values[qnt], quantile_values[qnt+1]] for qnt in range(quantiles_num)}
        categorized_column_df = pd.DataFrame(categorized_column, columns=[column_name]).reset_index(drop=True)
        self.categorical_onehotenc(categorized_column_df, 'numerical', values_dict)

#======================================================================================
class GroupbyEstimator():
    
    def __init__(self, fitted_cgm, estimator_factory, rv_generator_name, optimization_method='COBYLA', group=None):
        self.fitted_cgm = fitted_cgm
        self.estimator_factory = estimator_factory
        self.rv_generator_name = rv_generator_name
        self.optimization_method = optimization_method
        self.group = group
    
    def make_group_estimators(self):
        self.group_estimators = {}
        self.group_length = {}
        if self.group==None:
            for group in self.fitted_cgm.customer_groups:
                df_group_temp = self.fitted_cgm.df_out
                for g_item_idx in range(len(group)):
                    df_group_temp = df_group_temp.loc[
                        df_group_temp[self.fitted_cgm.used_column_names[g_item_idx]] == group[g_item_idx]]

                bid = df_group_temp[self.fitted_cgm.columns_bid_rank[0]]
                rank = df_group_temp[self.fitted_cgm.columns_bid_rank[1]]
                cv = CrossValidator(estimator=self.estimator_factory, 
                                    rv_generator_name=self.rv_generator_name,
                                    bid=bid, 
                                    rank=rank, 
                                    num_competitors=max(rank)-1)

                #method='trust-constr, SLSQP', TNC, Nelder-Mead, COBYLA
                max_iter0 = None
                if self.optimization_method in ['trust-constr', 'SLSQP', 'Nelder-Mead']: max_iter0 = 50
                cv.optimized_search(scipy.optimize.minimize, method=self.optimization_method, max_iter=max_iter0)
                simest = cv.best_estimator
                simest.run_all(bid, rank, num_competitors=None) 
                self.group_length[group] = len(rank)
                self.group_estimators[group] = simest
        else:
            group = self.group
            df_group_temp = self.fitted_cgm.df_out
            for g_item_idx in range(len(group)):
                df_group_temp = df_group_temp.loc[
                    df_group_temp[self.fitted_cgm.used_column_names[g_item_idx]] == group[g_item_idx]]

            bid = df_group_temp[self.fitted_cgm.columns_bid_rank[0]]
            rank = df_group_temp[self.fitted_cgm.columns_bid_rank[1]]
            cv = CrossValidator(estimator=self.estimator_factory, 
                                rv_generator_name=self.rv_generator_name,
                                bid=bid, 
                                rank=rank, 
                                num_competitors=max(rank)-1)
            max_iter0 = None
            if self.optimization_method in ['trust-constr', 'SLSQP', 'Nelder-Mead']: max_iter0 = 50
            cv.optimized_search(scipy.optimize.minimize, method=self.optimization_method, max_iter=max_iter0)
            simest = cv.best_estimator
            simest.run_all(bid, rank, num_competitors=None) 
            self.group_length[group] = len(rank)
            self.group_estimators[group] = simest
    
    def predict_ranks(self, group, bid_amount):
        bid = [bid_amount]*self.group_length[group]
        self.group_estimators[group].get_simulated_ranks(bid)
        return self.group_estimators[group].simulated_ranks
    
#======================================================================================

#======================================================================================

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
from Classes import *

#state_dict
def app():

    #state_dict = {'upload': 0, 'confirm1': 0, 'conf1_state': {}, 'confirm1': 0, 'conf2_state': {} }
    
    st.title("Check it out using your dataset...")
    
    st.sidebar.markdown("""[**See on my GitHub**](https://github.com/saeedshaker/capstone_TDI)""")
    
    st.markdown("### See the `Demo` tab before trying this.")
    #if state_dict['upload'] == 0:
    #    state_dict['upload'] = 1
    
    uploaded_file = st.file_uploader('Upload your auction data', type=['csv', 'xls'], accept_multiple_files=False, key=None)
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df['cost_made_up'] = np.random.randint(5,20,10000)
        st.dataframe(df.head())
        #st.write("Let's specify some features and how they should be used:")


        st.markdown("""## Specify the feature categories.
                       \n Specify `bid` and `rank` columns.""")

        col1, col2 = st.beta_columns(2)
        with col1:
            bidcolname = st.selectbox("Choose bid amount column", list(df.columns)) #2, 9
        with col2:
            rankcolname = st.selectbox("Choose rank column", list(df.columns))

        st.markdown(""" Specify `click` and `purchase` columns. (these are binary)""")

        col1_1, col1_2 = st.beta_columns(2)
        with col1_1:
            clickcolname = st.selectbox("Choose 'click' column", list(df.columns))
        with col1_2:
            purchasecolname = st.selectbox("Choose 'purchase' column", list(df.columns))

        st.markdown(""" Specify other columns that are not used in the training. These are the columns that are not related to user types.""")
        
        features_to_ignore = st.multiselect(
            r"Choose all the features that are not related to customer type; e.g. `id`, duplicates, etc.",
            [item for item in list(df.columns) if item not in [bidcolname, rankcolname, clickcolname, purchasecolname]]) 

        #features_to_pass_on = st.multiselect(
        #    "Choose other features that aren't related to user types; e.g. `click`, `purchased`, etc.", 
        #    [item for item in list(df.columns) if item not in features_to_ignore + [bidcolname, rankcolname]], 
        #    default=['click', 'policies sold'])

        st.write('Based on your choices, the following features are related to customer characteristics: ',
                 [item for item in list(df.columns) if item not in features_to_ignore + [bidcolname, rankcolname, clickcolname, purchasecolname]])
        
        proceed_checkbox = False
        with st.beta_expander("Correct? "):
            if bidcolname and rankcolname and clickcolname and purchasecolname and st.checkbox("Proceed"): #and features_to_ignore and
                proceed_checkbox = True
                #get the bid and rank features
                bid, rank = list(df[bidcolname]), list(df[rankcolname])
                features_to_pass_on = [clickcolname, purchasecolname]
                cgm = CustomerGroupsMaker(df,
                                          columns_to_ignore=features_to_ignore, 
                                          columns_to_pass_on=features_to_pass_on, 
                                          columns_bid_rank=[bidcolname, rankcolname])
                cgm.make_groups()

                group = st.selectbox(r"Choose a user group corresponding to : (" + ' ; '.join(cgm.used_column_names) + ')',
                                     list(cgm.customer_groups),15)

                col01, col02, col03 = st.beta_columns(3)
                with col01:
                    bid_from = st.selectbox("Choose lowest possible bid", list(range(3*max(bid))),5) #2, 9
                with col02:
                    bid_to = st.selectbox("Choose highest possible bid", list(range(bid_from+1, 6*max(bid))),14)
                with col03:
                    bids_num = st.selectbox("How many bid grid points", list(range(5, 101)),25)
            else:
                proceed_checkbox = False
                st.markdown("Correctly specify the relevant features first.")
            
            #=================================================================================================================================
            # get all the user inputs here, including user groups and ask user to confirm to proceed
            # try caching?
            # add if to button and indent...
            if st.button('Train the model and get results') and proceed_checkbox:
            #=================================================================================================================================
        

                st.markdown(""" ## Model Results:""")
                #                \n After specifying features we proceed with model.
                #                The model is run and the competitors are simulated in a way to provide the best match for the observed data.
                #                To inspect how good is the model fit, we can check the rank frequency observed in the data 
                #                against that produced by the model. Below we observe that for the all user groups combined.
                #                """)

                #if st.button('confirm selection, and make calculations'):
                #with st.beta_expander("See model fit for all groups combined : "):
#-----------------------------
                #rv_generator_dist = MixtureGaussian2().rvs # gengamma.rvs # skewnorm.rvs # 
#                cv = CrossValidator(SimulationEstimator, 
#                                    'mixture_gaussian',
#                                    bid, 
#                                    rank, 
#                                    num_competitors=None)

                #optimizer_func: optimize.minimize, optimize.shgo, optimize.dual_annealing, optimize.broyden1
                #optimize.differential_evolution, optimize.basinhopping
                #for optimize.minimize: method='trust-constr, SLSQP', TNC, Nelder-Mead, COBYLA
#                cv.optimized_search(optimize.minimize, method='COBYLA',max_iter=None)

#                simest = cv.best_estimator
#                simest.run_all(bid, rank, num_competitors=None)


        #        filename = 'SimulationEstimator.pkl'
        #        dill.dump(simest, open(filename, 'wb'))
        #        with open('Files/SimulationEstimator.pkl' ,'rb') as f:
        #            simest = dill.load(f)

#                fig1, ax = plt.subplots(1,1, figsize=(8, 3), facecolor='w', edgecolor='k')
                #ax = ax.ravel()
#                bins = np.linspace(.5, 5.5, 6)
#                ranks_count_data = sorted([(v[0],v[1]) for v in collections.Counter(rank).items()])
#                ranks_count_simu = sorted([(v[0],v[1]) for v in collections.Counter(simest.simulated_ranks).items()])
#                _ = ax.bar([i[0] for i in ranks_count_data], [i[1] for i in ranks_count_data], alpha=0.5,color='b', label='data')
#                _ = ax.bar([i[0] for i in ranks_count_simu], [i[1] for i in ranks_count_simu], alpha=0.5,color='r', label='model')
#                _ = ax.set_title('model simulation vs data', size=18)
#                _ = ax.set_ylabel('freq.', size=14)
#                _ = ax.set_xlabel('ranks', size=14)
#                ax.legend(loc='best')
                #_ = ax[1].hist(simest.simulated_bids, bins = 20)
#                st.write(fig1)

#-----------------------------
                st.markdown('\n')
                st.markdown('\n') 
                #st.markdown(""" We can also check the rank frequency observed in the data 
                #                vs the model for each customer group separately.
                #            """)    

                #if st.button('confirm selection, and make calculations 2222'):
                #with st.beta_expander("See model fit for each groups separately : "):
#-----------------------------
#                cgm = CustomerGroupsMaker(df,
#                                          columns_to_ignore=features_to_ignore, 
#                                          columns_to_pass_on=features_to_pass_on, 
#                                          columns_bid_rank=[bidcolname, rankcolname])
#                cgm.make_groups()
#                df_out_column_description = cgm.df_out_column_description

                #gbe = GroupbyEstimator(cgm, SimulationEstimator, 'mixture_gaussian', 'COBYLA')
                gbe = GroupbyEstimator(cgm, SimulationEstimator, 'mixture_gaussian', 'COBYLA', group)
                gbe.make_group_estimators()

                #filename = 'GroupByEstimator.pkl'
                #dill.dump(gbe, open(filename, 'wb'))
                #with open('Files/GroupByEstimator.pkl' ,'rb') as fg:
                #    gbe = dill.load(fg)

                st.write("Columns used for grouping : (" + ' ; '.join(cgm.used_column_names) + ')')

                fig, ax = plt.subplots(1,1, figsize=(8, 3), facecolor='w', edgecolor='k')
                #ax = ax.ravel()
                #bins = np.linspace(.5, 5.5, 6)
                #ranks_count_data = sorted([(v[0],v[1]) for v in collections.Counter(rank).items()])
                #ranks_count_simu = sorted([(v[0],v[1]) for v in collections.Counter(simest.simulated_ranks).items()])
                #_ = ax.bar([i[0] for i in ranks_count_data], [i[1] for i in ranks_count_data], alpha=0.5,color='b', label='data')
                #_ = ax.bar([i[0] for i in ranks_count_simu], [i[1] for i in ranks_count_simu], alpha=0.5,color='r', label='model')
                #_ = ax.set_ylabel('freq.', size=14)
                #_ = ax.set_xlabel('ranks', size=14)
                #ax.legend(loc='best')
                ##_ = ax[1].hist(simest.simulated_bids, bins = 20)
                #st.write(fig1)




                #fig, ax = plt.subplots(int(len(cgm.customer_groups)/4), 4, figsize=(14, 7), facecolor='w', edgecolor='k')
                #fig.subplots_adjust(hspace=.5)
                #fig.suptitle("Columns used for grouping : (" + ' ; '.join(cgm.used_column_names) + ')', size=13)
                #ax = ax.ravel()

                df_group_temp = cgm.df_out.copy()
                for g_item_idx in range(len(group)):
                    df_group_temp = df_group_temp.loc[
                        df_group_temp[cgm.used_column_names[g_item_idx]] == group[g_item_idx]]
                rank = df_group_temp[rankcolname]
                simest_g = gbe.group_estimators[group]
                ranks_count_data = sorted([(v[0],v[1]) for v in collections.Counter(rank).items()])
                ranks_count_simu = sorted([(v[0],v[1]) for v in collections.Counter(simest_g.simulated_ranks).items()])
                ax.bar([i[0] for i in ranks_count_data], [i[1] for i in ranks_count_data], alpha=0.5,color='b')
                ax.bar([i[0] for i in ranks_count_simu], [i[1] for i in ranks_count_simu], alpha=0.5,color='r')
                ax.set_title(group)

                st.write(fig)
#-----------------------------

                st.markdown(""" ## Analyze rank probabilities
                                   \n Rank probability analysis for a given customer group, and a bid range.""") 

    #            group = st.selectbox(r"Choose a user group corresponding to : (" + ' ; '.join(cgm.used_column_names) + ')',
    #                                 list(cgm.customer_groups),15)

    #            col01, col02, col03 = st.beta_columns(3)
    #            with col01:
    #                bid_from = st.selectbox("Choose lowest possible bid", list(range(3*max(bid))),5) #2, 9
    #            with col02:
    #                bid_to = st.selectbox("Choose highest possible bid", list(range(bid_from+1, 6*max(bid))),14)
    #            with col03:
    #                bids_num = st.selectbox("How many bid grid points", list(range(5, 101)),25)

                #bids_num = bid_to - bid_from # 20
                #bid_from = 5
                #bid_to = 20

                #if st.button('Confirm group selection and proceed'):
                #with st.beta_expander("See probability of obtaining different ranks for given bid range and user groups : "):
#-----------------------------                    
                #group = cgm.customer_groups[15]

                df_group_temp = cgm.df_out.copy()
                #group = cgm.customer_groups[15] #chosen in the select box
                for g_item_idx in range(len(group)):
                    df_group_temp = df_group_temp.loc[
                        df_group_temp[cgm.used_column_names[g_item_idx]] == group[g_item_idx]]
                rank = df_group_temp[rankcolname]

                bid_from_to = list(np.round(np.linspace(bid_from,bid_to, bids_num), 2))
                bids_rank_probs_df = pd.DataFrame(columns=['Bid', 'rank1_prob', 'rank2_prob', 'rank3_prob', 'rank4_prob', 'rank5_prob'])

                bids_rank_probs_row = {}
                for bid_amount in bid_from_to:
                    sim_ranks = gbe.predict_ranks(group, bid_amount)
                    ranks_count_simu = sorted([(v[0],v[1]) for v in collections.Counter(sim_ranks).items()])
                    ranks_count_sum = sum([v[1] for v in ranks_count_simu])
                    ranks_prob = {int(v[0]): v[1]/ranks_count_sum for v in ranks_count_simu}

                    bids_rank_probs_row[bids_rank_probs_df.columns[0]] = bid_amount
                    for col_idx in range(1, len(bids_rank_probs_df.columns)):
                        bids_rank_probs_row[bids_rank_probs_df.columns[col_idx]] = ranks_prob.get(col_idx, 0)
                    bids_rank_probs_df = bids_rank_probs_df.append(bids_rank_probs_row, ignore_index=True)

                #bids_rank_probs_df_plot = pd.DataFrame(bids_rank_probs_df.iloc[:,1:], index=bids_rank_probs_df.iloc[:,0])
                fig, ax = plt.subplots(figsize=(12,6))  
                bids_rank_probs_df.plot(ax = ax,
                                        x = 'Bid',
                                        kind = 'bar', 
                                        stacked = True,
                                        mark_right = True)
                _ = ax.set_title('Probabilities of getting different ranks for different bids', size=18)
                _ = ax.set_xlabel('Bid amount', size=16)
                _ = ax.set_ylabel('Ad rank probability', size=16)

                st.write(fig)

#-----------------------------
                #==========================================================

                st.markdown(""" ## Analyze purchase probabilities
                                   \n Purchase probability analysis for the customer group and bid range specified above.""") 

                df_group_temp = cgm.df_out.copy()

                #col001, col002 = st.beta_columns(2)
                #with col001:
                #    clickcolname = st.selectbox("Choose `click` column", features_to_pass_on, 0)
                #with col002:
                #    purchasecolname = st.selectbox("Choose `purchase` column", features_to_pass_on, 1)

                #with st.beta_expander("See probability of a given bid turning into purchase for given user groups : "):
#-----------------------------
                ranks_list = list(df_group_temp.groupby(rankcolname)[rankcolname].count().keys())
                ranks_list
                #df_out_ranks = {}
                prob_click_for_ranks = {}
                prob_purchase_clicked_for_ranks = {}
                for r in ranks_list:
                    df_out_ranks = df_group_temp.loc[df_group_temp[rankcolname] == r]
                    prob_click_for_ranks[r] = sum(df_out_ranks[clickcolname])/len(df_out_ranks[clickcolname])

                    df_out_ranks_clicked = df_out_ranks.loc[df_out_ranks[clickcolname] == 1]
                    prob_purchase_clicked_for_ranks[r] = sum(df_out_ranks_clicked[purchasecolname])/len(df_out_ranks_clicked[purchasecolname])

                prob_bid_turning_to_sale = {}
                for b in bid_from_to:
                    prob_bid_turning_to_sale[b] = 0
                    for r_idx in range(len(ranks_list)):
                        prob_rank_i = list(bids_rank_probs_df.loc[bids_rank_probs_df['Bid']==b][bids_rank_probs_df.columns[r_idx+1]])[0]
                        prob_click_for_rank_i = prob_click_for_ranks.get(ranks_list[r_idx], 0)
                        prob_purch_cond_on_click_rank_i = prob_purchase_clicked_for_ranks.get(ranks_list[r_idx], 0)

                        prob_bid_turning_to_sale[b] += prob_rank_i*prob_click_for_rank_i*prob_purch_cond_on_click_rank_i



                fig1, ax = plt.subplots(1,1, figsize=(12, 6), facecolor='w', edgecolor='k')
                _ = ax.scatter([k for k,v in prob_bid_turning_to_sale.items()], [v for k,v in prob_bid_turning_to_sale.items()])
                _ = ax.set_title('probability of different bids turning to purchase', size=18)
                _ = ax.set_ylabel('prob. sale', size=14)
                _ = ax.set_xlabel('bid', size=14)

                st.write(fig1)
#-----------------------------
                st.markdown(""" ## What's next?
                                   \n Given the probability of any given bid turning into a purchase for any user type,
                                   combined with profitability of that user type, it would be relatively easy to
                                   determine the bid amount that maximizes profits.
                                   \n This will be added in the next update...""") 

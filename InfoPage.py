import pandas as pd
import numpy as np
import streamlit as st

def app():

    st.title("A brief explanation of my method")
    st.sidebar.markdown("""[**See on my GitHub**](https://github.com/saeedshaker/capstone_TDI)""")
    
    st.markdown(""" My algorithm consists of two main steps:
                    \n - Finding the best estimator, and
                    \n - Getting the results using the best estimator.
    """)
    
    st.markdown(""" ## Finding the best estimator:
                    \n The main two features used to train the model are `bid` and `rank`. Something like the following:
    """)
    
    bid = [8]*5 + [10]*5
    rank = [3,4,3,5,4,1,5,1,3,2]
    bidrank_dict = {'bid': bid, 'rank': rank}
    bid_rank_df = pd.DataFrame(bidrank_dict)
    
    st.dataframe(bid_rank_df.head(8))
    
    st.markdown(""" For any given bid amount in the data set at hand, and given these two features, 
                    we simulate some competitors' bids from some distribution.
                    The goal is to set the parameters of that distribution in a way to obtain the ranks that closely matches
                    what we observe in the data. We can choose the number of competitors' that we want to simulate.
                    The number of competitors can be set by the prior information on the number of main players in this market
                    that are competing against our company. after simulation we can get something like the following:
    """)
    
    comps = np.random.randint(14, size=[4,10])+3
    comps_dict = {}
    for i in range(4):
        comps_dict['comp'+str(i+1)] = comps[i]
    comps_dict['others'] = ['.']*10
    comps_df = pd.DataFrame(comps_dict)
    
    st.dataframe(comps_df.head(8))
    
    st.markdown(""" Now, given the simulated competitors' bids as well as our company's observed bids,
                    we can find the implied ranks from these simulations.
    """)
    
    sim_rank = [3,4,3,2,4,1,5,4,3,2]
    sim_rank_dict = {'simulated rank': sim_rank}
    sim_rank_df = pd.DataFrame(sim_rank_dict)
    st.dataframe(sim_rank_df.head(8))
    
    st.markdown(""" As mentioned above, the goal is to get as close as possible to the observed ranks gor each observed bid amount.
                    After finding the right model paramters that matches closely the observed patterns in the data,
                    we can start to do analysis by making use of the simulated competitors and as if we know how they will behave.
    """)
    
    st.markdown(""" ## Getting the results using the best estimator:
                    \n Here I do analysis for any given customer type.
                    That is, I first split the data based on user/customer characteristics, 
                    and then train the model separately for each group as well as for all the groups combined.
                    The premise is that the competitors will treat different user types differently and
                    will have different bidding behavior for different user types.
    """)
    
    st.markdown(""" For any given user type and bid amount, I specify a range of possible bids to analyze.
                    I then use the simulated competitors' data from the previous step 
                    to obtain the probability of getting different ranks for any possible bid.
                    See the following:
    """)
    
    rank_probs = [[0,0,.1,.2,'..',.7], [0,.1,.2,.3,'..',.25], [.15,.2,.3,.3,'..',.05], [.25,.3,.3,.2,'..',0], [.6,.4,.1,0,'..',0]]
    rank_probs_dict = {}
    rank_probs_dict['bid'] = [3,5,7,10,'..',16]
    for i in range(5):
        rank_probs_dict['rank '+str(i+1)] = rank_probs[i]
    rank_probs_df = pd.DataFrame(rank_probs_dict)
    #rank_probs_df.set_index('bid', inplace=True)
    
    #st.dataframe(rank_probs_df.style.apply(lambda x: "background-color: red"))
    
    def color_survived(val):
        color = 'whitesmoke' if val else 'red'
        return f'background-color: {color}'

    #st.dataframe(df.style.apply(highlight_survived, axis=1))
    st.write("probability of getting different ranks given different possible bids")
    st.dataframe(rank_probs_df.style.applymap(color_survived, subset=['bid']))
    
    #st.write(rank_probs_df.head(6))

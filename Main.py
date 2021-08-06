import streamlit as st
import pandas as pd
#import altair as alt

#from PIL import Image

#def insert_markdown_file(filepath):
#    '''filepath might be something like markdown-01.md'''
#    with open(filepath, 'r') as f:
#        contents = f.read()
#    return contents

def app():
    #st.balloons()
    
    st.title("Online Ad Auctions: How Much to Bid?")
    st.sidebar.markdown("""[**See on my GitHub**](https://github.com/saeedshaker/capstone_TDI)""")
    
    #image = Image.open('Files/bid_keyboard.jpg')
    st.image('Files/bid_keyboard.jpg', use_column_width=True)
    
    st.markdown(""" ## Check this app out if:
                        \n * your company participates in online Ad auctions market 
                        \n * want to find best bid to optimize conversion or revenue
                        \n * have limited data on other market competitors""")
    
    st.image('Files/bid_online.jpg', use_column_width=True)
    
    st.markdown(""" ## In Online Ad Markets: \n
                    \n First, companies bid for impressions knowing some of the end user characteristics.
                    Then the platform decides which adds to show to customers based on multiple factors such as
                    bid amount, relevance of the add to customer type, some company related metrics, etc.
                    
                    \n Bid amount is perhaps the most defining factor over which a company has full control. 
                    Therefore, if we assume that other factors are constant 
                    (although some of them might be improveable which is a separate topic to discuss)
                    it is crucial to determine the right bid amount in order to achieve high conversion rate
                    while keeping the cost low.
                """)
    st.image('Files/outline_diagram.jpg', use_column_width=True)
    
    st.markdown(""" ## Determining the right bid is challenging:
                    \n Finding the right amount to bid is not easy because there are a lot of factors to take into account.
                    For example, you might want to know the following about the end users:
                    \n * how profitable a certain type of customer is, 
                    \n * how likely it is for that customer to click on your Ads,
                    \n * conditional on click, what is the conversion rate for that customer.
                    \n * and many more...
                    
                    \n These are all important questions. However, with some historical data available on different customer types
                    it is possible to make analysis and get some insights on the end users.
                """)
    
    st.image('Files/targetCustomer.jpg', use_column_width=True)
    
    st.markdown(""" ## The biggest challenge:
                    \n The most crucial piece of information is related to the behavior of other competitors in the market.
                    These competitors are mainly companies with similar products that compete for the same customers.
                    The collective actions of your competitors is the most important determinant of whether your Ad
                    will be shown to the target customers, and whether it will be shown in the rank to have the desired impact.
                """)
    
    st.markdown(""" ## This app:
                    \n **My project will answer the following questions:**
                    
                    \n Can we have any understanding on the competitors' collective actions? `Yes`
                    \n Is it possible to optimize our bid to get the desired outcome while anticipating what other competitors will do? `Yes`
                  
                """)
    
    st.markdown(""" ## How my method works?
                    \n Check out the `Technical Details` tab.
                   
                """)
    
if __name__ == '__main__':
    app()

# import streamlit as st
# import json

# # Load outputs from JSON files
# with open('cnn_output.json', 'r') as f:
#     cnn_output = json.load(f)

# with open('rforest_out.json', 'r') as f:
#     prediction_label = json.load(f)

# with open('stock_news_out.json', 'r') as f:
#     news_sentiment = json.load(f)

# # Calculate weighted final output
# final_output = (0.25 * cnn_output) + (0.25 * prediction_label) + (0.5 * news_sentiment)

# # Classify based on final output
# if final_output >= 0.75 or final_output > 0.5:  # Adjust threshold as needed
#     classification = 1
#     message = "Stock may grow in the next week."
# else:
#     classification = 0
#     message = "Stock may move down in the next week."

# # Display the results using Streamlit
# st.write("Final Output:", final_output)
# st.write("Classification:", classification)
# st.write(message)


import streamlit as st
import json

# Load outputs from JSON files
with open('cnn_output.json', 'r') as f:
    cnn_output = json.load(f)

with open('rforest_out.json', 'r') as f:
    prediction_label = json.load(f)

with open('stock_news_out.json', 'r') as f:
    news_sentiment = json.load(f)

# Ensure all inputs are binary (0 or 1)
cnn_vote = round(cnn_output)
rforest_vote = round(prediction_label)
news_vote = round(news_sentiment)

# Majority voting: at least 2 out of 3 must be 1 for final label = 1
votes = [cnn_vote, rforest_vote, news_vote]
vote_sum = sum(votes)

if vote_sum >= 2:
    classification = 1
    message = "Stock may rise next week"
else:
    classification = 0
    message = "Stock may move down in the next week."

# Display the results using Streamlit
st.write("Votes:", {"CNN": cnn_vote, "Random Forest": rforest_vote, "News Sentiment": news_vote})
st.write("Vote Sum:", vote_sum)
st.write("Classification:", classification)
st.write(message)

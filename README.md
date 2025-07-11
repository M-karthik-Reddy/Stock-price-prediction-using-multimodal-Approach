🧠 Project Overview : 
This project presents a multimodal approach to predicting stock market prices by integrating three diverse data modalities: numerical data, visual data (candlestick charts), and textual information (financial news). The goal is to leverage the unique strengths of each data type and corresponding machine learning models to improve the accuracy and interpretability of stock price movement predictions.

📈 Methodology
🔢 Numerical Data Analysis
We utilize historical stock price metrics—such as open, close, high, low, and volume—to generate tabular numerical data. A Random Forest Classifier is trained on this structured data to learn complex patterns and relationships that influence price movements. This model provides a strong baseline using traditional quantitative analysis.

🖼️ Visual Feature Extraction : 
In parallel, we convert five-day candlestick charts of stock prices into images and apply Convolutional Neural Networks (CNNs)(EfficietNet B0)to extract deep visual features. These features help capture patterns like trends, breakouts, or reversals that are difficult to detect from raw numbers alone. It outputs either 1(rise) or 0(Fall) for next day, after training all historic chart data.

📰 Textual Sentiment Analysis : 
The third modality involves analyzing recent news articles relevant to the selected stocks. Using webscraping the latest news of stock & Natural Language Processing (NLP) techniques, we extract sentiment and relevance from these articles. This component helps us assess how real-world events and public opinion may influence stock behavior in the short term.

⚙️ Model Fusion and Prediction :
The predictions from each modality—numerical (Random Forest), visual (CNN), and textual (sentiment analysis)—are combined in an ensemble framework. This ensemble outputs a final binary prediction indicating whether the stock price is expected to rise (1) or fall (0) in the near future. By integrating complementary information from all three sources, the model aims to offer a more holistic and accurate forecast.


<img width="1919" height="870" alt="image" src="https://github.com/user-attachments/assets/3a81b769-6348-421a-a1c8-6817aceb686d" />


<img width="1916" height="908" alt="image" src="https://github.com/user-attachments/assets/194429c1-0d0e-451f-82e0-a3036a54d417" />


<img width="1899" height="898" alt="image" src="https://github.com/user-attachments/assets/e0091b6d-2b98-4838-9eca-b6c26f12279c" />




# import pandas as pd
# import numpy as np
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import plotly.express as px
# import streamlit as st
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt


# # Load the Excel data
# stock_data = pd.read_excel('nifty_stocks.xlsx')

# # Extract company names and corresponding symbols
# company_names = stock_data['Company Name'].tolist()
# company_to_symbol = stock_data.set_index('Company Name')['Symbol'].to_dict()

# # Sidebar inputs
# selected_company = st.sidebar.selectbox("Select Company", [""] + company_names)
# exchange = st.sidebar.selectbox('Exchange', ["", "NSE", "BSE", "NYSE"])
# start_date = st.sidebar.date_input('Start Date')
# end_date = st.sidebar.date_input("End Date")

# # Retrieve the symbol for the selected company
# selected_symbol = company_to_symbol.get(selected_company, None)

# # Append the correct suffix based on selected exchange
# if selected_symbol:
#     ticker = selected_symbol
#     if exchange == 'NSE':
#         ticker += '.NS'
#     elif exchange == 'BSE':
#         ticker += ".BO"
#     else:
#         ticker += ''

#     # Download stock data
#     data = yf.download(ticker, start_date, end_date)

#     # Check if data is empty or the date range is incorrect
#     if data.empty:
#         st.write("No data available for the selected date range or stock symbol. Please select a different range or check the symbol.")
#     elif start_date >= end_date:
#         st.write("The start date must be before the end date. Please adjust the date range.")
#     else:
#         # Proceed if data is available
#         st.write(f"Company Name: {selected_company}")

#         # Save to CSV file
#         data.to_csv(f'{selected_company}_data.csv')
        
#         data.columns = data.columns.droplevel(1)
#         # st.write(data.columns)
#         # Plotting if data is not empty
#         if not data.empty:
#             fig = px.line(data, x=data.index, y='Close', title=f"{selected_company} Stock Price")
#             st.plotly_chart(fig)
#         else:
#             st.write("No data available for the selected date range. Please adjust the dates.")
            
#         data.reset_index(inplace=True)
#         data['Change'] = (data['Close'].shift(-1) > data['Close']).astype(int)
#         # Set the first value to 0 after computing the full column
#         data['Change'] = data['Change'].shift(1)
#         data=data.drop(index=0).reset_index(drop=True)
#         # data.at['0','Change']=0
#         data['SMA_5'] = data['Close'].rolling(window=5).mean()
#         data['Price_Volume'] = data['Close'] * data['Volume']

#         st.write(data.head())

#         # Check if required columns are present
#         required_columns = ['Open', 'High', 'Low', 'Close', 'Close', 'Volume','SMA_5','Price_Volume']
#         if not all(column in data.columns for column in required_columns):
#             st.write("Some required columns are missing in the data.")
#         else:
#             # Features and target
#             features = data[['Open', 'High', 'Low', 'Close',  'Volume','SMA_5']].values
#             target = data['Change'].values

#             # Normalize the features
#             scaler = StandardScaler()
#             features = scaler.fit_transform(features)

#             split_train_value = st.sidebar.selectbox("Enter train size:", [50, 60, 70, 80, 90])
#             split_test_value = 100 - split_train_value

#             # Split the dataset into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=split_train_value / 100, random_state=42)

#             # Sidebar options for Random Forest
#             n_estimators = st.sidebar.slider("Select number of trees (n_estimators):", 10, 200, step=10, value=100)
#             max_depth = st.sidebar.slider("Select max depth:", 1, 20, step=1, value=10)
#             min_samples_split = st.sidebar.slider("Min samples split:", 2, 10, step=1, value=2)
#             min_samples_leaf = st.sidebar.slider("Min samples leaf:", 1, 10, step=1, value=1)

#             if st.sidebar.button('Start Training'):
#                 # Create the Random Forest classifier with regularization
#                 mypipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
#                                              ('model', RandomForestClassifier(
#                                                  n_estimators=n_estimators,
#                                                  max_depth=max_depth,
#                                                  min_samples_split=min_samples_split,
#                                                  min_samples_leaf=min_samples_leaf,
#                                                  random_state=0))
#                                              ])
                
                

#                 # Cross-validation scores
#                 scores_train = -1 * cross_val_score(mypipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#                 scores_test = -1 * cross_val_score(mypipeline, X_test, y_test, cv=5, scoring='neg_mean_absolute_error')

#                 # Train the model
#                 mypipeline.fit(X_train, y_train)

#                 # Predict and evaluate the model on the test set
#                 y_pred_train = mypipeline.predict(X_train)
#                 y_pred_test = mypipeline.predict(X_test)

#                 # Calculate accuracy
#                 train_accuracy = accuracy_score(y_train, y_pred_train)
#                 test_accuracy = accuracy_score(y_test, y_pred_test)

#                 # Display results
#                 st.write(f"Train Accuracy: {train_accuracy * 100:.2f}%")
#                 st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")
                
#                 # --- F1 Score ---
#                 f1 = f1_score(y_test, y_pred_test)
#                 st.write(f"F1 Score: {f1:.2f}")
                
#                 # --- Confusion Matrix ---
#                 cm = confusion_matrix(y_test, y_pred_test)
#                 fig_cm, ax_cm = plt.subplots()
#                 disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#                 disp.plot(ax=ax_cm)
#                 st.pyplot(fig_cm)
                
#                 # --- ROC AUC Curve ---
#                 # Ensure binary classification (0/1)
#                 if len(np.unique(y_test)) == 2:
#                     y_proba = mypipeline.predict_proba(X_test)[:, 1]
#                     fpr, tpr, _ = roc_curve(y_test, y_proba)
#                     auc_score = roc_auc_score(y_test, y_proba)

#                     fig_auc, ax_auc = plt.subplots()
#                     ax_auc.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
#                     ax_auc.plot([0, 1], [0, 1], linestyle='--')
#                     ax_auc.set_xlabel('False Positive Rate')
#                     ax_auc.set_ylabel('True Positive Rate')
#                     ax_auc.set_title('ROC AUC Curve')
#                     ax_auc.legend()
#                     st.pyplot(fig_auc)


#                 # Prediction for the last day (next day)
#                 last_day_features = scaler.transform(data[['Open', 'High', 'Low', 'Close','Volume','SMA_5']].iloc[-1:].values)
#                 next_day_prediction = mypipeline.predict(last_day_features)

#                 # Display next day prediction
#                 prediction_label = 1 if next_day_prediction[0] == 1 else 0
#                 st.write(f"Next Day Prediction: {prediction_label}")

                

            
# import json
# # Save to a JSON file
# if __name__ == "__main__":
#     with open('rforest_out.json', 'w') as f:
#        json.dump(prediction_label, f)



import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load the Excel data
stock_data = pd.read_excel('nifty_stocks.xlsx')

# Extract company names and corresponding symbols
company_names = stock_data['Company Name'].tolist()
company_to_symbol = stock_data.set_index('Company Name')['Symbol'].to_dict()

# Sidebar inputs
selected_company = st.sidebar.selectbox("Select Company", [""] + company_names)
exchange = st.sidebar.selectbox('Exchange', ["", "NSE", "BSE", "NYSE"])
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input("End Date")

selected_symbol = company_to_symbol.get(selected_company, None)

# Append the correct suffix based on selected exchange
if selected_symbol:
    ticker = selected_symbol
    if exchange == 'NSE':
        ticker += '.NS'
    elif exchange == 'BSE':
        ticker += '.BO'

    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.warning("No data available for the selected date range or stock symbol.")
    elif start_date >= end_date:
        st.warning("Start date must be before the end date.")
    else:
        st.write(f"### Company: {selected_company}")
        data.to_csv(f'{selected_company}_data.csv')

        data.columns = data.columns.droplevel(1)
        # st.write(data.columns)
        # Plotting if data is not empty
        if not data.empty:
            fig = px.line(data, x=data.index, y='Close', title=f"{selected_company} Stock Price")
            st.plotly_chart(fig)
        else:
            st.write("No data available for the selected date range. Please adjust the dates.")
            
        data.reset_index(inplace=True)
        data['Change'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        # Set the first value to 0 after computing the full column
        data['Change'] = data['Change'].shift(1)
        data=data.drop(index=0).reset_index(drop=True)
        # data.at['0','Change']=0
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['Price_Volume'] = data['Close'] * data['Volume']
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        data['ROC'] = data['Close'].pct_change(periods=5)
        
        # RSI measures oversold or overbought conditions.
        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))


        st.write(data.head(20))

        # Check if required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Close', 'Volume','SMA_5','Price_Volume','Momentum','ROC','RSI']
        if all(col in data.columns for col in required_columns):
            features = data[required_columns].values
            target = data['Change'].values

            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            

            split_train_value = st.sidebar.selectbox("Train size %", [50, 60, 70, 80, 90], index=2)
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=(100 - split_train_value) / 100, random_state=42
            )

            # Model hyperparameters
            n_estimators = st.sidebar.slider("n_estimators", 10, 200, step=10, value=100)
            max_depth = st.sidebar.slider("max_depth", 1, 20, step=1, value=10)
            min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, step=1, value=2)
            min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, step=1, value=1)

            model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM"])

            if st.sidebar.button("Start Training"):
                param_grid = {}
                # Model pipelines
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(random_state=0)
                    param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
                }
                elif model_choice == "XGBoost":
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
                    param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.05, 0.1]
             }
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=0)
                    param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1]
        }
                elif model_choice == "LightGBM":
                    model = LGBMClassifier(random_state=0)
                    param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10],
            'classifier__learning_rate': [0.05, 0.1]
        }

                pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('classifier', model)
                ])
                
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                
                y_pred_train = best_model.predict(X_train)
                y_pred_test = best_model.predict(X_test)

                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)


                st.write(f"### Model: {model_choice}")
                st.write(f"Best Parameters: {grid_search.best_params_}")
                st.write(f"Train Accuracy: {train_acc * 100:.2f}%")
                st.write(f"Test Accuracy: {test_acc * 100:.2f}%")

                # Cross-validation
                train_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
                test_scores = cross_val_score(pipeline, X_test, y_test, cv=5, scoring='accuracy')

                pipeline.fit(X_train, y_train)
                from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                

                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)
                
                # F1 Score
                f1 = f1_score(y_test, y_pred_test)
                st.write(f"F1 Score: {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred_test)
                fig_cm, ax_cm = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down (0)", "Up (1)"])
                disp.plot(ax=ax_cm, cmap=plt.cm.Blues)
                st.pyplot(fig_cm)
                
                # ROC Curve
                if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                  y_prob = pipeline.predict_proba(X_test)[:, 1]
                  fpr, tpr, _ = roc_curve(y_test, y_prob)
                  roc_auc = auc(fpr, tpr)

                  fig_roc, ax_roc = plt.subplots()
                  ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                  ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                  ax_roc.set_xlabel("False Positive Rate")
                  ax_roc.set_ylabel("True Positive Rate")
                  ax_roc.set_title("ROC Curve")
                  ax_roc.legend(loc="lower right")
                  st.pyplot(fig_roc)
                else:
                  st.info("ROC Curve is not available for this model.")
                # st.write(f"### Model: {model_choice}")
                # st.write(f"Train Accuracy: {train_acc * 100:.2f}%")
                # st.write(f"Test Accuracy: {test_acc * 100:.2f}%")

                # Predict next day
                last_day_features = scaler.transform(data[required_columns].iloc[-1:].values)
                next_day_prediction = pipeline.predict(last_day_features)
                prediction_label = int(next_day_prediction[0])

                st.write(f"### Next Day Prediction (0 = Down, 1 = Up): `{prediction_label}`")

                # Save prediction to JSON
                with open('rforest_out.json', 'w') as f:
                    json.dump({"prediction": prediction_label}, f)
        else:
            st.error("Some required columns are missing in the dataset.")
else:
    st.warning("Please select a valid company and exchange.")

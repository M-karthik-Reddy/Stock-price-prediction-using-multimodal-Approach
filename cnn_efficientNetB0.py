import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import plotly.express as px
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
import datetime
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from collections import Counter
from tensorflow.keras.utils import load_img, img_to_array

import json

# Set Streamlit page configuration
st.set_page_config(page_title='Stock Prediction Dashboard', layout='wide')

# Load the Excel data
stock_data = pd.read_excel('nifty_stocks.xlsx')

# Extract company names and corresponding symbols
company_names = stock_data['Company Name'].tolist()
company_to_symbol = stock_data.set_index('Company Name')['Symbol'].to_dict()

st.title('📈 Stock Prediction Dashboard')

# Sidebar inputs
selected_company = st.sidebar.selectbox("Select Company", [""] + company_names)
exchange = st.sidebar.selectbox('Exchange', ["", "NSE", "BSE", "NYSE"])
start_date = st.sidebar.date_input('Start Date', value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

# Initialize session state for CNN model activation
if 'cnn_model_active' not in st.session_state:
    st.session_state.cnn_model_active = False

# Proceed if a company is selected
if selected_company:
    selected_symbol = company_to_symbol.get(selected_company, None)

    # Append the correct suffix based on selected exchange
    if selected_symbol:
        ticker = selected_symbol
        if exchange == 'NSE':
            ticker += '.NS'
        elif exchange == 'BSE':
            ticker += ".BO"
        elif exchange == 'NYSE':
            ticker += ''
        else:
            ticker += ''

        # Display company name
        st.write(f"**Company Name:** {selected_company}")

        # Download stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Check for NaN or invalid data
        if data.isnull().values.any():
            data = data.fillna(method='ffill')
            data = data.dropna()
            st.write("⚠️ Missing data was found and handled.")

        # Save to CSV file
        csv_filename = f'{selected_company}_data.csv'
        data.to_csv(csv_filename)

        data.columns = data.columns.droplevel(1)
        # st.write(data.columns)

        # Plotting if data is not empty
        if not data.empty:
            fig = px.line(data, x=data.index, y='Close', title=f"{selected_company} Stock Price")
            st.plotly_chart(fig)
        else:
            st.write("No data available for the selected date range. Please adjust the dates.")

        # Read dataset
        dataset = pd.read_csv(csv_filename)
        # Drop the first row of actual data (with 'Ticker' and 'RELIANCE.NS')
        dataset = dataset.drop(index=0).reset_index(drop=True)
        # Drop the first row (contains 'Date', 'None', etc.)
        dataset = dataset.drop(index=0).reset_index(drop=True)
        dataset = dataset.rename(columns={'Price':'Date'})


        st.write("Columns in dataset:", dataset.columns.tolist())

        # Ensure 'Date' column is of datetime type
        dataset['Date'] = pd.to_datetime(dataset['Date'])

        # Convert relevant columns to numeric
        price_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in price_columns:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

        # Show cleaned result
        st.write("### Sample of Cleaned Data")
        st.write(dataset.head())
        

        # Preprocess dataset
        ohlc = dataset.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
        ohlc['Year'] = ohlc['Date'].dt.isocalendar().year
        ohlc['Week_Number'] = ohlc['Date'].dt.isocalendar().week

        # Ensure stock data contains valid numeric values
        ohlc = ohlc[(ohlc['Open'] > 0) & (ohlc['High'] > 0) & (ohlc['Low'] > 0) & (ohlc['Close'] > 0)]
        weeks = list(ohlc.groupby(['Year', 'Week_Number']))

        st.write(f"**Number of Weeks:** {len(weeks)}")

        # Sidebar options for splitting data
        split_train_value = st.sidebar.selectbox('Select Train Split %', [None, 40, 50, 60, 70, 80, 90])

        if split_train_value is not None:
            split_test_value = 100 - split_train_value
            st.write(f'**Train Ratio:** {split_train_value}%')
            st.write(f"**Test Ratio:** {split_test_value}%")

            split_idx_train = int(len(weeks) * (split_train_value / 100))
            train_weeks = weeks[:split_idx_train]
            test_weeks = weeks[split_idx_train:]

            # Create training and testing directories
            train_dir = 'stock_images/training'
            test_dir = 'stock_images/testing'

            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Function to save images and labels
            def save_weekly_images(weeks, directory):
                labels = []
                
                total_weeks = len(weeks) - 1  # We use len-1 because we always compare current with next week
                # split_index = int(total_weeks * split_idx_train)
                
                for i in range(total_weeks):
                   week_data = weeks[i][1]
                   next_week_data = weeks[i + 1][1]

                  # Set index for mplfinance
                   week_data = week_data.set_index('Date')
        
                 # Generate and save image
                   fig, ax = mpf.plot(week_data, type='candle', style='charles', ylabel='Price', returnfig=True)
                   img_name = f"Week_{i + 1}.png"
                   img_path = os.path.join(directory, img_name)
                   fig.savefig(img_path)
                   plt.close(fig)

                # Compute label
                   current_week_close = week_data['Close'].iloc[-1]
                   next_week_close = next_week_data['Close'].iloc[-1]
                   label = 1 if next_week_close > current_week_close else 0

                   labels.append((img_name, label))

                return labels

                
            # Save images and labels
            st.write("### Saving Training Images...")
            train_labels = save_weekly_images(train_weeks, train_dir)

            st.write("### Saving Testing Images...")
            test_labels = save_weekly_images(test_weeks, test_dir)
            
            
            # Combine labels if needed
            all_labels = {
              "train": train_labels,
               "test": test_labels
            }

            st.success("✅ Images saved directly to training and testing folders, and labels collected.")

            # Button to activate CNN model
            if st.button('Train CNN Model'):
                st.session_state.cnn_model_active = True

            # Displays CNN options when the button is clicked
            if st.session_state.cnn_model_active:
                st.subheader("🧠 CNN Model Training")

                # CNN Model settings
                train_batch_size = st.selectbox('Train Batch Size', [32, 64, 128], key='train_batch_size')
                test_batch_size = st.selectbox('Test Batch Size', [32, 64, 128], key='test_batch_size')
                # epochs = st.number_input('Epochs', min_value=1, max_value=100, value=10, step=1, key='epochs')
                
                # Convert train and test label tuples to DataFrame
                train_df = pd.DataFrame(train_labels, columns=["filename", "label"])
                test_df = pd.DataFrame(test_labels, columns=["filename", "label"])
                train_df["filename"] = train_df["filename"].apply(lambda x: os.path.join(train_dir, x))
                test_df["filename"] = test_df["filename"].apply(lambda x: os.path.join(test_dir, x))
                
                # Convert labels to string for binary class_mode
                train_df["label"] = train_df["label"].astype(str)
                test_df["label"] = test_df["label"].astype(str)
                
                st.write("Unique labels in train:", train_df['label'].unique())
                st.write("Label counts in test_df:")
                st.write(test_df['label'].value_counts())


                # Data generators for CNN
                train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
                test_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True )

                train_generator = train_datagen.flow_from_dataframe(
                   train_df,
                   x_col="filename",
                   y_col="label",
                   target_size=(224, 224),
                   batch_size=train_batch_size,
                   class_mode='binary',
                #    shuffle=True
                )

                test_generator = test_datagen.flow_from_dataframe(
                   test_df,
                   x_col="filename",
                   y_col="label",
                   target_size=(224, 224),
                   batch_size=test_batch_size,
                   class_mode='binary',
                   shuffle=False
                )
                
                from collections import Counter
                st.write(f"Class Distribution in Training Set: {Counter(train_generator.classes)}")
                st.write(f"Class Distribution in Testing Set: {Counter(test_generator.classes)}")

                # Calculate class weights
                labels = train_generator.classes
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(labels),
                    y=labels
                )
                class_weights = dict(enumerate(class_weights))

                # Load base model
                base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
                base_model.trainable = True

                # Add custom head
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(0.2)(x)
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.2)(x)
                output = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=base_model.input, outputs=output)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                # early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)
                # ,callbacks=[early_stop, reduce_lr] 

                # Train model
                history = model.fit(train_generator, epochs=20, validation_data=test_generator,class_weight=class_weights)
                # Evaluate model on test data
                test_loss, test_acc = model.evaluate(test_generator)
                
                
                st.write(f'Test Accuracy: {test_acc:.4f}')
                st.write(f'Training Accuracy: {history.history["accuracy"][-1]:.4f}')
                
                # Predict on last image (latest week chart)
                st.write("### 📈 Predicting Current Week...")

                last_img_path = test_df["filename"].iloc[-1]  # latest image path
                img = load_img(last_img_path, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]
                pred_label = 1 if prediction > 0.5 else 0

                st.write(f'📊 Predicted Label for Current Week: **{pred_label}** (Probability: {prediction:.4f})')
                
                 # Get true labels and predictions
                y_true = test_generator.classes
                y_pred_prob = model.predict(test_generator).ravel()
                y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
                
                st.write("y_true sample:", y_true[:10])
                st.write("y_preD_prob:",y_pred_prob[:10])
                st.write("y_pred sample:", y_pred[:10])


                 # Compute metrics
                from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred_prob)
                cm = confusion_matrix(y_true, y_pred)
                
                # Plot AUC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
                fig_auc, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='lower right')
                
                st.write(f'🔁 F1 Score: {f1:.2f}')
                st.write(f'🎯 AUC Score: {auc:.2f}')
                st.write('📉 Confusion Matrix:')
                st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
                st.pyplot(fig_auc) 
                
 

import json

if __name__ == "__main__":
    with open('cnn_output.json', 'w') as f:
        json.dump(cnn_output, f)
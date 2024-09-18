from xgboost import XGBRegressor

def model_testing_xgboost(df, test_dates, features, n_steps=10, n_estimators = 100, learning_rate = 0.1, max_depth = 5,subsample = 0.9,gamma = 0.3):  

    predicted_df = df.copy()
    predicted_df['predictions'] = None

    for test_date in test_dates:
        trainsize = len(predicted_df[predicted_df['DateTime'] < test_date])
        non_numeric_columns = df.select_dtypes(include=['object']).columns
        df = df.drop(columns=non_numeric_columns)

        # As the final test is on predicting one step forward, I thought it best to have many iterations of guessing a single step,
        # then increasing the training data size and predicting the next one. Typically forecasting would be a recursive process
        
        for i in range(n_steps):
            df_train = df.iloc[:trainsize + i]
            
            X_train = df_train[features]
            y_train = df_train['demand']
            
            xgb_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,subsample = subsample,gamma = gamma)
            xgb_model.fit(X_train, y_train)
            X_test = df.iloc[[trainsize + i]][features]
            
            forecast = xgb_model.predict(X_test)
            predicted_df.iloc[trainsize + i, predicted_df.columns.get_loc('predictions')] = forecast[0]

    return predicted_df

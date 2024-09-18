from statsmodels.tsa.arima.model import ARIMA

def model_testing(df,test_dates,p,d,q):   

    predicted_df = df.copy()
    predicted_df['predictions'] = None
    
    for test_date in test_dates:
    
        trainsize = len(df[df.DateTime < test_date])
        # Actual values for the next 10 time steps
        # Predict the next 10 steps
        for i in range(10):
            df_train = df.iloc[:trainsize + i]
            model = ARIMA(df_train['demand'], order=(p,d,q))
            arima_model_fit = model.fit()
            
            forecast = arima_model_fit.predict(start=trainsize + i, end = trainsize + i).values
            predicted_df.iloc[trainsize + i, predicted_df.columns.get_loc('predictions')] = forecast[0]

    return predicted_df

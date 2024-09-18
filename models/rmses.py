import numpy as np
from sklearn.metrics import mean_squared_error

def rmse_evaluator(df,test_dates,mini = False):

    all_actual_values = []
    all_predicted_values = []
    all_rmse_values = []

    for test_date in test_dates:
        trainsize = len(df[df.DateTime < test_date])
        actual_values = df.iloc[trainsize:trainsize + 10].demand
        predicted_values = df.iloc[trainsize:trainsize + 10].predictions

        rmse = round(np.sqrt(mean_squared_error(actual_values, predicted_values)), 6)
        all_rmse_values.append(rmse)

        if not mini:
            print(f'\nPredictions for date {test_date}:')
            print(f"{'Actual':<15}{'Predicted':<15}")
            for actual, predicted in zip(actual_values, predicted_values):
                print(f'{actual:<15}       {predicted:<15}')
        print(f'rmse for this set:{rmse}')
        all_actual_values.extend(actual_values)
        all_predicted_values.extend(predicted_values)
        
    
    total_rmse = round(np.sqrt(mean_squared_error(all_actual_values, all_predicted_values)), 6)
    print('total_rmse',total_rmse)
        
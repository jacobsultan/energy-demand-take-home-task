import matplotlib.pyplot as plt 

def plot_predictions(df, test_dates):
    for test_date in test_dates:
        starting_time = df[df.DateTime == test_date].index.values.item()
        df_for_time = df.iloc[starting_time - 30: starting_time + 20]

        plt.figure(figsize=(10, 6)) 

        plt.plot(df_for_time['DateTime'], df_for_time['demand'], label='Actual (Full Day)', color='blue')
        plt.plot(df_for_time['DateTime'], df_for_time['predictions'], label='Predicted data', color='red', marker='o')

        plt.title(f'Full Day Demand with Predictions for {test_date}')
        plt.xlabel('Time')
        plt.ylabel('Demand')

        x_labels = df_for_time['DateTime'][::4] 
        plt.xticks(x_labels, rotation=45) 

        plt.legend()
        plt.tight_layout()
        plt.show()
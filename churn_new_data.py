import pandas as pd 
from pycaret.classification import predict_model, load_model


def load_data (filepath):
    w5 = pd.read_csv(filepath)
    return w5

def make_churn_predictions(w5):
    model = load_model('LDA')
    predictions = predict_model(model, data=w5)

    print(predictions.columns)

    if 'prediction_label' in predictions.columns:
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True)
        
        return predictions['Churn_prediction']
    else:
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")
    

if __name__ == "__main__":
    w5 = load_data('/Users/vinodreddy/Downloads/prepared_churn_data.csv')
    predictions = make_churn_predictions(w5)
    print('predictions:')
    print(predictions)
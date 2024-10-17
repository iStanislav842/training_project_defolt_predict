import dill
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def main():
    with open('data/prediction_model.pkl', 'rb') as file:
        model = dill.load(file)

    path = 'data/'
    df_test = pd.read_csv(f'{path}/df_test_0.csv')
    #df_test = pd.read_csv(f'{path}/df_test_1.csv')
    print(f'df_test.shape={df_test.shape}')
    y = model['model'].predict(df_test)
    print(f'предсказание для df_test:{y}')
    print(f'сведения о модели:{model["metadata"]}')
             

if __name__ == '__main__':
    main()
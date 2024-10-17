import dill
import os
import tqdm
import datetime
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


def main():
    #чтение и преобразование партиции chunk_path.pq к DataFrame read_parquet_dataset_from_local + prepare_transactions_dataset
    def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0, num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
        res = []
        dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) if filename.startswith('train')])
        print(dataset_paths)
        start_from = max(0, start_from)
        chunks = dataset_paths[start_from: start_from + num_parts_to_read]
        if verbose:
            print('Reading chunks:\n')
            for chunk in chunks:
                print(chunk)
        for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
            print('chunk_path', chunk_path)
            chunk = pd.read_parquet(chunk_path,columns=columns)
            res.append(chunk)
        return pd.concat(res).reset_index(drop=True)

    def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1, num_parts_total: int=50, save_to_path=None, verbose: bool=False):
        preprocessed_frames = []
        for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once), desc="Transforming transactions data"):
            transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, verbose=verbose)
            features = [x for x in transactions_frame.columns if x not in ['id']]
            transactions_frame[features] = transactions_frame[features].astype('int8')
            if save_to_path:
                block_as_str = str(step)
                if len(block_as_str) == 1:
                    block_as_str = '00' + block_as_str
                else:
                    block_as_str = '0' + block_as_str
                transactions_frame.to_parquet(os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet'))
            preprocessed_frames.append(transactions_frame)
        return pd.concat(preprocessed_frames)
    
    #препроцессинг данных: ohe + groupby(by=['id']).sum()
    def ohe_features_transform_and_groupby_by_id_sum(data):
        if data.shape[0] < 200000:
            list_ohe_futures = list(data.columns)
            list_ohe_futures.remove('id')
            ohe = OneHotEncoder(sparse_output=False, dtype='int8')
            ohe.fit(data[list_ohe_futures])
            data_ohe = ohe.transform(data[list_ohe_futures])
            ohe_categorical_var_list = ohe.get_feature_names_out()
            data[ohe_categorical_var_list] = data_ohe
            data.drop(list_ohe_futures, axis= 1 , inplace= True)
            data = data.groupby(by=['id'], as_index=False).sum()   
        else:
            start_id = 0
            stop_id = 3000000 #len(data.id.unique())
            slice = 300000
            df_all_ohe = pd.DataFrame()
            for id in np.arange(start_id, stop_id, slice):
                print(f'----- в {datetime.datetime.now().time()} start_id=[{id}, stop_id={id + slice})')
                df_cut = data[(data.id >= id) & (data.id < (id + slice))]
                list_ohe_futures = list(data.columns)
                list_ohe_futures.remove('id')
                ohe = OneHotEncoder(sparse_output=False, dtype='int8')
                ohe.fit(df_cut[list_ohe_futures])
                data_ohe = ohe.transform(df_cut[list_ohe_futures])
                ohe_categorical_var_list = ohe.get_feature_names_out()
                df_cut[ohe_categorical_var_list] = data_ohe
                df_cut.drop(list_ohe_futures, axis= 1 , inplace= True)
                df_all_ohe = pd.concat([df_all_ohe, df_cut], axis = 0)
                df_all_ohe = df_all_ohe.fillna(0)
                features = [x for x in df_all_ohe.columns if x not in ['id']]
                df_all_ohe[features] = df_all_ohe[features].astype('int8')
            data = df_all_ohe.groupby(by=['id'], as_index=False).sum()   
        return data
    
    #препроцессинг данных: добавление во фрейм нулевых столбцов, как в главном фрейме
    def add_columns(data):
        df_all_columns = pd.read_csv(f'{path}/df_1_row_all_columns.csv')
        list_add_columns = list(set(df_all_columns.columns) - set(data.columns))
        data[list_add_columns] = 0
        return data
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
        ])
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int8', 'int64', 'float64']))
        ])
    models = [LogisticRegression(class_weight='balanced', C=10),
              LGBMClassifier(class_weight='balanced', early_stopping_rounds=500, verbose=-1, n_jobs=8, objective='binary', metric='auc',
                             learning_rate=0.05, max_depth=20, n_estimators=1000, num_leaves=100),
              RandomForestClassifier(class_weight='balanced', n_estimators=90, max_depth=6, min_samples_leaf=7, min_samples_split=7)
            ]
    
    path = 'data/'
    targets = pd.read_csv(f'{path}train_target.csv')
    df = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=12, num_parts_total=12, save_to_path=path)
     
    try:
        Xval = pd.read_csv(f'{path}Xval.csv')
        yval = targets[targets['id'].isin(train_test_split(list(df.id.unique()), train_size=0.9, random_state=842)[1])]['flag']
    except:
        df = ohe_features_transform_and_groupby_by_id_sum(df)
        df[:1].to_csv(f'{path}/df_1_row_all_columns.csv', index=False)
        Xval = df[df['id'].isin(train_test_split(list(df.id.unique()), train_size=0.9, random_state=842)[1])]
        Xval.to_csv(f'{path}/Xval.csv', index=False)
        yval = targets[targets['id'].isin(train_test_split(list(df.id.unique()), train_size=0.9, random_state=842)[1])]['flag']
        df = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=12, num_parts_total=12, save_to_path=path)

    best_score = .0
    best_model = None   
    for model in models:
        pipe = Pipeline(steps=[
                    ('concat_slices_and_prepare', FunctionTransformer(ohe_features_transform_and_groupby_by_id_sum)),
                    ('add_columns', FunctionTransformer(add_columns)), 
                    ('preprocessor', preprocessor),
                    ('select_in_list_GridSearchCV_best_models', model) 
                ])       
            
        if (str(model).find('LogisticRegression') == 0) or (str(model).find('RandomForestClassifier') == 0):
            X = df
            y = targets[targets['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])]['flag']
            pipe.fit(X[X['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])], y)
            X = df
            y = targets[targets['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])]['flag']
            score = roc_auc_score(y, pipe.predict_proba(X[X['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])])[:, 1])
        else:
            X = df
            y = targets[targets['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])]['flag']
            pipe.fit(X[X['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])], y, 
                        select_in_list_GridSearchCV_best_models__eval_set=[(Xval, yval)])     
            score = roc_auc_score(y, pipe.predict_proba(X[X['id'].isin(train_test_split(list(X.id.unique()), train_size=0.9, random_state=842)[0])])[:, 1])

        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_model = model
            best_model_str = str(best_model)
    print(f'best_model={best_model}, best_roc_auc_score={best_score}')
   
    with open(f'{path}/prediction_model.pkl', 'wb') as file:
        dill.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'Final project of the Machine Learning Junior course',
            'author': 'iStanislav',
            'version': 1,
            'date': datetime.datetime.now(),
            'model': best_model_str,
            'ROC-AUC=': best_score
        }
    }, file, recurse=True)  
    print(f'в {datetime.datetime.now().time()} создание модели завершено')

if __name__ == '__main__':
    main()
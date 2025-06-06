import pandas as pd
import numpy as np
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def encode_status(df):
    if 'Status' in df.columns:
        df['Status'] = df['Status'].map({'Developing': 0, 'Developed': 1})
    return df

def select_features(df):
    selected = [
        'Status',
        'Adult Mortality',
        'infant deaths',
        'Hepatitis B',
        'Polio',
        'Diphtheria',
        'HIV/AIDS',
        'thinness  1-19 years',
        'thinness 5-9 years',
        'Schooling',
        'Life expectancy'
    ]
    df = df[selected]
    return df

def scale_features(df, target_col):
    scaler = StandardScaler()
    features = df.drop(columns=[target_col])
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    features_scaled[target_col] = df[target_col].values
    return features_scaled

def split_and_save(df, config):
    target_col = 'Life expectancy'
    train_df, test_df = train_test_split(df, test_size=config['data_split']['test_size'],
                                         random_state=config['data_split']['random_state'])
    train_df.to_csv(config['data_split']['trainset_path'], index=False)
    test_df.to_csv(config['data_split']['testset_path'], index=False)

def main(config_path):
    config = load_config(config_path)
    df = pd.read_csv(config['data_load']['raw_dataset'])

    df = encode_status(df)
    df = select_features(df)
    df = df.dropna()
    df = scale_features(df, 'Life expectancy')
    

    split_and_save(df, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)

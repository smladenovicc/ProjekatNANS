"""
Modul za učitavanje i preprocessing podataka o laptopima.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re


def load_data(filepath='../data/laptop_price.csv'):
    df = pd.read_csv(filepath, encoding='latin-1')
    print(f"Ucitano {len(df)} redova i {len(df.columns)} kolona")
    return df


def clean_data(df):
    
    df_clean = df.copy()
    cols_to_drop = ['laptop_ID', 'Product']
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"Uklonjene kolone: {cols_to_drop}")
    
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print(f"Nedostajuće vrednosti:\n{missing[missing > 0]}")
    else:
        print("Nema nedostajućih vrednosti")
    
    return df_clean


def extract_numeric_features(df):
    
    df_numeric = df.copy()
    df_numeric['Ram'] = df_numeric['Ram'].str.extract(r'(\d+)').astype(int)
    df_numeric['Weight'] = df_numeric['Weight'].str.replace('kg', '').str.strip().astype(float)
    df_numeric['Inches'] = pd.to_numeric(df_numeric['Inches'], errors='coerce')
    
    print(f"Ekstraktovane numericke vrednosti: Ram_GB, Weight_kg")
    
    return df_numeric


def process_screen_resolution(df):
    df_screen = df.copy()
    
    
    resolution = df_screen['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    df_screen['Screen_Width'] = resolution[0].astype(float)
    df_screen['Screen_Height'] = resolution[1].astype(float)
    df_screen['Total_Pixels'] = df_screen['Screen_Width'] * df_screen['Screen_Height']
    
    
    df_screen['IPS_Panel'] = df_screen['ScreenResolution'].str.contains('IPS', case=False, na=False).astype(int)
    df_screen['Touchscreen'] = df_screen['ScreenResolution'].str.contains('Touch', case=False, na=False).astype(int)
    df_screen['Retina'] = df_screen['ScreenResolution'].str.contains('Retina', case=False, na=False).astype(int)
    
    print(f"Obradjena rezolucija ekrana: Screen_Width, Screen_Height, Total_Pixels, IPS_Panel, Touchscreen, Retina")
    
    return df_screen


def process_memory(df):
    
    df_memory = df.copy()
    
    df_memory['Has_SSD'] = df_memory['Memory'].str.contains('SSD', case=False, na=False).astype(int)
    df_memory['Has_HDD'] = df_memory['Memory'].str.contains('HDD', case=False, na=False).astype(int)
    df_memory['Has_Flash'] = df_memory['Memory'].str.contains('Flash', case=False, na=False).astype(int)
    df_memory['Has_Hybrid'] = df_memory['Memory'].str.contains('Hybrid', case=False, na=False).astype(int)
    df_memory['Has_MultiDrive'] = df_memory['Memory'].str.contains(r'\+', na=False).astype(int)
    
  
    
    def extract_total_capacity(mem_str):
        total = 0
        
        gb_matches = re.findall(r'(\d+\.?\d*)GB', mem_str)
        total += sum(float(x) for x in gb_matches)
        
        tb_matches = re.findall(r'(\d+\.?\d*)TB', mem_str)
        total += sum(float(x) * 1024 for x in tb_matches)
        
        return total if total > 0 else np.nan
    
    df_memory['Total_Storage_GB'] = df_memory['Memory'].apply(extract_total_capacity)
    
    print(f"Obradjena memorija: Has_SSD, Has_HDD, Has_Flash, Has_Hybrid, Has_MultiDrive, Total_Storage_GB")
    
    return df_memory


def process_cpu(df):
    
    df_cpu = df.copy()
    
    
    def extract_cpu_brand(cpu_str):
        cpu_str = str(cpu_str)
        if 'Intel' in cpu_str:
            return 'Intel'
        elif 'AMD' in cpu_str:
            return 'AMD'
        else:
            return 'Other'
    
    df_cpu['CPU_Brand'] = df_cpu['Cpu'].apply(extract_cpu_brand)
    
    
    def extract_cpu_type(cpu_str):
        cpu_str = str(cpu_str)
        
        if 'Intel' in cpu_str:
            if 'i9' in cpu_str:
                return 'Intel i9'
            elif 'i7' in cpu_str:
                return 'Intel i7'
            elif 'i5' in cpu_str:
                return 'Intel i5'
            elif 'i3' in cpu_str:
                return 'Intel i3'
            elif 'Xeon' in cpu_str:
                return 'Intel Xeon'
            elif 'Atom' in cpu_str:
                return 'Intel Atom'
            elif 'Celeron' in cpu_str:
                return 'Intel Celeron'
            elif 'Pentium' in cpu_str:
                return 'Intel Pentium'
            elif 'Core M' in cpu_str:
                return 'Intel Core M'
            else:
                return 'Intel Other'
        
        elif 'AMD' in cpu_str:
            if 'Ryzen' in cpu_str:
                return 'AMD Ryzen'
            elif 'A12' in cpu_str:
                return 'AMD A12'
            elif 'A10' in cpu_str:
                return 'AMD A10'
            elif 'A9' in cpu_str:
                return 'AMD A9'
            elif 'A6' in cpu_str:
                return 'AMD A6'
            else:
                return 'AMD Other'
        
        else:
            return 'Other'
    
    df_cpu['CPU_Type'] = df_cpu['Cpu'].apply(extract_cpu_type)
    
   
    cpu_speed = df_cpu['Cpu'].str.extract(r'(\d+\.?\d*)\s*GHz')
    df_cpu['CPU_Speed_GHz'] = cpu_speed[0].astype(float)
    
    print("Obradjeen CPU: CPU_Brand, CPU_Type, CPU_Speed_GHz")
    
    return df_cpu


def process_gpu(df):
    
    df_gpu = df.copy()
    
    def extract_gpu_brand(gpu_str):
        if 'Intel' in gpu_str:
            return 'Intel'
        elif 'Nvidia' in gpu_str or 'GeForce' in gpu_str:
            return 'Nvidia'
        elif 'AMD' in gpu_str or 'Radeon' in gpu_str:
            return 'AMD'
        else:
            return 'Other'
    
    df_gpu['GPU_Brand'] = df_gpu['Gpu'].apply(extract_gpu_brand)
    
    print(f"Obradjen GPU: GPU_Brand")
    
    return df_gpu


def encode_categorical_features(df, categorical_columns):
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print(f"Primenjujem one-hot encoding na: {categorical_columns}")
    print(f"Ukupno kolona nakon encodinga: {len(df_encoded.columns)}")
    
    return df_encoded

def split_and_scale_data(df, target_col='Price_euros', test_size=0.2, val_size=0.2, random_state=42):
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # prvo delim na train i temp (temp = val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )
    
    # temp delim na validation i test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    print("\n Podela podataka:")
    print(f"  Training:   {len(X_train)} instanci ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} instanci ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(X_test)} instanci ({len(X_test)/len(df)*100:.1f}%)")
    
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print(f"\n Standardizovano {len(numeric_features)} numerickih features")
    
    feature_names = X_train.columns.tolist()
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names

def full_preprocessing_pipeline(filepath='../data/laptop_price.csv'):
    
    df = load_data(filepath)

    df = clean_data(df)

    df = extract_numeric_features(df)
    df = process_screen_resolution(df)
    df = process_memory(df)
    df = process_cpu(df)
    df = process_gpu(df)

    # provereavam nan vredosti nakon feature engeneringa
    print("\nProvera NaN vrednosti nakon feature engineeringa:")
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()

    if total_nan > 0:
        print("Postoje NaN vrednosti:")
        print(nan_counts[nan_counts > 0])
        print(f"Ukupno NaN vrednosti: {total_nan}")
    else:
        print("Nema NaN vrednosti.")


    df_processed = df.copy()

    categorical_cols = ['Company', 'TypeName', 'OpSys','CPU_Brand', 'CPU_Type', 'GPU_Brand']
    df_encoded = encode_categorical_features(df, categorical_cols)

   
    cols_to_drop = ['ScreenResolution', 'Cpu', 'Memory', 'Gpu']
    df_encoded = df_encoded.drop(columns=cols_to_drop, errors='ignore')

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = split_and_scale_data(df_encoded)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, df_processed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def set_plot_style():
    #stil za grafike
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_price_distribution(df):

    # funkcija prikazuje  istribuciju cena laptopa
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(df['Price_euros'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Cena(EUR)', fontsize=12)
    axes[0].set_ylabel('Broj laptopa', fontsize=12)
    axes[0].set_title('Price_euros distribucija', fontsize=14, fontweight='bold')
    axes[0].axvline(df['Price_euros'].mean(), color='red', linestyle='--', linewidth=2, label=f'Srednja vrednost: {df["Price_euros"].mean():.2f}€')
    axes[0].axvline(df['Price_euros'].median(), color='green', linestyle='--', linewidth=2, label=f'Medijana: {df["Price_euros"].median():.2f}€')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    

    axes[1].boxplot(df['Price_euros'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Cena(EUR)', fontsize=12)
    axes[1].set_title('Box Plot cena-detekcija outliera', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print("Statiskitka: ")
    print("=" * 60)
    print(f"Srednja vrednost: {df['Price_euros'].mean():.2f} €")
    print(f"Medijana:         {df['Price_euros'].median():.2f} €")
    print(f"Std. devijacija:  {df['Price_euros'].std():.2f} €")
    print(f"Minimum:          {df['Price_euros'].min():.2f} €")
    print(f"Maximum:          {df['Price_euros'].max():.2f} €")
    print(f"25% percentil:    {df['Price_euros'].quantile(0.25):.2f} €")
    print(f"75% percentil:    {df['Price_euros'].quantile(0.75):.2f} €")
    print("=" * 60)


def plot_categorical_distributions(df):

    categorical_cols = ['Company', 'TypeName', 'OpSys',
                        'CPU_Brand', 'CPU_Type', 'GPU_Brand']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, col in enumerate(categorical_cols):
        if col in df.columns:

            value_counts = df[col].value_counts().head(8).sort_values()

            axes[idx].barh(value_counts.index,
                           value_counts.values,
                           color='#4C72B0')

            axes[idx].set_title(f'{col} distribucija',
                                fontsize=13,
                                fontweight='bold')

            axes[idx].set_xlabel('Broj laptopova')

            for i, v in enumerate(value_counts.values):
                axes[idx].text(v, i,
                               f' {v}',
                               va='center',
                               fontsize=10)

            axes[idx].grid(False)

    plt.tight_layout()
    plt.show()

def plot_numeric_distributions(df, col, bins=30):
    
    x = df[col]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # histogram
    axes[0].hist(x, bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(col, fontsize=12)
    axes[0].set_ylabel('Broj laptopova', fontsize=12)
    axes[0].set_title(f'Distribucija: {col}', fontsize=14, fontweight='bold')
    
    axes[0].axvline(x.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {x.mean():.2f}')
    axes[0].axvline(x.median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {x.median():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #  boxplot
    axes[1].boxplot(x, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel(col, fontsize=12)
    axes[1].set_title(f'Boxplot: {col}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print(f"Statistika:")
    print("=" * 60)
    print(f"Srednja vrednost: {x.mean():.2f}")
    print(f"Medijana:         {x.median():.2f}")
    print(f"Std. devijacija:  {x.std():.2f}")
    print(f"Minimum:          {x.min():.2f}")
    print(f"Maximum:          {x.max():.2f}")
    print(f"25% percentil:    {x.quantile(0.25):.2f}")
    print(f"75% percentil:    {x.quantile(0.75):.2f}")
    print("=" * 60)

def plot_binary_distribution(df, cols):
    
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    
    if n == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols):
        counts = df[col].value_counts().sort_index()
        counts.index = ["Ne", "Da"]  
        
        counts.plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Broj laptopova")
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_price_by_categorical(df, categorical_col):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # box plot
    top_categories = df[categorical_col].value_counts().head(10).index
    df_filtered = df[df[categorical_col].isin(top_categories)]
    
    axes[0].boxplot([df_filtered[df_filtered[categorical_col] == cat]['Price_euros'].values 
                     for cat in top_categories],
                    labels=top_categories, patch_artist=True)
    axes[0].set_xlabel(categorical_col, fontsize=12)
    axes[0].set_ylabel('Cena (EUR)', fontsize=12)
    axes[0].set_title(f'Distribucija cena po {categorical_col}', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # bar plot(prosecne cene)
    avg_prices = df_filtered.groupby(categorical_col)['Price_euros'].mean().sort_values(ascending=False)
    axes[1].barh(range(len(avg_prices)), avg_prices.values, color='steelblue', alpha=0.7)
    axes[1].set_yticks(range(len(avg_prices)))
    axes[1].set_yticklabels(avg_prices.index)
    axes[1].set_xlabel('Prosecna cena (EUR)', fontsize=12)
    axes[1].set_title(f'Prosecne cene po {categorical_col}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(avg_prices.values):
        axes[1].text(v + max(avg_prices.values)*0.01, i, f'{v:.0f}€', 
                    va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_price_by_numeric(df, numeric_col):
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # scatter plot s regresionom linijom
    x = df[numeric_col].dropna()
    y = df.loc[x.index, 'Price_euros']
    
    ax.scatter(x, y, alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidths=0.5)
    
    # ovde dodajem regresionu liniju
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", linewidth=2, label=f'Regresija: y={z[0]:.2f}x+{z[1]:.2f}')
    
    correlation = np.corrcoef(x, y)[0, 1]
    
    ax.set_xlabel(numeric_col, fontsize=12)
    ax.set_ylabel('Cena (EUR)', fontsize=12)
    ax.set_title(f'Odnos: {numeric_col} vs Cena (korelacija: {correlation:.3f})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"\n{'='*60}")
    print("Outlier analiza:")
    print(f"{'='*60}")
    print(f"Q1 (25%):         {Q1:.2f}")
    print(f"Q3 (75%):         {Q3:.2f}")
    print(f"IQR:              {IQR:.2f}")
    print(f"Donja granica:    {lower_bound:.2f}")
    print(f"Gornja granica:   {upper_bound:.2f}")
    print(f"Broj outliera:    {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return outliers


def plot_correlation_matrix(df, numeric_cols):
    
    numeric_cols = numeric_cols + ['Price_euros']
    corr_matrix = df[numeric_cols].corr()
    
    # heatmeap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Matrica korelacija', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # najjace korelacije sa cenom
    price_corr = corr_matrix['Price_euros'].drop('Price_euros').sort_values(ascending=False)
    for feature, corr in price_corr.head(10).items():
        print(f"{feature:25s}: {corr:6.3f}")
    print(f"{'='*60}\n")


def calculate_vif(df, numeric_features):
   
    X = df[numeric_features].dropna()
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_features
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(numeric_features))
    ]
    
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    print("VIF rezultati:")
    print(vif_data)
    
    return vif_data

def plot_memory_types_distribution(df):
   
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # pie chart distribucija ntipova
    memory_counts = pd.Series({
        'SSD': df['Has_SSD'].sum(),
        'HDD': df['Has_HDD'].sum(),
        'Flash': df['Has_Flash'].sum(),
        'Hybrid': df['Has_Hybrid'].sum()
    })
    
    axes[0].pie(memory_counts.values, labels=memory_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    axes[0].set_title('Distribucija tipova memorije', fontsize=14, fontweight='bold')
    
    #box plot cene po tipovima ategorijeee
    memory_types = []
    prices = []
    
    for mem_type in ['Has_SSD', 'Has_HDD', 'Has_Flash', 'Has_Hybrid']:
        if mem_type in df.columns:
            prices_subset = df[df[mem_type] == 1]['Price_euros'].values
            if len(prices_subset) > 0:
                memory_types.append(mem_type.replace('Has_', ''))
                prices.append(prices_subset)
    
    axes[1].boxplot(prices, labels=memory_types, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_xlabel('Tip memorije', fontsize=12)
    axes[1].set_ylabel('Cena (EUR)', fontsize=12)
    axes[1].set_title('Cene prema tipu memorije', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

    plt.show()


def analyze_multidrive_effect(df):
    from scipy.stats import ttest_ind
    
    mean_prices = df.groupby('Has_MultiDrive')['Price_euros'].mean()
    
    multi_mean = mean_prices.get(1, 0)
    single_mean = mean_prices.get(0, 0)
    difference = multi_mean - single_mean
    
    print("Prosecne cene:")
    print(f"Bez MultiDrive: {single_mean:.2f} EUR")
    print(f"Sa MultiDrive:  {multi_mean:.2f} EUR")
    print(f"Razlika u prosecnoj ceni: {difference:.2f} EUR")
    
    # t-test
    multi = df[df['Has_MultiDrive'] == 1]['Price_euros']
    single = df[df['Has_MultiDrive'] == 0]['Price_euros']
    
    t_stat, p_value = ttest_ind(multi, single, equal_var=False)
    
    print(f"\nT-statistika: {t_stat:.3f}")
    print(f"P-vrednost: {p_value:.5f}")

def plot_key_interactions(df):
  
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #  RAM × CPU Speed
    if 'Ram' in df.columns and 'CPU_Speed_GHz' in df.columns:
        ram_cpu = df['Ram'] * df['CPU_Speed_GHz']
        
        axes[0].scatter(ram_cpu, df['Price_euros'], alpha=0.5)
        
        z = np.polyfit(ram_cpu, df['Price_euros'], 1)
        p = np.poly1d(z)
        axes[0].plot(ram_cpu, p(ram_cpu), "r--")

        corr = np.corrcoef(ram_cpu, df['Price_euros'])[0, 1]
        axes[0].set_title(f'RAM x CPU Speed (r={corr:.3f})')
        axes[0].set_xlabel('RAM x CPU Speed')
        axes[0].set_ylabel('Cena')

    #  pixel Density
    if 'Total_Pixels' in df.columns and 'Inches' in df.columns:
        pixel_density = df['Total_Pixels'] / (df['Inches'] ** 2)
        
        axes[1].scatter(pixel_density, df['Price_euros'], alpha=0.5)
        
        z = np.polyfit(pixel_density, df['Price_euros'], 1)
        p = np.poly1d(z)
        axes[1].plot(pixel_density, p(pixel_density), "r--")

        corr = np.corrcoef(pixel_density, df['Price_euros'])[0, 1]
        axes[1].set_title(f'Pixel Density (r={corr:.3f})')
        axes[1].set_xlabel('Pixel Density')
        axes[1].set_ylabel('Cena')

    plt.tight_layout()
    plt.show()
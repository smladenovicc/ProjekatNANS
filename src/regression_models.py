import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


class RegressionModels:
    
    def __init__(self):
        
        self.models = {
            "OLS": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Huber": HuberRegressor(epsilon=1.35, max_iter=200),
        }
        self.results = {}
        self.predictions = {}
        self.vif_results = None

    
    def analyze_multicollinearity(self, X_train):

        X_vif = X_train.select_dtypes(include=[np.number]).astype(float) #ovo mora zbog bool vrednosti
        feature_names = X_vif.columns.tolist()

        X = X_vif.values

        for i, name in enumerate(feature_names):
            vif = variance_inflation_factor(X, i)
            print(f"{name}: {vif:.2f}")

            

    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        
        self.results = {}
        self.predictions = {}


        for name, model in self.models.items():

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            self.results[name] = {
                "train_mae": train_mae,
                "train_r2": train_r2,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "model": model,
            }

            self.predictions[name] = {
                "train": (y_train, y_train_pred),
                "val": (y_val, y_val_pred),
            }

            print(f"{name}")
            print(f"  Train → MAE: {train_mae:.2f}, R2: {train_r2:.4f}")
            print(f"  Val   → MAE: {val_mae:.2f}, R2: {val_r2:.4f}\n")


    def evaluate_on_test(self, X_test, y_test):

        for name, model_data in self.results.items():

            model = model_data["model"]
            y_test_pred = model.predict(X_test)

            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            self.results[name]["test_mae"] = test_mae
            self.results[name]["test_r2"] = test_r2
            self.predictions[name]["test"] = (y_test, y_test_pred)

            print(f"{name}")
            print(f"  MAE: {test_mae:.2f}")
            print(f"  R2 : {test_r2:.4f}\n")



    def compare_ols_ridge_coefficients(self, feature_names, top_n=20):

        ols_model = self.results["OLS"]["model"]
        ridge_model = self.results["Ridge"]["model"]

        ols_coefs = ols_model.coef_
        ridge_coefs = ridge_model.coef_

        results = []

        for name, ols, ridge in zip(feature_names, ols_coefs, ridge_coefs):
            diff = abs(ols - ridge)
            results.append((name, ols, ridge, diff))

        results.sort(key=lambda x: x[3], reverse=True)

        print(f"{'Feature':<30} {'OLS':>10} {'Ridge':>10} {'Razlika':>12}")
        print("-" * 75)

        for feature, ols, ridge, diff in results[:top_n]:
            print(f"{feature:<30} {ols:>10.3f} {ridge:>10.3f} {diff:>12.3f}")

        
   
    def analyze_feature_importance(self, feature_names, model_name="Ridge", top_n=20):

        model = self.results[model_name]["model"]
        coefficients = model.coef_

        features = []

        for name, coef in zip(feature_names, coefficients):
            features.append((name, coef, abs(coef)))

    
        features.sort(key=lambda x: x[2], reverse=True) #sortiram po apsolutnoj vrednosti koef

        print(f"\n{model_name}")
        print("-" * 65)
        print(f"{'Feature':<35} {'Koeficijent':>15}")
        print("-" * 65)

        for name, coef, _ in features[:top_n]:
            print(f"{name:<35} {coef:>15.3f}")
    

    def analyze_outlier_impact(self, X_train, y_train, X_val, y_val, outlier_threshold_pct=95):

        price_threshold = np.percentile(y_train, outlier_threshold_pct)
        outlier_mask = y_train > price_threshold

        X_train_no = X_train[~outlier_mask]
        y_train_no = y_train[~outlier_mask]

        models_local = {
            "OLS": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Huber": HuberRegressor(epsilon=1.35, max_iter=200),
        }

        for name, model in models_local.items():
            model.fit(X_train_no, y_train_no)
            y_val_pred = model.predict(X_val)

            mae_no = mean_absolute_error(y_val, y_val_pred)
            r2_no = r2_score(y_val, y_val_pred)

            mae_with = self.results.get(name, {}).get("val_mae", np.nan)
            r2_with = self.results.get(name, {}).get("val_r2", np.nan)

            print(f"{name}:")
            print(f"  MAE with={mae_with:.2f}  without={mae_no:.2f}  diff={mae_with - mae_no:+.2f}")
            print(f"  R2  with={r2_with:.4f}  without={r2_no:.4f}  diff={r2_no - r2_with:+.4f}")
        
    def plot_feature_importance_comparison(self, feature_names, top_n=15):

        required_models = ["OLS", "Ridge", "Huber"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, model_name in enumerate(required_models):
            model = self.results[model_name]["model"]
            coefs = model.coef_

            df = pd.DataFrame({
                "Feature": feature_names,
                "Coef": coefs
            })

            
            df = df.reindex(df["Coef"].abs().sort_values(ascending=False).index)
            df_top = df.head(top_n).sort_values("Coef")

            colors = ["green" if x > 0 else "red" for x in df_top["Coef"]]

            axes[idx].barh(df_top["Feature"], df_top["Coef"], color=colors)
            axes[idx].axvline(0, color="black")
            axes[idx].set_title(model_name)
            axes[idx].set_xlabel("Koeficijent")

            print("\n" + "="*70)
            print(f"MODEL: {model_name}")
            print("="*70)
            for _, row in df_top.sort_values("Coef", key=abs, ascending=False).iterrows():
                znak = "+" if row["Coef"] > 0 else "-"
                print(f"{row['Feature']:35s} {znak}{abs(row['Coef']):.2f}")

        plt.tight_layout()
        plt.show()
    
    def plot_predictions_and_residuals(self):
        
        required_models = ["OLS", "Ridge", "Huber"]
        

        fig, axes = plt.subplots(2, 3, figsize=(22, 10))

        for idx, model_name in enumerate(required_models):
            y_val, y_val_pred = self.predictions[model_name]["val"]
            residuals = y_val - y_val_pred

           
            axes[0, idx].scatter(y_val, y_val_pred, alpha=0.5, s=18, edgecolors="black", linewidths=0.3)
            min_v = min(y_val.min(), y_val_pred.min())
            max_v = max(y_val.max(), y_val_pred.max())
            axes[0, idx].plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)

            axes[0, idx].set_title(f"{model_name} (Val R2={self.results[model_name]['val_r2']:.4f})")
            axes[0, idx].set_xlabel("Stvarna cena")
            axes[0, idx].set_ylabel("Predvidjena cena")
            axes[0, idx].grid(True, alpha=0.3)

            
            axes[1, idx].scatter(y_val_pred, residuals, alpha=0.5, s=18, edgecolors="black", linewidths=0.3)
            axes[1, idx].axhline(0, linestyle="--", linewidth=1)
            axes[1, idx].set_title(f"Reziduali (Val MAE={self.results[model_name]['val_mae']:.2f})")
            axes[1, idx].set_xlabel("Predvidjena cena")
            axes[1, idx].set_ylabel("Rezidual")
            axes[1, idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    
    def plot_model_comparison(self):
       
        metrics_data = []
        for name, results in self.results.items():
            metrics_data.append({
                "Model": name,
                "Train MAE": results.get("train_mae", np.nan),
                "Val MAE": results.get("val_mae", np.nan),
                "Test MAE": results.get("test_mae", np.nan),
                "Train R2": results.get("train_r2", np.nan),
                "Val R2": results.get("val_r2", np.nan),
                "Test R2": results.get("test_r2", np.nan),
            })

        metrics_df = pd.DataFrame(metrics_data).set_index("Model")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        metrics_df[["Train MAE", "Val MAE", "Test MAE"]].plot(kind="bar", ax=axes[0], alpha=0.85)
        axes[0].set_title("MAE ")
        axes[0].set_ylabel("MAE")
        axes[0].set_xlabel("Model")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis="x", rotation=15)

        metrics_df[["Train R2", "Val R2", "Test R2"]].plot(kind="bar", ax=axes[1], alpha=0.85)
        axes[1].set_title("R2 ")
        axes[1].set_ylabel("R2")
        axes[1].set_xlabel("Model")
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plt.show()

    def brand_effect_report(self, feature_names, model_name="Ridge", top_n=10):
        
        model = self.results[model_name]["model"]
        coefs = model.coef_

        df = pd.DataFrame({
            "Feature": feature_names,
            "Coef": coefs
        })

        # uzimam samo brend kolone
        brand_df = df[df["Feature"].str.startswith("Company")].copy()

        brand_df["AbsCoef"] = brand_df["Coef"].abs()
        brand_df = brand_df.sort_values("AbsCoef", ascending=False)

        # top pozitivni (skuplji)
        top_pos = brand_df.sort_values("Coef", ascending=False).head(top_n)
        print(f"\nBrendovi koji povecavau cenu (pozitivan koeficijent):")
        for _, row in top_pos.iterrows():
            print(f"  {row['Feature']:25s}  coef = {row['Coef']:+.3f}")

        # top negativni (jeftiniji)
        top_neg = brand_df.sort_values("Coef", ascending=True).head(top_n)
        print(f"\nBbrendovi koji Ssmanjuju cenu (negativan koeficijent):")
        for _, row in top_neg.iterrows():
            print(f"  {row['Feature']:25s}  coef = {row['Coef']:+.3f}")

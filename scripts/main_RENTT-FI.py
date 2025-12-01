import sys
from collections import Counter

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from scripts.main_RENTT import create_neural_network
from src.config import ConfigWrapper, RENTTFIConfig
from src.DT import CompleteTree, Tree
from src.feature_importance import (DalexFeatureImportance,
                                    KrippendorffAlphaAnalysis, Plot,
                                    RENTTFeatureImportance, RMSEAnalysis,
                                    SageFeatureImportance)
from src.read_data import load_and_prepare_dataset
from src.utils import (Framework, TaskType, create_directories,
                       extract_hidden_layers_from_filename)

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def save_results_to_csv(result_dict: dict, filename: str):
    df = pd.DataFrame({k: [v] for k, v in result_dict.items()})
    df.to_csv(filename, mode="w", encoding="utf-8", index=False, sep=";")

def set_computation_flags(config: RENTTFIConfig) -> dict:
    flags = {
        "global_sota_nn": config.compute_global_comparison_sota_nn_dt or config.compute_global_comparison_sota_rentt,
        "global_sota_dt": config.compute_global_comparison_sota_nn_dt,
        "local_sota_nn": config.compute_local_comparison_sota_nn_dt or config.compute_local_comparison_sota_rentt,
        "local_sota_dt": config.compute_local_comparison_sota_nn_dt,
    }
    return flags

# ===========================================================================
# PREPARATION OF MODELS
# ===========================================================================

def prepare_models(config: RENTTFIConfig, directories: dict, hidden_layers: list, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    nn = create_neural_network(config, directories["models"], hidden_layers)
    neural_network_filename = "DNNmodel_" + config.dataset.model_filename + (".h5" if config.framework == Framework.TENSORFLOW else ".pt")
    nn.create_model(load_existing=True, model_filename=neural_network_filename, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    if config.use_complete_tree:
        output_file_decision_tree = str(directories["models"]) + "/CompleteTree_" + config.dataset.model_filename + ".pkl"
    else:
        output_file_decision_tree = str(directories["models"]) + "/PrunedTree_" + config.dataset.model_filename + ".pkl"

    layers_info = nn.load_layer_info()

    print("Loading Decision Tree model...")

    TreeClass = CompleteTree if config.use_complete_tree else Tree
    
    tree = TreeClass(
        layers_info,
        num_input=config.dataset.num_features,
        activation_function=config.activation_function,
        task_type=config.dataset.task_type.value,
        num_output=config.dataset.num_output,
        framework=config.framework
    )
    tree.load_tree(output_file_decision_tree)
    return nn, tree

# ==========================================================================
# FEATURE IMPORTANCE CALCULATIONS
# ==========================================================================

def compute_rentt_feature_importances(config: RENTTFIConfig, directories: dict, tree, X_total: np.ndarray, y_total: np.ndarray, results: dict, plotting: bool=True):
    RENTT = RENTTFeatureImportance(config)
    if plotting:
        plot = Plot(config)
    # local
    if config.compute_local_fe:
        fe, fe_intercept = RENTT.calculate_local_feature_effect(X_total[:config.num_samples], tree)
        results["local"].update({
            "RENTT_FI_local_feature_effect": fe,
            "RENTT_FI_local_feature_effect_intercept": fe_intercept
        })

    if config.compute_local_fc:
        fc_local, fc_local_intercept = RENTT.calculate_local_feature_contribution(
            X_total[:config.num_samples], tree,
            results["local"].get("RENTT_FI_local_feature_effect", None),
            results["local"].get("RENTT_FI_local_feature_effect_intercept", None)
        )
        results["local"].update({
            "RENTT_FI_local_feature_contribution": fc_local,
            "RENTT_FI_local_feature_contribution_intercept": fc_local_intercept
        })

    # regional
    if config.compute_regional_fe:
        fe_reg, fe_reg_intercept = RENTT.calculate_regional_feature_effect(tree)
        results["regional"].update({
            "RENTT_FI_regional_feature_effect": fe_reg,
            "RENTT_FI_regional_feature_effect_intercept": fe_reg_intercept
        })

    if config.compute_regional_fc:
        fc_reg, fc_reg_intercept = RENTT.calculate_regional_feature_contribution(
            X_total[:config.num_samples], tree,
            results["local"].get("RENTT_FI_local_feature_contribution", None),
            results["local"].get("RENTT_FI_local_feature_contribution_intercept", None)
        )
        results["regional"].update({
            "RENTT_FI_regional_feature_contribution": fc_reg,
            "RENTT_FI_regional_feature_contribution_intercept": fc_reg_intercept
        })
    
    # gloabl
    if config.compute_global_fe:
        fe_global, fe_global_intercept = RENTT.calculate_global_feature_effect(tree)
        if config.dataset.task_type == TaskType.CLASSIFICATION:
            classes = Counter(y_total)
            classes = dict(sorted(classes.items()))
            weights = np.array(list(classes.values()))
            fe_class_agg = np.average(fe_global, axis=0, weights=weights)
            fe_intercept_class_agg = np.average(fe_global_intercept, axis=0, weights=weights)
            results["global"].update({
                "RENTT_FI_global_feature_effect_classwise": fe_global,
                "RENTT_FI_global_feature_effect_intercept_classwise": fe_global_intercept,
                "RENTT_FI_global_feature_effect_class_aggregation": fe_class_agg,
                "RENTT_FI_global_feature_effect_intercept_class_aggregation": fe_intercept_class_agg,
            })
            if plotting:
                if "sorting" not in results:
                    sorting = np.argsort(fe_class_agg)
                    results["sorting"] = sorting
                plot.plot_feature_importances(
                    directories["feature_importance"],
                    fe_class_agg,
                    "RENTT-FI Feature Effect (Class Aggregated)",
                    sorting=None
                )
                plot.plot_feature_importances(
                    directories["feature_importance"],
                    fe_global,
                    "RENTT-FI Feature Effect (Classwise)",
                    sorting=None
                )
        elif config.dataset.task_type == TaskType.REGRESSION:
            results["global"].update({
                "RENTT_FI_global_feature_effect": fe_global,
                "RENTT_FI_global_feature_effect_intercept": fe_global_intercept,
            })
            if plotting is not None:
                if "sorting" not in results:
                    sorting = np.argsort(fe_global)
                    results["sorting"] = sorting
                plot.plot_feature_importances(
                    directories["feature_importance"],
                    fe_global,
                    "RENTT-FI Feature Effect",
                    sorting=None
                )

    if config.compute_global_fc:
        fc_global, fc_global_intercept = RENTT.calculate_global_feature_contribution(
            X_total[:config.num_samples], tree, y_total[:config.num_samples],
            results["local"].get("RENTT_FI_local_feature_contribution", None),
            results["local"].get("RENTT_FI_local_feature_contribution_intercept", None)
        )
        if config.dataset.task_type == TaskType.CLASSIFICATION:
            fc_global = np.array(fc_global)
            fc_global_intercept = np.array(fc_global_intercept)
            classes = Counter(y_total)
            classes = dict(sorted(classes.items()))
            weights = np.array(list(classes.values()))

            has_nan = np.isnan(fc_global).any(axis=1)
            if has_nan.any():
                print(f"WARNING: Feature Contribution has NaN for classes: {np.where(has_nan)[0].tolist()}")

            valid_mask = ~has_nan
            if valid_mask.any():
                fc_class_agg = np.average(
                    fc_global[valid_mask], 
                    axis=0, 
                    weights=weights[valid_mask]
                )
                fc_intercept_class_agg = np.average(
                    fc_global_intercept[valid_mask], 
                    axis=0, 
                    weights=weights[valid_mask]
                )
            else:
                print("ERROR: All classes have NaN values, using zeros")
                fc_class_agg = np.zeros(fc_global.shape[1])
                fc_intercept_class_agg = np.zeros(fc_global_intercept.shape[1])
            
            fc_class_agg = np.average(fc_global, axis=0, weights=weights)
            fc_intercept_class_agg = np.average(fc_global_intercept, axis=0, weights=weights)
           
            results["global"].update({
                "RENTT_FI_global_feature_contribution_classwise": fc_global,
                "RENTT_FI_global_feature_contribution_intercept_classwise": fc_global_intercept,
                "RENTT_FI_global_feature_contribution_class_aggregation": fc_class_agg,
                "RENTT_FI_global_feature_contribution_intercept_class_aggregation": fc_intercept_class_agg,
            })
            if plotting is not None:
                if "sorting" not in results:
                    sorting = np.argsort(fc_class_agg)
                    results["sorting"] = sorting
                plot.plot_feature_importances(
                    directories["feature_importance"],
                    fc_class_agg,
                    "RENTT-FI Feature Contribution (Class Aggregated)",
                    sorting=None
                )
                plot.plot_feature_importances(
                    directories["feature_importance"],
                    fc_global,
                    "RENTT-FI Feature Contribution (Classwise)",
                    sorting=None
                )
        elif config.dataset.task_type == TaskType.REGRESSION:
            results["global"].update({
                "RENTT_FI_global_feature_contribution": fc_global,
                "RENTT_FI_global_feature_contribution_intercept": fc_global_intercept,
            })
            if plotting is not None:
                if "sorting" not in results:
                    sorting = np.argsort(fc_global)
                    results["sorting"] = sorting
                plot.plot_feature_importances(
                    directories["feature_importance"],
                    fc_global,
                    "RENTT-FI Feature Contribution",
                    sorting=None
                )

def calculate_global_sota_methods(config: RENTTFIConfig, nn, tree, directories: dict, X_train: np.ndarray, y_train: np.ndarray, X_total: np.ndarray, y_total: np.ndarray, results: dict, flags: dict, plotting: bool=True):
    if plotting == True:
        plot = Plot(config)
    if flags["global_sota_nn"] or flags["global_sota_dt"]:
        Sage = SageFeatureImportance(X_train, y_train, config)
    if flags["global_sota_nn"]:
        shap_nn, shap_std_nn, shap_intercept_nn, sage_nn, sage_std_nn, sage_intercept_nn = Sage.calculate_global_feature_importance_nn(
            nn, X_total[:config.num_samples], y_total[:config.num_samples]
        )
        results["global"].update({
            "Shapley_Effect_NN": shap_nn,
            "Shapley_Effect_Std_NN": shap_std_nn,
            "Shapley_Effect_Intercept_NN": shap_intercept_nn,
            "SAGE_NN": sage_nn,
            "SAGE_Std_NN": sage_std_nn,
            "SAGE_Intercept_NN": sage_intercept_nn
        })
        if plotting == True:
            if "sorting" not in results:
                sorting = np.argsort(shap_nn)
                results["sorting"] = sorting
            plot.plot_feature_importances(directories["feature_importance"], shap_nn, "Shapley Effect NN", sorting=None)
            plot.plot_feature_importances(directories["feature_importance"], sage_nn, "Sage NN", sorting=None)

    if flags["global_sota_dt"]:
        shap_dt, shap_std_dt, shap_intercept_dt, sage_dt, sage_std_dt, sage_intercept_dt = Sage.calculate_global_feature_importance_dt(
            tree, X_total[:config.num_samples], y_total[:config.num_samples]
        )
        results["global"].update({
            "Shapley_Effect_DT": shap_dt,
            "Shapley_Effect_Std_DT": shap_std_dt,
            "Shapley_Effect_Intercept_DT": shap_intercept_dt,
            "SAGE_DT": sage_dt,
            "SAGE_Std_DT": sage_std_dt,
            "SAGE_Intercept_DT": sage_intercept_dt,
        })
        if plotting == True:
            if "sorting" not in results:
                sorting = np.argsort(shap_dt)
                results["sorting"] = sorting
            plot.plot_feature_importances(directories["feature_importance"], shap_dt, "Shapley Effect DT", sorting=None)
            plot.plot_feature_importances(directories["feature_importance"], sage_dt, "Sage DT", sorting=None)

def calculate_local_sota_methods(config: RENTTFIConfig, nn, tree, X_train: np.ndarray, y_train: np.ndarray, X_total: np.ndarray, results: dict, flags: dict):
    if flags["local_sota_nn"] or flags["local_sota_dt"]:
        dalex_calculator = DalexFeatureImportance(X_train, y_train, config)
    if flags["local_sota_nn"]:
        lime_nn, shap_nn, bd_nn, lime_intercept_nn, shap_intercept_nn, bd_intercept_nn = dalex_calculator.calculate_local_feature_importance_nn(
            nn, X_total[:config.num_samples]
        )
        results["local"].update({
            "LIME_NN": lime_nn, "LIME_Intercept_NN": lime_intercept_nn,
            "SHAP_NN": shap_nn, "SHAP_Intercept_NN": shap_intercept_nn,
            "BD_NN": bd_nn, "BD_Intercept_NN": bd_intercept_nn,
        })
    if flags["local_sota_dt"]:
        lime_dt, shap_dt, bd_dt, lime_intercept_dt, shap_intercept_dt, bd_intercept_dt = dalex_calculator.calculate_local_feature_importance_dt(
            tree, X_total[:config.num_samples]
        )
        results["local"].update({
            "LIME_DT": lime_dt, "LIME_Intercept_DT": lime_intercept_dt,
            "SHAP_DT": shap_dt, "SHAP_Intercept_DT": shap_intercept_dt,
            "BD_DT": bd_dt, "BD_Intercept_DT": bd_intercept_dt,
        })

# ==========================================================================
# COMPARISON ANALYSIS
# ===========================================================================
def run_comparison_analysis(config: RENTTFIConfig, results: dict, directories: dict):
    if config.compute_local_comparison_sota_nn_dt:
        run_local_nn_dt_comparison(config, results, directories)

    if config.compute_local_comparison_sota_rentt:
        run_local_sota_rentt_comparison(config, results, directories)

    if config.compute_global_comparison_sota_nn_dt:
        run_global_nn_dt_comparison(config, results, directories)

    if config.compute_global_comparison_sota_rentt:
        run_global_sota_rentt_comparison(config, results, directories)


def run_local_nn_dt_comparison(config: RENTTFIConfig, results: dict, directories: dict):
    RMSE = RMSEAnalysis(config)
    rmse_lime = RMSE.calculate_rmse(
        ["LIME_NN", "LIME_DT"],
        [
            results["local"].get("LIME_NN"),
            results["local"].get("LIME_DT")
        ],
    )
    rmse_shap = RMSE.calculate_rmse(
        ["SHAP_NN", "SHAP_DT"],
        [
            results["local"].get("SHAP_NN"),
            results["local"].get("SHAP_DT")
        ],
    )
    rmse_bd = RMSE.calculate_rmse(
        ["BD_NN", "BD_DT"],
        [
            results["local"].get("BD_NN"),
            results["local"].get("BD_DT")
        ],
    )
    RMSE.save_rmse_stats(
        [rmse_lime, rmse_shap, rmse_bd],
        f"{directories['feature_importance']}/rmse_local_nn_vs_dt.csv"
    )

def run_local_sota_rentt_comparison(config: RENTTFIConfig, results: dict, directories: dict):
    RMSE = RMSEAnalysis(config)
    try:
        lime = np.concatenate([
            results["local"]["LIME_NN"], 
            results["local"]["LIME_Intercept_NN"].reshape(-1, 1)
        ], axis=1)
        shap = np.concatenate([
            results["local"]["SHAP_NN"], 
            results["local"]["SHAP_Intercept_NN"].reshape(-1, 1)
        ], axis=1)
        bd = np.concatenate([
            results["local"]["BD_NN"], 
            results["local"]["BD_Intercept_NN"].reshape(-1, 1)
        ], axis=1)
    except Exception:
        lime = np.concatenate([
            results["local"]["LIME_DT"], 
            results["local"]["LIME_Intercept_DT"].reshape(-1, 1)
        ], axis=1)
        shap = np.concatenate([
            results["local"]["SHAP_DT"], 
            results["local"]["SHAP_Intercept_DT"].reshape(-1, 1)
        ], axis=1)
        bd = np.concatenate([
            results["local"]["BD_DT"], 
            results["local"]["BD_Intercept_DT"].reshape(-1, 1)
        ], axis=1)

    rentt = np.concatenate([
        results["local"]["RENTT_FI_local_feature_contribution"], 
        results["local"]["RENTT_FI_local_feature_contribution_intercept"].reshape(-1, 1)
    ], axis=1)

    rmse = RMSE.calculate_rmse(
        ["LIME", "SHAP", "BD", "RENTT-FI"],
        [lime, shap, bd, rentt],
        f"{directories['feature_importance']}/rmse_local_rentt", plot=config.plot_analysis,
    )
    RMSE.save_rmse_stats(rmse, f"{directories['feature_importance']}/rmse_local_rentt.csv")


    if config.dataset.num_features != 1:
        KrippendorffAlpha = KrippendorffAlphaAnalysis(config)
        try:
            lime_ranks = np.argsort(results["local"]["LIME_NN"], axis=1)[:, ::-1]
            shap_ranks = np.argsort(results["local"]["SHAP_NN"], axis=1)[:, ::-1]
            bd_ranks = np.argsort(results["local"]["BD_NN"], axis=1)[:, ::-1]
            fi_ranks = np.argsort(results["local"]["RENTT_FI_local_feature_contribution"], axis=1)[:, ::-1]
        except Exception:
            lime_ranks = np.argsort(results["local"]["LIME_DT"], axis=1)[:, ::-1]
            shap_ranks = np.argsort(results["local"]["SHAP_DT"], axis=1)[:, ::-1]
            bd_ranks = np.argsort(results["local"]["BD_DT"], axis=1)[:, ::-1]
            fi_ranks = np.argsort(results["local"]["RENTT_FI_local_feature_contribution"], axis=1)[:, ::-1]

        methods_ranked = [lime_ranks, shap_ranks, bd_ranks, fi_ranks]
        methods_valued = [
            results["local"].get("LIME_NN", results["local"].get("LIME_DT")),
            results["local"].get("SHAP_NN", results["local"].get("SHAP_DT")),
            results["local"].get("BD_NN", results["local"].get("BD_DT")),
            results["local"]["RENTT_FI_local_feature_contribution"]
        ]

        alphas_all_ordinal = KrippendorffAlpha.krippendorff_alpha_local(
            methods_ranked, "ordinal"
        )
        alphas_all_interval = KrippendorffAlpha.krippendorff_alpha_local(
            methods_valued, "interval"
        )

        pd.DataFrame({
            "Krippendorff_Alpha_LIME_vs_SHAP_Ordinal": [alphas_all_ordinal[0]],
            "Krippendorff_Alpha_LIME_vs_BD_Ordinal": [alphas_all_ordinal[1]],
            "Krippendorff_Alpha_LIME_vs_RENTT_FI_Ordinal": [alphas_all_ordinal[2]],
            "Krippendorff_Alpha_SHAP_vs_BD_Ordinal": [alphas_all_ordinal[3]],
            "Krippendorff_Alpha_SHAP_vs_RENTT_FI_Ordinal": [alphas_all_ordinal[4]],
            "Krippendorff_Alpha_BD_vs_RENTT_FI_Ordinal": [alphas_all_ordinal[5]],
        }).to_csv(
            f"{directories['feature_importance']}/krippendorff_alpha_local_ordinal.csv", index=False, sep=";"
        )
        pd.DataFrame({
            "Krippendorff_Alpha_LIME_vs_SHAP_Interval": [alphas_all_interval[0]],
            "Krippendorff_Alpha_LIME_vs_BD_Interval": [alphas_all_interval[1]],
            "Krippendorff_Alpha_LIME_vs_RENTT_FI_Interval": [alphas_all_interval[2]],
            "Krippendorff_Alpha_SHAP_vs_BD_Interval": [alphas_all_interval[3]],
            "Krippendorff_Alpha_SHAP_vs_RENTT_FI_Interval": [alphas_all_interval[4]],
            "Krippendorff_Alpha_BD_vs_RENTT_FI_Interval": [alphas_all_interval[5]],
        }).to_csv(
            f"{directories['feature_importance']}/krippendorff_alpha_local_interval.csv", index=False, sep=";"
        )

        if config.plot_analysis:
            KrippendorffAlpha.plot_krippendorff_violin(
                method_names=["LIME", "SHAP", "BD", "RENTT-FI"], method_results=alphas_all_ordinal,
                save_path=f"{directories['feature_importance']}/krippendorff_alpha_local_ordinal.png", scale=1
            )
            KrippendorffAlpha.plot_krippendorff_violin(
                method_names=["LIME", "SHAP", "BD", "RENTT-FI"], method_results=alphas_all_interval,
                save_path=f"{directories['feature_importance']}/krippendorff_alpha_local_interval.png", scale=1
            )

def run_global_nn_dt_comparison(config: RENTTFIConfig, results: dict, directories: dict):
    RMSE = RMSEAnalysis(config)
    rmse_shap = RMSE.calculate_rmse(
        ["Shapley_Effect_NN", "Shapley_Effect_DT"],
        [
            results["global"].get("Shapley_Effect_NN"),
            results["global"].get("Shapley_Effect_DT")
        ]
    )
    rmse_sage = RMSE.calculate_rmse(
        ["SAGE_NN", "SAGE_DT"],
        [
            results["global"].get("SAGE_NN"),
            results["global"].get("SAGE_DT")
        ]
    )
    RMSE.save_rmse_stats(
        [rmse_shap, rmse_sage],
        f"{directories['feature_importance']}/rmse_global_nn_vs_dt.csv"
    )

def run_global_sota_rentt_comparison(config: RENTTFIConfig, results: dict, directories: dict):
    KrippendorffAlpha = KrippendorffAlphaAnalysis(config)
    try:
        shap = results["global"]["Shapley_Effect_NN"]
        sage = results["global"]["SAGE_NN"]
    except Exception:
        shap = results["global"]["Shapley_Effect_DT"]
        sage = results["global"]["SAGE_DT"]

    rentt = results["global"].get("RENTT_FI_global_feature_effect_class_aggregation", 
                                    results["global"].get("RENTT_FI_global_feature_effect"))

    shap_ranks = np.argsort(shap, axis=-1)[..., ::-1]
    sage_ranks = np.argsort(sage, axis=-1)[..., ::-1]
    rentt_ranks = np.argsort(rentt, axis=-1)[..., ::-1]

    methods_ranked = [shap_ranks, sage_ranks, rentt_ranks]
    methods_valued = [shap, sage, rentt]
    alphas_all_ordinal = KrippendorffAlpha.krippendorff_alpha_global(
        methods_ranked, "ordinal"
    )
    alphas_all_interval = KrippendorffAlpha.krippendorff_alpha_global(
        methods_valued, "interval"
    )

    pd.DataFrame({
        "Krippendorff_Alpha_SHAP_vs_SAGE": [alphas_all_ordinal[0]],
        "Krippendorff_Alpha_SHAP_vs_RENTT-FI": [alphas_all_ordinal[1]],
        "Krippendorff_Alpha_SAGE_vs_RENTT-FI": [alphas_all_ordinal[2]],
    }).to_csv(
        f"{directories['feature_importance']}/krippendorff_alpha_global_ordinal.csv", index=False
    )
    pd.DataFrame({
        "Krippendorff_Alpha_SHAP_vs_SAGE": [alphas_all_interval[0]],
        "Krippendorff_Alpha_SHAP_vs_RENTT-FI": [alphas_all_interval[1]],
        "Krippendorff_Alpha_SAGE_vs_RENTT-FI": [alphas_all_interval[2]],
    }).to_csv(
        f"{directories['feature_importance']}/krippendorff_alpha_global_interval.csv", index=False
    )

    RMSE = RMSEAnalysis(config)
    try:
        # Einfach die Intercepts als einzelne Werte anhängen
        shap_vec = np.append(shap, results["global"]["Shapley_Effect_Intercept_NN"])
        sage_vec = np.append(sage, results["global"]["SAGE_Intercept_NN"])
        
        rentt_features = results["global"].get(
            "RENTT_FI_global_feature_contribution_class_aggregation", 
            results["global"].get("RENTT_FI_global_feature_contribution")
        )
        rentt_intercept = results["global"].get(
            "RENTT_FI_global_feature_contribution_intercept_class_aggregation", 
            results["global"].get("RENTT_FI_global_feature_contribution_intercept")
        )
        rentt_vec = np.append(np.atleast_1d(rentt_features), rentt_intercept)
        
    except Exception:
        # Fallback auf DT
        shap_vec = np.append(shap, results["global"]["Shapley_Effect_Intercept_DT"])
        sage_vec = np.append(sage, results["global"]["SAGE_Intercept_DT"])
        
        rentt_features = results["global"].get(
            "RENTT_FI_global_feature_contribution_class_aggregation", 
            results["global"].get("RENTT_FI_global_feature_contribution")
        )
        rentt_intercept = results["global"].get(
            "RENTT_FI_global_feature_contribution_intercept_class_aggregation", 
            results["global"].get("RENTT_FI_global_feature_contribution_intercept")
        )
        rentt_vec = np.append(np.atleast_1d(rentt_features), rentt_intercept)

    # Die Funktion macht automatisch reshape(-1, 1) bei 1D-Arrays

    rmse = RMSE.calculate_rmse(
        ["SHAP", "SAGE", "RENTT-FI"],
        [shap_vec, sage_vec, rentt_vec]
    )
    RMSE.save_rmse_stats(rmse, f"{directories['feature_importance']}/rmse_global.csv")


@hydra.main(version_base=None, config_path="../config", config_name="RENTT-FI")
def main(cfg: DictConfig):
    try:
        config= ConfigWrapper(cfg)

        print("Configuration:")
        print(f" Dataset: {config.dataset.name}")
        print(f" Using Framework: {config.framework.name}")
        print(f" Task Type: {config.dataset.task_type.name}")
        print(f" Using Complete Tree: {config.use_complete_tree}")
        print(f" Number of Samples for FI Computation: {config.num_samples}")

        directories = create_directories(config.dataset.name)

        hidden_layers = extract_hidden_layers_from_filename(
            config.dataset.model_filename
        )

        filenames = {
            "local": f"{directories['feature_importance']}/LocalFI_data_{config.dataset.model_filename}.csv",
            "global": f"{directories['feature_importance']}/GlobalFI_data_{config.dataset.model_filename}.csv",
            "regional": f"{directories['feature_importance']}/RegionalFI_data_{config.dataset.model_filename}.csv"
        }

        X_train, X_test, y_train, y_test, X_total, y_total = load_and_prepare_dataset(
            config, directories["data"]
        )
        nn, tree = prepare_models(config, directories, hidden_layers, X_train, X_test, y_train, y_test)
        print("Starting Feature Importance Calculations...")

        results = {
            "global": {"feature_names": np.array(config.dataset.feature_names)},
            "regional": {"feature_names": np.array(config.dataset.feature_names)},
            "local": {"feature_names": np.array(config.dataset.feature_names)}
        }

        flags = set_computation_flags(config)

        compute_rentt_feature_importances(config, directories, tree, X_total, y_total, results)

        if config.compute_regional_fe or config.compute_regional_fc:
            save_results_to_csv(results["regional"], filenames["regional"])


        calculate_global_sota_methods(config, nn, tree, directories, X_train, y_train, X_total, y_total, results, flags)

        if config.compute_global_fe or config.compute_global_fe or flags["global_sota_nn"] or flags["global_sota_dt"]:
            save_results_to_csv(results["global"], filenames["global"])

        calculate_local_sota_methods(config, nn, tree, X_train, y_train, X_total, results, flags)

        if config.compute_local_fe or config.compute_local_fc or flags["local_sota_nn"] or flags["local_sota_dt"]:
            save_results_to_csv(results["local"], filenames["local"])
                

        print("Starting Comparison Analysis...")

        run_comparison_analysis(config, results, directories)

        print(f"\n{'='*70}")
        print("Feature Importance Calculation and Comparison completed!")
        print(f"{'='*70}\n")


    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

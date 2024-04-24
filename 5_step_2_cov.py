import numpy as np
import pandas as pd
import os
import dill
import logging

from datetime import datetime

import mrtool
import pickle as pkl

try:
    import Meta_Regression_Analyses.hiv_tb_malaria.public_code.modeling_library.mr_functions as mr_functions
except ModuleNotFoundError:
    import modeling_library.mr_functions as mr_functions
    import modeling_library.mr_data_cleaning_functions as mr_data_cleaning_functions

log = logging.getLogger()

cause_int_ref_keywords = {
    "malaria_prevention": "malaria vaccines",
    "hiv_aids_prevention": "pre-exposure prophylaxis for hiv",
    "hiv_pr_wo_art": "pre-exposure prophylaxis for hiv",
    "hiv_aids_art": "antiretroviral therapy for hiv for prevention",
    "tuberculosis_prevention": "prophylaxis for people without active tb, tuberculosis screening with tuberculin skin test",
    "tuberculosis_treatment": "antibiotics for tuberculosis",
    "tuberculosis_diagnostic": "xpert rapid tuberculosis test",
    "syphilis_diagnostic": "antibiotics for syphilis, syphilis testing",
    "prep": "pre-exposure prophylaxis for hiv",
    "pep": "post-exposure prophylaxis for hiv",
    "prep_pep": "pre-exposure prophylaxis for hiv",
}

keywords_input = "intervention_keywords_final"


def fit_spline_drop_outliers(args):
    """
    This function calculates the spline for log_gdp variable,
    and drops ratios deemed to be outliers
    """

    # ====== IMPORTING CAUSE-INT COMBINATION DFs ======

    # Specify output directory
    output_dir = os.path.join("HIV_MALARIA", args.version)

    # Retrieve these options from the CLI
    cwalks_used = args.cwalks_used
    cause_int_combinations = args.cause_intervention
    monotonicity_value = args.monotonicity_value
    other_cov_names_spline_fit = [args.dalys_prev_spline_fit]
    kw_spline = args.kw_spline
    percent_outliers_dropped = args.percent_outliers_dropped

    log.info(
        f"Running with the cause intervention = {cause_int_combinations}, cwalks_used = {cwalks_used}, output_dir = {output_dir}, with_spline_fit = {other_cov_names_spline_fit}"
    )

    # importing in cause int combination
    cause_inter = pd.read_csv(f"{output_dir}/{cause_int_combinations}_w_outliers.csv")

    # importing cwalk params
    cwalk_params = pd.read_csv(f"{output_dir}/cwalk_param_summary_to_use.csv")
    cwalk_params = cwalk_params.set_index("covariate")

    # ==== CREATING COVARIATE DICTIONARIES ====

    # Cwalk dictionary with priors for cause_int combinations where they exist
    df_sub = cwalk_params[cwalk_params["cause_int_comb"] == cause_int_combinations]
    cwalk_covs = df_sub.index.to_list()
    cwalk_priors = df_sub[["beta", "se_beta"]]

    cwalk_params_dict = {w: cwalk_priors.loc[w, ["beta", "se_beta"]].to_numpy() for w in cwalk_covs}

    # here need to drop null and inf values for each df subset based on the crosswalk covariates in the cwalk_covs
    cwalk_covs = df_sub.index.to_list()
    cause_inter = cause_inter[cause_inter[cwalk_covs].notnull().all(axis=1)]

    # ===== checking ratios by cause_int combination ===============
    ratio_counts_by_cause_int = mr_data_cleaning_functions.create_df_ratio_counts_per_cause_int(cause_inter)

    # wanting to compare outputs, so saving outputs with date so can a number of times
    date = datetime.now().strftime("%Y_%m_%d-%H.%M.%S")

    # ======== CREATING DUMMY COLS FOR KEYWORDS ==========

    # getting dummy columns
    dummies = pd.get_dummies(cause_inter[keywords_input]).rename(columns=lambda x: str(x))
    cause_inter = pd.concat([cause_inter, dummies], axis=1)

    # identifying what the keyword reference case is to drop from df
    ref_case = cause_int_ref_keywords[cause_int_combinations]
    cause_inter = cause_inter.drop(columns=[ref_case])

    # list of keyword dummies
    keyword_dummies = list(cause_inter[keywords_input].unique())
    keyword_dummies = [x for x in keyword_dummies if x != ref_case]

    if kw_spline == True:
        other_cov_names_spline_fit = other_cov_names_spline_fit + keyword_dummies

    other_cov_names_spline_fit = [i for i in other_cov_names_spline_fit if i is not None]

    log.info(f"keyword dummies: {keyword_dummies}")
    log.info(f"covs_in_spline-fit: {other_cov_names_spline_fit}")

    # ===== FIT SIGNAL MODEL - SPLINE ======

    # Define response name
    response_name = "log_icer_usd"
    se_name = "log_icer_se"
    spline_cov = "log_GDP_2019usd_per_cap"
    study_id_name = "random_effects_id"
    data_id_name = "RatioID"

    # fitting signal
    pkl_path = f"{output_dir}/{cause_int_combinations}_SIGNAL_MR_2.pkl"

    if not os.path.exists(pkl_path):
        signal_mr = mr_functions.fit_signal_model(
            cause_inter,
            resp_name=response_name,
            se_name=se_name,
            spline_cov=spline_cov,
            study_id_name=study_id_name,
            data_id_name=data_id_name,
            # here we add other covariates that will be considered when calculating the spline
            other_cov_names=cwalk_covs + other_cov_names_spline_fit,
            other_cov_gpriors=cwalk_params_dict,
            h=percent_outliers_dropped,
            num_samples=20,
            deg=2,
            n_i_knots=2,
            knots_type="frequency",
            prior_spline_monotonicity=monotonicity_value,
            knot_bounds=np.array([[0.1, 0.6], [0.4, 0.9]]),
            interval_sizes=np.array([[0.1, 0.7], [0.1, 0.7], [0.1, 0.7]]),
        )
        with open(pkl_path, "wb") as out_file:
            dill.dump(signal_mr, out_file)
    else:
        print("signal mr object has already been fitted")
        with open(pkl_path, "rb") as in_file:
            signal_mr = dill.load(in_file)

    signal_df = mr_functions.create_signal(
        signal_mr,
        spline_cov,
        spline_cov_values=cause_inter[spline_cov].to_numpy(),
        data_id_name=data_id_name,
        data_ids=cause_inter[data_id_name].to_numpy(),
    )

    # Calculating w, so as to trim outliers
    w_df = mr_functions.get_ws(signal_mr, data_id_name=data_id_name)

    # adding w to the signal_df
    w_df["RatioID"] = w_df["RatioID"].str.lower()
    signal_df = signal_df.merge(w_df, on=[data_id_name])

    # merging on new spline values and w values
    cause_inter_new_spline = cause_inter.copy()
    cause_inter_new_spline = cause_inter_new_spline.merge(
        signal_df[[data_id_name, "new_spline_cov", "w"]], on=[data_id_name]
    )

    mr_data_cleaning_functions.exporting_csv(
        cause_inter_new_spline, f"{output_dir}/{cause_int_combinations}_data_w_spline_before_dropped.csv"
    )

    # dropping outliers based on
    cause_inter_outliers_dropped = cause_inter_new_spline[cause_inter_new_spline["w"] > 0.5].reset_index()

    ratio_counts_by_cause_int = mr_data_cleaning_functions.create_df_ratio_counts_per_cause_int(
        cause_inter_outliers_dropped
    )

    mr_data_cleaning_functions.exporting_csv(
        ratio_counts_by_cause_int, f"{output_dir}/step_d_ratio_counts_before_criteria_{date}.csv"
    )

    # ====== Tracking which rows were dropped =================
    dropped_outliers = cause_inter_new_spline[cause_inter_new_spline["w"] < 0.5].reset_index()

    mr_data_cleaning_functions.exporting_csv(
        dropped_outliers, f"{output_dir}/{cause_int_combinations}_dropped_outliers_df.csv"
    )

    # ====== DROPPING RATIOS WHERE 2 ARTICLES, 3 ICERS REQUIREMENT NOT MET AFTER OUTLIERS ==========

    # drop keywords where there is only one article per keyword
    cause_inter_outliers_dropped = mr_data_cleaning_functions.drops_kw_not_meeting_criteria(
        cause_inter_outliers_dropped
    )

    mr_data_cleaning_functions.test_kw_dropped(cause_inter_outliers_dropped)

    # ===== CHECKING OUTPUT FROM STEP D =========
    step_d_summary_df = mr_data_cleaning_functions.summarizing_script_outputs(
        cause_inter_outliers_dropped, index_to_calc=["new_spline_cov", "w", "ratio_count"]
    )
    mr_data_cleaning_functions.exporting_csv(
        step_d_summary_df, f"{output_dir}/{cause_int_combinations}_step_d_mean_w_spline.csv"
    )

    # ===== EXPORTING DF WITH SPLINE ADDED AND OUTLIERS REMOVED ====
    mr_data_cleaning_functions.exporting_csv(
        cause_inter_outliers_dropped, f"{output_dir}/{cause_int_combinations}_w_dummies_wo_outliers_df.csv"
    )

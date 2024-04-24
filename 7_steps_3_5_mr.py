import numpy as np
import pandas as pd
import os
import dill
import logging
import pickle as pkl
import copy

from datetime import datetime

# helps to create the interaction variables
from itertools import product

import mrtool

try:
    import mr_functions
except ModuleNotFoundError:
    import modeling_library.mr_functions as mr_functions
    import modeling_library.mr_data_cleaning_functions as mr_data_cleaning_functions

# import ce team functions and paths
try:
    from path_provider import get_path_provider
except ModuleNotFoundError:
    from utilities.path_provider import get_path_provider

# Retrieve and reset the dynamic path provider
pth = get_path_provider()

# use this if you want to reload an updated path files
pth.reset()

log = logging.getLogger()

# Sets of covariate options
covariate_options = {
    "dalys_spline_cdr": [
        "log_dalys_per_cap",
        "new_spline_cov",
        "CostsDiscountRate",
    ],
    "dalys_no_spline_cdr": [
        "log_dalys_per_cap",
        "CostsDiscountRate",
    ],
    "prev_spline_cdr": [
        "log_prevalence_per_cap",
        "new_spline_cov",
        "CostsDiscountRate",
    ],
    "prev_no_spline_cdr": [
        "log_prevalence_per_cap",
        "CostsDiscountRate",
    ],
    "dalys_spline_dr": [
        "log_dalys_per_cap",
        "new_spline_cov",
        "DiscountRate",
    ],
    "dalys_no_spline_dr": [
        "log_dalys_per_cap",
        "DiscountRate",
    ],
    "prev_spline_dr": [
        "log_prevalence_per_cap",
        "new_spline_cov",
        "DiscountRate",
    ],
    "prev_no_spline_dr": [
        "log_prevalence_per_cap",
        "DiscountRate",
    ],
    "only_cdr": [
        "CostsDiscountRate",
    ],
    "only_dr": [
        "DiscountRate",
    ],
    "prev_only": [
        "log_prevalence_per_cap",
    ],
    "gdp_only": [
        "log_GDP_2019usd_per_cap",
    ],
    "spline_only": [
        "new_spline_cov",
    ],
    "dalys_only": [
        "log_dalys_per_cap",
    ],
    "cost_only": [
        "log_per_year_or_full_int_cost",
    ],
    "eff_only": [
        "efficacy",
    ],
    "cost_time": [
        "log_per_year_or_full_int_cost",
        "TimeHorizonMagnitude",
    ],
    "cost_eff": [
        "log_per_year_or_full_int_cost",
        "efficacy",
    ],
    "cost_eff_second": ["log_per_year_or_full_int_cost", "efficacy", "second_line"],
    "cost_eff_dalys": [
        "log_per_year_or_full_int_cost",
        "efficacy",
        "log_dalys_per_cap",
    ],
    "cost_eff_th": ["log_per_year_or_full_int_cost", "efficacy", "TimeHorizonMagnitude"],
    "cost_payer": [
        "log_per_year_or_full_int_cost",
        "payer_or_sector",
    ],
    "cost_eff_payer": [
        "log_per_year_or_full_int_cost",
        "efficacy",
        "payer_or_sector",
    ],
    "cost_payer_cdr": [
        "log_per_year_or_full_int_cost",
        "payer_or_sector",
        "CostsDiscountRate",
    ],
    "cost_prev": ["log_per_year_or_full_int_cost", "log_prevalence_per_cap"],
    "th_payer_cost_cdr": [
        "log_per_year_or_full_int_cost",
        "TimeHorizonMagnitude",
        "payer_or_sector",
        "CostsDiscountRate",
    ],
    "th_payer_cost_cdr_prev": [
        "log_per_year_or_full_int_cost",
        "TimeHorizonMagnitude",
        "payer_or_sector",
        "CostsDiscountRate",
        "log_prevalence_per_cap",
    ],
    "eff_dalys": [
        "efficacy",
        "log_dalys_per_cap",
    ],
    "eff_prev": [
        "efficacy",
        "log_prevalence_per_cap",
    ],
    "cost_payer_dr": [
        "log_per_year_or_full_int_cost",
        "payer_or_sector",
        "DiscountRate",
    ],
    "payer_only": [
        "payer_or_sector",
    ],
    "th_payer_cost_cdr_eff": [
        "log_per_year_or_full_int_cost",
        "TimeHorizonMagnitude",
        "payer_or_sector",
        "CostsDiscountRate",
        "efficacy",
    ],
    "th_payer_cost_eff": [
        "log_per_year_or_full_int_cost",
        "TimeHorizonMagnitude",
        "payer_or_sector",
        "efficacy",
    ],
    "th_payer": [
        "TimeHorizonMagnitude",
        "payer_or_sector",
    ],
    "cost_sens_spec": [
        "log_per_year_or_full_int_cost",
        "sensitivity",
        "specificity",
    ],
    "th_only": [
        "TimeHorizonMagnitude",
    ],
    "rp_cdr_measure": [
        "ReaderPerspectiveID",
        "CostsDiscountRate",
        "measure",
    ],
    "no_covs": [],
    "cause_tb": ["cause_interaction"],
    "cost_eff_msm": [
        "log_per_year_or_full_int_cost",
        "efficacy",
        "msm_risk_group",
    ],
    "cost_eff_comparator": [
        "log_per_year_or_full_int_cost",
        "efficacy",
        "standard_care_comparator",
    ],
}

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


def sel_covs_priors_fit_pred(args):
    # ========= Retrieve these options from the CLI ===============================

    if args.directory == "malaria":
        pred_dir = pth.HIV_MALARIA
        output_dir = os.path.join(pth.HIV_MALARIA, args.version)

    sens_an_dropped = args.sens_an_dropped
    cwalks_used = args.cwalks_used
    cause_int_combinations = args.cause_intervention
    potential_covs = covariate_options[args.covariates]
    dalys_included_not = [args.dalys_forced]

    log.info(
        f"Running with the cause intervention = {cause_int_combinations},"
        f"sens_an_dropped = {sens_an_dropped}, "
        f"cwalks_used = {cwalks_used}, "
        f"output_dir = {output_dir}, "
        f"covariates_offered = {potential_covs}"
    )

    # ======USING DF INTER W DUMMIES =========
    cause_inter_w_dummies = pd.read_csv(f"{output_dir}/{cause_int_combinations}_w_dummies_wo_outliers_df.csv")

    # ===== setting up cov dictionaries - cwalks and no cwalks =================
    # importing cwalk params
    cwalk_params = pd.read_csv(f"{output_dir}/cwalk_param_summary_to_use.csv")
    cwalk_params = cwalk_params.set_index("covariate")

    # Cwalk dictionary with priors for cause_int combinations where they exist
    df_sub = cwalk_params[cwalk_params["cause_int_comb"] == cause_int_combinations]
    cwalk_covs = df_sub.index.to_list()
    cwalk_priors = df_sub[["beta", "se_beta"]]
    cwalk_param_dict = {w: cwalk_priors.loc[w, ["beta", "se_beta"]].to_numpy() for w in cwalk_covs}

    cwalk_prior_dict = {key: value for key, value in cwalk_param_dict.items()}

    # Define response name - for use in loop below for
    response_name = "log_icer_usd"
    se_name = "log_icer_se"
    study_id_name = "random_effects_id"
    data_id_name = "RatioID"

    # ====== SETTING UP LIST OF POTENTIAL COVARIATES AND CWALK COVS WITH PRIORS ============
    candidate_covs = list(set(potential_covs).difference(set(cwalk_covs)))

    # =====DROPPING RATIOS THAT DON'T MEET OUR 2 ARTICLES, 3 ICER REQUIREMENT =====
    cause_inter_w_dummies = mr_data_cleaning_functions.drops_kw_not_meeting_criteria(cause_inter_w_dummies)

    # want to check if function above has worked
    mr_data_cleaning_functions.test_kw_dropped(cause_inter_w_dummies)

    # ===== CREATING LIST OF KEYWORD DUMMY VARIABLES =============
    # identifying what the keyword reference case is to drop from df
    ref_case = cause_int_ref_keywords[cause_int_combinations]

    # making a list of all keywords still in the dataset
    kw_dummies = list(cause_inter_w_dummies[keywords_input].unique())

    # dropping the ref case for specific cause_int_combination
    # as ref case should not have a covariate for it (that is the intercept)
    kw_dummies = [x for x in kw_dummies if x != ref_case]

    # ==== VERIFY VARIATION IN COVARIATE COLUMNS ================
    # verifying that all the covariates in the candidate covs lists have variation
    # this is likely not needed here but in for extra precaution
    # there has been model runs where candidate covs have included no variation because
    # certain articles / keywords had been dropped removing any variation in that column

    verified_candidate_covs = [
        cov for cov in candidate_covs if mr_data_cleaning_functions.is_non_unique_values_col(cause_inter_w_dummies, cov)
    ]

    log.info(f"verified candidate covs: {verified_candidate_covs}")
    log.info(f"verified keywords: {kw_dummies}")

    # ========================= CHECKING SUMMARY STATISTICS AND EXPORTING DF =================
    # taking mean of the icer and then count number of ratios
    summary_prior_cov_selection = mr_data_cleaning_functions.summarizing_script_outputs(
        cause_inter_w_dummies, index_to_calc=["log_icer_usd", "ratio_count"]
    )
    # always including a date for summary runs
    mr_data_cleaning_functions.exporting_csv(
        summary_prior_cov_selection, f"{output_dir}/{cause_int_combinations}_step_e_sum_prior_cov_selection.csv"
    )

    dalys_included_not = [i for i in dalys_included_not if i is not None]

    # ======== STEP 3: COVARIATE SELECTION =======================

    pkl_path = f"{output_dir}/{cause_int_combinations}_SEL_COVS.pkl"

    if not os.path.exists(pkl_path):
        selected_covs = mr_functions.select_covariates(
            df=cause_inter_w_dummies,
            candidate_covs=verified_candidate_covs,
            include_covs=["intercept", "new_spline_cov"] + dalys_included_not + cwalk_covs + kw_dummies,
            resp_name=response_name,
            se_name=se_name,
            study_id_name=study_id_name,
            # need to confirm this change in tb diagnostic
            # beta_gprior=cwalk_param_dict,
        )

        with open(pkl_path, "wb") as out_file:
            pkl.dump(selected_covs, out_file, protocol=pkl.HIGHEST_PROTOCOL)

    else:
        print("covariates have already been selected.")
        with open(pkl_path, "rb") as in_file:
            selected_covs = pkl.load(in_file)

    # ================================ REMOVING SENS ANALYSIS NOT USED ================
    # This has to be done on a per cause-int basis
    # because it depends on which covs were selected for inclusion in the model
    # selected_covs_dict - that has selected covs
    # dfs - cause_inter_w_dummies_dict

    selected_covs_to_sa_dict = {
        "log_per_year_or_full_int_cost": "log_total_cost_per_cap_usd",
        "DiscountRate": "discountrate",
        "CostDiscountRate": "costsdiscountrate",
        "cost_discount_3_over": "costsdiscountrate",
        "TimeHorizonMagnitude": "timehorizonmagnitude",
        "efficacy": "efficacy",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "payer_or_sector": "ReaderPerspectiveID",
    }

    # only need this if sens analysis are included in the dataframe
    if sens_an_dropped == False:
        list_sens_analysis = list(cause_inter_w_dummies["sens_vars_new"].unique())

        try:
            list_sens_analysis.remove(np.nan)
        except ValueError:
            pass
        for key, value in selected_covs_to_sa_dict.items():
            if key in selected_covs:
                list_sens_analysis = [x for x in list_sens_analysis if x != value]

        list_sens_analysis.extend(["discountrate, costsdiscountrate"])

        # testing when drop all values form df
        log.info(f"df_count = {cause_inter_w_dummies.shape} list of sens to be remove = {list_sens_analysis}")

        # dropping the rows where the sens_vars_new contains an items from the list
        cause_inter_wo_sens = cause_inter_w_dummies[~cause_inter_w_dummies["sens_vars_new"].isin(list_sens_analysis)]
    else:
        cause_inter_wo_sens = cause_inter_w_dummies

    log.info(f"after sens dropped = {cause_inter_wo_sens.shape},")

    # ========= TESTING OF MODELING INPUT ========= =======

    list_covs = ["intercept"] + selected_covs
    cause_inter_wo_sens = cause_inter_wo_sens[cause_inter_wo_sens[list_covs].notnull().all(axis=1)]

    log.info(f"after null values dropped = {cause_inter_wo_sens.shape},")

    # =====DROPPING RATIOS THAT DON'T MEET OUR 2 ARTICLES, 3 ICER REQUIREMENT - FINAL CHECK =====
    cause_inter_wo_sens = mr_data_cleaning_functions.drops_kw_not_meeting_criteria(cause_inter_wo_sens)
    log.info(f"after test to meet 2 articles criteria = {cause_inter_wo_sens.shape},")

    # want to check if function above has worked
    mr_data_cleaning_functions.test_kw_dropped(cause_inter_wo_sens)
    log.info(mr_data_cleaning_functions.test_kw_dropped(cause_inter_wo_sens))

    # ========================= exporting final dfs used for modeling =================
    summary_post_cov_selection = mr_data_cleaning_functions.summarizing_script_outputs(
        cause_inter_wo_sens, index_to_calc=["log_icer_usd", "ratio_count"]
    )

    # exporting data used for model
    mr_data_cleaning_functions.exporting_csv(
        cause_inter_wo_sens, f"{output_dir}/{cause_int_combinations}_df_modeling.csv"
    )
    # always including a date for summary runs
    mr_data_cleaning_functions.exporting_csv(
        summary_post_cov_selection, f"{output_dir}/{cause_int_combinations}_step_e_sum_post_cov_selection.csv"
    )

    # ===== CHECKING OUTPUT FROM STEP D =========
    # creating dataframe with number of rows per cause_int df
    ratio_counts_by_cause_int = mr_data_cleaning_functions.create_df_ratio_counts_per_cause_int(cause_inter_wo_sens)

    # print this as a way to check output from this step
    mr_data_cleaning_functions.exporting_csv(
        ratio_counts_by_cause_int, f"{output_dir}/{cause_int_combinations}_step_e_ratio_counts.csv"
    )

    # ========= Verifying keywords still in dataset after dropping sens analysis, nulls etc ============

    # dropping the ref case for specific cause_int_combination
    # as ref case should not have a covariate for it (that is the intercept)
    update_kw = list(cause_inter_wo_sens[keywords_input].unique())
    log.info(f"first update_kw print: {update_kw}")
    update_kw = [x for x in update_kw if x != ref_case]

    if cause_int_combinations == "syphilis_diagnostic":
        update_kw = []

    log.info(f"selected covs before: {selected_covs}")

    set_to_remove = set(kw_dummies).difference(set(update_kw))
    selected_covs = [x for x in selected_covs if not x in set_to_remove]

    log.info(f"selected covs after: {selected_covs}")
    log.info(f"kw_dummies: {kw_dummies}")
    log.info(f"update_fw: {update_kw}")

    log.info(f"number rows in dataset: {cause_inter_wo_sens.shape}")
    log.info(f"prior dict: {cwalk_prior_dict}")

    # =========STEP 4: GAUSSIAN PRIORS CALC ===========
    # Use cross validation to calculate mses which then inform the sd for the priors wo cwalks

    # setting up the csv path where results will be saved to
    csv_path = f"{output_dir}/{cause_int_combinations}_RESULTS.csv"

    if not os.path.exists(csv_path):
        cv_sds, cv_mses = mr_functions.k_fold_cv_gaussian_prior(
            k=10,
            df=cause_inter_wo_sens,
            resp_name=response_name,
            se_name=se_name,
            study_id_name=study_id_name,
            data_id_name=data_id_name,
            covs=selected_covs,
            beta_gpriors=cwalk_prior_dict,
            initial_upper_prior_sd=1.0,
            num_sds_per_step=5,
        )

        cv_sds = cv_sds[np.argsort(cv_mses)]
        cv_mses = cv_mses[np.argsort(cv_mses)]
        cv_results = pd.DataFrame({"sd": cv_sds, "mse": cv_mses})
        cv_results.to_csv(csv_path, index=False)
    else:
        print("Gaussian priors have already been calculated")
        cv_results = pd.read_csv(csv_path)
        cv_mses = cv_results["mse"].to_numpy()
        cv_sds = cv_results["sd"].to_numpy()

    # ======= CALC STANDARD DEVIATION PRIORS ==============

    prior_sd = cv_results["sd"].to_numpy()[np.argmin(cv_results["mse"].to_numpy())]

    gpriors = {
        v: np.array([0, prior_sd / cause_inter_wo_sens[v].std()])
        for v in selected_covs
        if v not in ["intercept"] + list(cwalk_prior_dict.keys())
    }

    gpriors.update(cwalk_prior_dict)
    gpriors.update({"intercept": [np.array([0, np.inf]), np.array([0, np.inf])]})

    log.info(f"selected covs: {selected_covs}")
    log.info(f"list of gpriors: {gpriors.keys()}")
    log.info(f"gpriors included: {gpriors}")
    log.info(f"list of cwalk prior: {cwalk_prior_dict.keys()}")
    log.info(f"cwalk priors updated during the select covs step {cwalk_prior_dict.keys()}")
    log.info(f"list of cwalk guassian prior: {cwalk_prior_dict.keys()}")

    gpriors_df = pd.DataFrame(gpriors.items(), columns=["Covariate", "prior"])
    mr_data_cleaning_functions.exporting_csv(gpriors_df, f"{output_dir}/{cause_int_combinations}_gpriors.csv")

    # ====== FIT MR MODEL=================================
    pkl_path = f"{output_dir}/{cause_int_combinations}_FINAL_MODEL.pkl"

    if not os.path.exists(pkl_path):
        mr = mr_functions.fit_with_covs(
            df=cause_inter_wo_sens,
            covs=selected_covs,
            resp_name=response_name,
            se_name=se_name,
            study_id_name=study_id_name,
            data_id_name=data_id_name,
            z_covs=["intercept"],
            trim_prop=0.0,
            spline_cov=None,
            gprior_dict=gpriors,
            inner_max_iter=2000,
            outer_max_iter=1000,
        )

        with open(pkl_path, "wb") as out_file:
            dill.dump(mr, out_file)

    else:
        with open(pkl_path, "rb") as in_file:
            mr = dill.load(in_file)

    # saving mr parameters
    np.random.seed(5032198)

    beta_samples_pd = mrtool.core.other_sampling.sample_simple_lme_beta(1000, mr)

    mr_summary = mr_functions.summarize_parameters(mr, "log_GDP_2019usd_per_cap")

    mr_summary[["beta", "beta_se", "beta_variance", "gamma"]] = np.round(
        mr_summary[["beta", "beta_se", "beta_variance", "gamma"]], decimals=4
    )

    # writing to file model parameters
    mr_data_cleaning_functions.exporting_csv(mr_summary, f"{output_dir}/{cause_int_combinations}_model_parameters.csv")

    # ================= Compute Fit Statistics==========================

    cause_inter_wo_sens = cause_inter_wo_sens[
        cause_inter_wo_sens[[v for v in selected_covs + ["log_icer_usd"]]].notnull().all(axis=1)
    ]
    cause_inter_wo_sens = cause_inter_wo_sens[~np.isinf(cause_inter_wo_sens["log_icer_usd"])]

    # fitting mr and calculating r2
    fit_df = mr_functions.create_fit_df(
        mr=mr,
        df=cause_inter_wo_sens,
        resp_name=response_name,
        study_id_name=study_id_name,
        other_id_col_names=[],
        data_id_name=data_id_name,
    )

    r2s = mr_functions.r2(mr, fit_df, response_name)

    csv_path = f"{output_dir}/{cause_int_combinations}_r2s.csv"

    if not os.path.exists(csv_path):
        r2s.to_csv(csv_path, index=True)

    mr_data_cleaning_functions.exporting_csv(fit_df, f"{output_dir}/{cause_int_combinations}_fit_df.csv")

    # =================== STEP 6: PREDICTIONS =============================

    # reimporting signal from spline step d
    pkl_path = f"{output_dir}/{cause_int_combinations}_SIGNAL_MR_2.pkl"
    with open(pkl_path, "rb") as in_file:
        signal_mr = dill.load(in_file)

    # importing predictions df for each of the cause_int combinations
    preds_input = pd.read_csv(f"{pred_dir}/predictions/{cause_int_combinations}_predictions_df.csv")

    log.info(f"number rows in predictions_df: = {preds_input.shape[0]}")
    log.info(f"{pred_dir}/predictions/{cause_int_combinations}_predictions_df.csv")

    preds_input["idx"] = np.arange(preds_input.shape[0])

    # setting up a dictionary, that will hold the prediction results
    preds_output = mr_functions.create_predictions(
        mr,
        signal_mr,
        preds_input,
        response_name,
        se_name,
        selected_covs,
        study_id_name,
        data_id_name,
        beta_samples=beta_samples_pd,
        seed=8721,
    )

    preds_output = preds_output.reset_index()
    preds_output_with_ul = mr_data_cleaning_functions.cleaning_preds(preds_output)

    summary_df = mr_data_cleaning_functions.summarizing_script_outputs(
        preds_output_with_ul, ["predicted_icer_usd", "ratio_ul", "ratio_counts"]
    )

    mr_data_cleaning_functions.exporting_csv(
        preds_output, f"{output_dir}/predictions/{cause_int_combinations}_predictions.csv"
    )
    mr_data_cleaning_functions.exporting_csv(
        summary_df, f"{output_dir}/predictions/{cause_int_combinations}_predictions_summary.csv"
    )

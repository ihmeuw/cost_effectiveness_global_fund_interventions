try:
    import crosswalk_functions
except ModuleNotFoundError:
    import modeling_library.crosswalk_functions as crosswalk_functions

import numpy as np
import pandas as pd
import itertools
import os

df = pd.read_csv("paired_df.csv")

# list of all variables that will have crosswalks
xwalk_cvts = [
    "log_per_year_or_full_int_cost",
    "log_tr_year_cost_diff_int_comp_pc_usd",
    "DiscountRate",
    "CostsDiscountRate",
    "TimeHorizonMagnitude",
    "ltd_or_societal",
    "log_total_cost_per_cap_usd",
    "log_cost_diff_int_comp_pc_usd",
    "log_pr_year_cost_diff_int_comp_pc_usd",
    "coverage",
    "sensitivity",
    "specificity",
    "efficacy",
    "log_discounted_dalys_from_start_per_cap",
    "log_discounted_dalys_from_birth_per_cap",
    "CostsDiscountRate,DiscountRate",
    "CostsDiscountRate,log_discounted_dalys_from_start_per_cap",
]

# creates dummy variables for causes
df = df.join(pd.get_dummies(df["cause"]))
all_causes = df["cause"].unique()
df["diagnostic_ref"] = np.where(df["treatment_ref"] + df["prevention_ref"] == 0, 1, 0)

# creating lists of keywords for the art and hiv_pr wo art (plus reduced hiv pr) categories so we can run them as a group
# in the cwalks
art_keywords = []

for value in df["intervention_keywords_final"]:
    if ("antiretroviral" in value) & (value not in art_keywords):
        art_keywords.append(value)

hiv_df = df[df["cause"] == "hiv/aids"]

hiv_pr_wo_art_kw = []

for i, row in hiv_df.iterrows():
    if (row["diagnostic_treatment_prevention"] == "prevention") & (
        "antiretroviral" not in row["intervention_keywords_final"]
    ):
        hiv_pr_wo_art_kw.append(row["intervention_keywords_final"])
        hiv_pr_wo_art_kw = list(set(hiv_pr_wo_art_kw))

hiv_pr_wo_art_kw.append("prevention of mother to child hiv transmission, antiretroviral therapy for hiv for prevention")

# adding reduced selection of hiv pr
hiv_pr_reduced_kw = ["male circumcision", "post-exposure prophylaxis for hiv", "pre-exposure prophylaxis for hiv"]

hiv_prep_pep = ["post-exposure prophylaxis for hiv", "pre-exposure prophylaxis for hiv"]
prep = ["pre-exposure prophylaxis for hiv"]
pep = ["post-exposure prophylaxis for hiv"]

# adding in a ref column for art and hiv prevention wo art so can get cwalks for these specific sets of icers
df["art_ref"] = np.where(df["intervention_keywords_final"].isin(art_keywords), 1, 0)
df["hiv_pr_wo_art_ref"] = np.where(df["intervention_keywords_final"].isin(hiv_pr_wo_art_kw), 1, 0)
df["hiv_pr_reduced_ref"] = np.where(df["intervention_keywords_final"].isin(hiv_pr_reduced_kw), 1, 0)
df["prep_pep_ref"] = np.where(df["intervention_keywords_final"].isin(hiv_prep_pep), 1, 0)
df["prep_ref"] = np.where(df["intervention_keywords_final"].isin(prep), 1, 0)
df["pep_ref"] = np.where(df["intervention_keywords_final"].isin(pep), 1, 0)


# RANDOM EFFECTS CROSSWALK

param_summaries = []

# Set a random seed for exact reproducibility.
np.random.seed(17835)

for cvt in xwalk_cvts:
    cvt_param_summaries = []
    if "," in cvt:
        cvt_list = cvt.split(",")
    else:
        cvt_list = [cvt]

    # Create difference variables for the covariate in question and the ICER.
    cvt_df = crosswalk_functions.create_diff_variables(
        df[df["sens_variable"] == cvt].copy(), "log_icer_usd", "log_icer_se", cvt_list, ("_sens", "_ref")
    )

    # Drop rows where either ratio has an NA ICER (e.g. other quadrants) or covariate value.
    cvt_df = cvt_df[
        cvt_df[[i + "_diff" for i in ["log_icer_usd", "log_icer_se"] + cvt_list]].notnull().all(axis=1)
    ].copy()

    if cvt == "log_pr_year_cost_diff_int_comp_pc_usd":
        intervention_types = ["prevention"]
    elif cvt == "log_tr_year_cost_diff_int_comp_pc_usd":
        intervention_types = ["treatment"]
    else:
        intervention_types = [
            "diagnostic",
            "treatment",
            "prevention",
            "combined",
            "art",
            "hiv_pr_wo_art",
            "hiv_pr_reduced",
            "prep_pep",
            "prep",
            "pep",
        ]

    for dtp in intervention_types:
        if dtp == "combined":
            dtp_df = cvt_df
        else:
            dtp_df = cvt_df[cvt_df[f"{dtp}_ref"] == 1]

        # only fitting models to intervention types where we have at least 4 sens_ref pairs
        # but if a model is fit with only 4 sens-ref pairs we won't use these results
        if dtp_df.shape[0] > 3:

            # this filters out causes less than one because then don't need random effects
            # wo random effects models will be run later in the code
            if dtp_df["cause"].nunique() > 1:
                if len(cvt_list) == 1:
                    xw_cause_randef = crosswalk_functions.cwalk(
                        dtp_df, "log_icer_usd", "log_icer_se", cvt, study_id="cause", random_effect=True
                    )

                    smry_cause_randef = crosswalk_functions.summarize_cwalk(
                        xw_cause_randef, cvt, n_samples=1000, seed=None
                    )[1]
                    smry_cause_randef["joint_model"] = cvt
                else:
                    xw_cause_randef = crosswalk_functions.cwalk_multivar(
                        dtp_df, "log_icer_usd", "log_icer_se", cvt_list, study_id="cause", random_effect=True
                    )
                    smry_cause_randef = crosswalk_functions.summarize_cwalk(
                        xw_cause_randef, cvt_list, n_samples=1000, seed=None
                    )[1]
                    smry_cause_randef["joint_model"] = "joint_" + cvt

                if (xw_cause_randef.gamma_soln != 0).any():
                    # when combine se of beta with gamma, need to add the variances and not the se. i.e. np.sqrt(sqr se_beta + gamma)
                    smry_cause_randef = smry_cause_randef.assign(
                        **{
                            f"se_beta_{i}": np.sqrt(smry_cause_randef["se_beta"] ** 2 + smry_cause_randef["gamma"])
                            for i in all_causes
                        }
                    )
                else:
                    smry_cause_randef["gamma"] = 0

                # create columns - metadata about what model was run
                smry_cause_randef["intervention_type"] = dtp
                smry_cause_randef["n_articles"] = dtp_df["PubMedID"].nunique()
                smry_cause_randef["cause_model"] = "random effects"

                if "covariate" in smry_cause_randef.columns:
                    smry_cause_randef = smry_cause_randef.drop("covariate", axis=1)

                cause_beta_cols = smry_cause_randef.filter(regex="beta_").columns

                smry_cause_randef = smry_cause_randef.assign(
                    **{f"total_{i}": smry_cause_randef["beta"] + smry_cause_randef[i] for i in cause_beta_cols}
                )

                # adding the summary for single covariate and single type to list of summaries for that covariate
                cvt_param_summaries.append(smry_cause_randef)

    if len(cvt_param_summaries) > 0:
        cvt_param_summary_df = pd.concat(cvt_param_summaries, axis=0)
        # list of df covariate summaries, contains all covariate
        # later on this will become a df
        param_summaries.append(cvt_param_summary_df)


# NON RANDOM EFFECTS CROSSWALKS
# separate models for separate causes

single_cause_summaries = []

for cvt, dtp, cause in itertools.product(
    xwalk_cvts,
    [
        "diagnostic",
        "treatment",
        "prevention",
        "combined",
        "art",
        "hiv_pr_wo_art",
        "hiv_pr_reduced",
        "prep_pep",
        "prep",
        "pep",
    ],
    all_causes,
):
    # checks that not trying to run model for log_pr but for non-prevention dataset
    if cvt == "log_pr_year_cost_diff_int_comp_pc_usd" and dtp != "prevention":
        continue
    if cvt == "log_tr_year_cost_diff_int_comp_pc_usd" and dtp != "treatment":
        continue
    if "," in cvt:
        cvt_list = cvt.split(",")
    else:
        cvt_list = [cvt]
    n_covs = len(cvt_list)

    msk = df["cause"] == cause
    if dtp != "combined":
        msk = msk & (df[f"{dtp}_ref"] == 1)
    if cvt in ["log_pr_year_cost_diff_int_comp_pc_usd", "log_cost_diff_int_comp_pc_usd"]:
        msk = msk & (df["sens_variable"] == "log_total_cost_per_cap_usd")
    else:
        msk = msk & (df["sens_variable"] == cvt)

    # Create difference variables for the covariate, intervention type, and cause in question and the ICER.
    sub_d = crosswalk_functions.create_diff_variables(
        df[msk].copy(), "log_icer_usd", "log_icer_se", cvt_list, ("_sens", "_ref")
    )
    # Drop rows where either ratio has an NA ICER (e.g. other quadrants) or covariate value.
    sub_df = sub_df[
        sub_df[[i + "_diff" for i in ["log_icer_usd", "log_icer_se"] + cvt_list]].notnull().all(axis=1)
    ].copy()
    idx = pd.MultiIndex.from_product([[dtp], [cause], cvt_list], names=("intervention_type", "cause", "covariate"))
    if sub_df.shape[0] > 2:
        if n_covs == 1:
            xw = crosswalk_functions.cwalk(sub_df, "log_icer_usd", "log_icer_se", cvt, "ArticleID")
            smry = crosswalk_functions.summarize_cwalk(xw, cvt, n_samples=1000)[1]
        else:
            if (sub_df[[f"{c}_diff" for c in cvt_list]].nunique() <= 1).any():
                smry = smry = pd.DataFrame(
                    {
                        "beta": np.nan,
                        "se_beta": np.nan,
                        "sample_size": sub_df.shape[0],
                        "n_articles": sub_df["PubMedID"].nunique(),
                    },
                    index=idx,
                )
            xw = crosswalk_functions.cwalk_multivar(sub_df, "log_icer_usd", "log_icer_se", cvt_list, "ArticleID")
            smry = crosswalk_functions.summarize_cwalk(xw, cvt_list, n_samples=1000)[1]
        smry.index = idx
    else:
        smry = pd.DataFrame(
            {
                "beta": np.nan,
                "se_beta": np.nan,
                "sample_size": sub_df.shape[0],
                "n_articles": sub_df["PubMedID"].nunique(),
            },
            index=idx,
        )

    print(smry)

    smry["n_articles"] = sub_df["PubMedID"].nunique()

    smry["joint_model"] = cvt if n_covs == 1 else f"joint_{cvt}"
    single_cause_summaries.append(smry)


sc_summary_df = pd.concat(single_cause_summaries, axis=0)
sc_summary_df.columns = sc_summary_df.columns.str.replace("multidrug.*tubercul.*stance", "MDR-TB", regex=True)

sc_summary_df = sc_summary_df[["joint_model", "n_articles", "sample_size", "beta", "se_beta"]].reset_index()
sc_summary_df = sc_summary_df.pivot(
    columns=["cause"],
    values=["beta", "se_beta", "sample_size", "n_articles"],
    index=["intervention_type", "joint_model", "covariate"],
)
sc_summary_df.columns = ["_".join(col) for col in sc_summary_df.columns.values]
sc_summary_df = sc_summary_df.reset_index()

sc_summary_df["cause_model"] = "separate models"
sc_summary_df = sc_summary_df.assign(
    **{f"total_{i}": sc_summary_df[i] for i in sc_summary_df.filter(regex="beta_").columns}
)
sc_summary_df["n_articles"] = sc_summary_df.filter(regex="n_articles").fillna(0).sum(axis=1)
sc_summary_df["sample_size"] = sc_summary_df.filter(regex="sample_size").fillna(0).sum(axis=1)

re_summary_df = pd.concat(param_summaries, axis=0)
re_summary_df = re_summary_df[
    ["intervention_type", "joint_model", "cause_model", "n_articles", "sample_size", "beta", "se_beta", "gamma"]
    + list(re_summary_df.filter(regex="beta_").columns)
]
re_summary_df = re_summary_df.reset_index().rename({"index": "covariate"}, axis=1)

param_summary_df = pd.concat([sc_summary_df, re_summary_df], axis=0, ignore_index=True)
param_summary_df.columns = param_summary_df.columns.str.replace("multidrug.*tubercul.*stance", "MDR-TB", regex=True)

column_order = (
    [
        "intervention_type",
        "joint_model",
        "covariate",
        "cause_model",
        "n_articles",
        "sample_size",
        "beta",
        "se_beta",
        "gamma",
    ]
    + list(param_summary_df.columns[param_summary_df.columns.str.contains("^beta_", regex=True)])
    + list(param_summary_df.columns[param_summary_df.columns.str.contains("se_beta_", regex=True)])
    + list(param_summary_df.columns[param_summary_df.columns.str.contains("^n_articles_", regex=True)])
    + list(param_summary_df.columns[param_summary_df.columns.str.contains("^sample_size_", regex=True)])
    + list(param_summary_df.columns[param_summary_df.columns.str.contains("^total_beta_", regex=True)])
)
param_summary_df = param_summary_df[column_order].sort_values(
    ["joint_model", "covariate", "intervention_type", "cause_model"]
)

output_path = "crosswalk_param_summaries.csv"

if not os.path.exists(output_path):
    param_summary_df.to_csv(output_path, index=False)
    print("Wrote output to " + output_path)

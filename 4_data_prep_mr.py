import numpy as np
import pandas as pd
import os
import pickle as pkl
import logging

from datetime import datetime

log = logging.getLogger()

try:
    import mr_data_cleaning_functions
except ModuleNotFoundError:
    import modeling_library.mr_data_cleaning_functions as mr_data_cleaning_functions


### ======IMPORTING DATA AND PRELIM DATA CLEANING ======

# specify if sens analysis are being used or not
sens_an_dropped = False

# Read in dataset
df = pd.read_csv("CLEANED_DF.csv")

# making all strings lowercase
cols = df.select_dtypes(object).columns
df[cols] = df[cols].apply(lambda x: x.str.lower())

# need to update the cause name of hiv/aids to just hiv
cause_cols = ["all_causes_per_ratio", "cause"]

for col in cause_cols:
    df[col] = df[col].str.replace("/", "_").str.replace(" - ", "_").str.replace(" ", "_")

# ====== DEFINE VARIABLE NAMES AND REMOVING NULLS INFINITE AND ZEROS =======

# define random effect id
df["random_effects_id"] = np.where(df["Random_Effects_ID"].isnull(), df["ArticleID"], df["Random_Effects_ID"])

# Define response name
response_name = "log_icer_usd"
se_name = "log_icer_se"
spline_cov = "log_GDP_2019usd_per_cap"
study_id_name = "random_effects_id"
data_id_name = "RatioID"

# Read in crosswalk covariates
# need to make sure that this aligns with the 'cwalk_covs' - defined above
cwalk_params = pd.read_csv("cwalk_param_summary_to_use.csv")

potential_cwalk_covs = list(set(cwalk_params["covariate"]))

# Filter out any missing observations or costs/icers that are infinite
df = df[
    (df["log_icer_usd"].notnull())
    & (~np.isinf(df["log_icer_usd"]))
    & (~np.isinf(df["log_per_year_or_full_int_cost"]))
    & (df["log_GDP_2019usd_per_cap"].notnull())
    & (df["log_dalys_per_cap"].notnull())
]

# identify unique RatioIDs/check for duplicates
if df["RatioID"].nunique() != df.shape[0]:
    raise ValueError("There are non unique RatioIDs")

df = df[df["flag_to_drop"] != 1]

# Make sure to drop all articles required
# This is a triple check
df = df[~df["intervention_keywords_final"].isin(["hiv vaccines", "vaginal microbicide gel"])]

# ==============DROPPING SENS ANALYSIS =====================
# Dropping sens analysis of covariates that won't be used
sens_vars_to_drop = [
    "prevalence",
    "adherence",
    "-0.2445054945054945",
    "coverage",
    "cd4_count",
    "cd4_cell_count",
    "sex_id",
    "tr_cost_comparator",
]

if sens_an_dropped == True:
    df = df[df["sens_vars_new"].isnull()]
else:
    df = df[~df["sens_vars_new"].isin(sens_vars_to_drop)]


# ======== ADDING IN COLUMNS FOR POTENTIAL COVS  ===========================
# add potential covariate columns
df["cost_discount_0"] = 1
df.loc[df["CostsDiscountRate"] != 0, "cost_discount_0"] = 0

df["discount_0"] = 1
df.loc[df["DiscountRate"] != 0, "discount_0"] = 0

df["not_dalys"] = 1
df.loc[df["measure"] == "dalys", "not_dalys"] = 0

# =====DROPPING RATIOS THAT DON'T MEET OUR 2 ARTICLES, 3 ICER REQUIREMENT =====
df = mr_data_cleaning_functions.drops_kw_not_meeting_criteria(df)

# want to check if function above has worked
mr_data_cleaning_functions.test_kw_dropped(df)

# creates a dataframe with counts to look at data
ratio_counts_by_cause_int = (
    df.groupby("intervention_keywords_final")[["ArticleID", "RatioID"]].agg("nunique").reset_index()
)

# ============== REPLACING TH ================
# update the time horizon value to be 75 years because it is more realistic
df1 = df.copy()
df1.loc[df1["TimeHorizonMagnitude"] == 100, "TimeHorizonMagnitude"] = 75

# ====== MAKING CAUSE AND INTERVENTION PAIRS ======
# making a dictionary of cause_int df
m_causes = ["malaria"]
hiv_causes = ["hiv_aids"]
syphilis_causes = ["syphilis"]
tb_causes = [
    "multidrug-resistant_tuberculosis_without_extensive_drug_resistance",
    "hiv_aids_drug-susceptible_tuberculosis",
    "tuberculosis",
]

# setting up cause-intervention dict, lists
causes_list = [hiv_causes, tb_causes, m_causes, syphilis_causes]
intervention_list = ["treatment", "prevention"]

# dictionary of dataframes
cause_inter_dict = dict()

for cause in causes_list:
    sub_cause = df1[df1["cause"].isin(cause)]
    string_cause = cause[-1]

    for inter in intervention_list:
        cause_inter = sub_cause[sub_cause["diagnostic_treatment_prevention"] == inter]
        key_name = string_cause + "_" + inter
        cause_inter_dict[key_name] = cause_inter

# need to add tb diagnostic and all art interventions
tb_int = df1[df1["cause"].isin(tb_causes)]
cause_inter_dict["tuberculosis_diagnostic"] = tb_int[tb_int["diagnostic_treatment_prevention"] == "diagnostic"]

del cause_inter_dict["syphilis_treatment"]

art = df1[df1.intervention_keywords_final.str.contains("antiretroviral")]

criteria_causes = [
    "antiretroviral therapy for hiv, immunological antiretroviral treatment initiation criteria",
    "antiretroviral therapy for hiv, expanded eligibility for antiretroviral treatment",
]

# Creating dummy variable for ratios where comparator id == 2
art["standard_care_comparator"] = 0
art.loc[art["ComparatorID"] == 2, "standard_care_comparator"] = 1
art.loc[art["intervention_keywords_original"].isin(criteria_causes), "standard_care_comparator"] = 1

cause_inter_dict["hiv_aids_art"] = art

# HIV PREVENTION
kw_drop_list_hiv_pr = [
    "antiretroviral therapy for hiv for prevention, hiv testing",
    "antiretroviral therapy for hiv for prevention",
    "antiretroviral therapy for hiv for prevention, immunological antiretroviral treatment initiation criteria",
    "antiretroviral therapy for hiv for prevention, methadone maintenance therapy",
    "pooled hiv testing, antiretroviral therapy for hiv for prevention",
    # additional keywords were dropped to get ensure definition consistency within the dataset
    "prevention of mother to child hiv transmission, antiretroviral therapy for hiv for prevention",
    "education to prevent hiv",
    "condom promotion",
    "voluntary hiv testing and counseling",
    "treatment of sexually transmitted infections",
    "hiv testing",
    "prevention of mother to child hiv transmission",
]

# creating hiv pr df
cause_inter_dict["hiv_pr_wo_art"] = cause_inter_dict["hiv_aids_prevention"][
    ~cause_inter_dict["hiv_aids_prevention"]["intervention_keywords_final"].isin(kw_drop_list_hiv_pr)
]

# creating prep df
cause_inter_dict["prep"] = cause_inter_dict["hiv_aids_prevention"][
    cause_inter_dict["hiv_aids_prevention"]["intervention_keywords_final"].str.contains(
        "pre-exposure prophylaxis for hiv"
    )
]

cause_inter_dict["prep"]["msm_risk_group"] = 0
cause_inter_dict["prep"].loc[cause_inter_dict["prep"]["risk_group_code"] == "msm", "msm_risk_group"] = 1


# creating pep df
cause_inter_dict["pep"] = cause_inter_dict["hiv_aids_prevention"][
    cause_inter_dict["hiv_aids_prevention"]["intervention_keywords_final"].str.contains(
        "post-exposure prophylaxis for hiv"
    )
]

# creating dummy variable for msm risk group or not
risk_group_pep = list(cause_inter_dict["pep"]["risk_group_keyword"].unique())
msm_group_pep = risk_group_pep.copy()
msm_group_pep = [s for s in msm_group_pep if "homosexuals" in s]

cause_inter_dict["pep"]["msm_risk_group"] = 0
cause_inter_dict["pep"].loc[cause_inter_dict["pep"]["risk_group_keyword"].isin(msm_group_pep), "msm_risk_group"] = 1

# creating df for prep and pep together
prep_pep = ["pre-exposure prophylaxis for hiv", "post-exposure prophylaxis for hiv"]

prep_pep_df = pd.concat([cause_inter_dict["prep"], cause_inter_dict["pep"]])
cause_inter_dict["prep_pep"] = prep_pep_df


# used for an experiment modeling just antibiotics for tuberculosis
cause_inter_dict["tuberculosis_treatment"] = cause_inter_dict["tuberculosis_treatment"][
    cause_inter_dict["tuberculosis_treatment"]["intervention_keywords_final"] == "antibiotics for tuberculosis"
]

# ===== CHECKING OUTPUT FROM STEP C =========
# creating dataframe with number of rows per cause_int df
rows_per_cause_int = mr_data_cleaning_functions.create_df_ratio_counts_per_cause_int(cause_inter_dict)

date = datetime.now().strftime("%Y_%m_%d-%H.%M.%S")

mr_data_cleaning_functions.exporting_csv(rows_per_cause_int, "step_c_ratio_counts.csv")

# ====== WRITING OUTPUT TO SETS OF DF =====
for key in cause_inter_dict.keys():
    mr_data_cleaning_functions.exporting_csv(cause_inter_dict[key], f"{key}_w_outliers.csv")

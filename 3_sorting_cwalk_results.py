# import ce team functions and paths
try:
    from path_provider import get_path_provider
except ModuleNotFoundError:
    from utilities.path_provider import get_path_provider

try:
    import mr_data_cleaning_functions
except ModuleNotFoundError:
    import modeling_library.mr_data_cleaning_functions as mr_data_cleaning_functions

# Retrieve and reset the dynamic path provider
pth = get_path_provider()

# use this if you want to reload an updated path files
pth.reset()

#
### end of filepath requirements
#

import numpy as np
import pandas as pd
import os


output_path = "crosswalk_param_summaries.csv"
param_summary_df = pd.read_csv(output_path)

## ============ ADD CODE TO SORT CWALKS TO USE ===========
sep_models = param_summary_df[param_summary_df["cause_model"] == "separate models"]
sep_models = sep_models[sep_models["intervention_type"] != "combined"]

sep_models["meets_ratio_count"] = "no"
sep_models["meets_article_count"] = "no"

# subset the cwalks by cause to make it easier to comprehend
causes = ["HIV/AIDS - Drug-susceptible Tuberculosis", "hiv/aids", "malaria", "MDR-TB", "syphilis", "tuberculosis"]

cols_needed = ["intervention_type", "joint_model", "covariate", "cause_model"]

# making a dict of cols by cause
cols_dict = dict()
for cause in causes:
    cols_dict[cause] = list(sep_models.columns[sep_models.columns.str.contains(cause)])

cwalks_cause_dict = dict()
for key in cols_dict.keys():
    cols = cols_needed + cols_dict[key]
    df = sep_models[cols]
    cwalks_cause_dict[key] = df


def meets_article_req(df):
    if df[f"n_articles_{key}"] > 1:
        return "yes"


def meets_ratio_req(df):
    if df[f"sample_size_{key}"] > 2:
        return "yes"


cwalk_cause_to_use = dict()

for key in cwalks_cause_dict.keys():
    cwalks_cause_dict[key]["meets_article_count"] = cwalks_cause_dict[key].apply(meets_article_req, axis=1)
    cwalks_cause_dict[key]["meets_ratio_count"] = cwalks_cause_dict[key].apply(meets_ratio_req, axis=1)
    cwalk_cause_to_use[key] = cwalks_cause_dict[key][
        (cwalks_cause_dict[key]["meets_article_count"] == "yes")
        & (cwalks_cause_dict[key]["meets_ratio_count"] == "yes")
    ]

causes = ["HIV/AIDS - Drug-susceptible Tuberculosis", "hiv/aids", "malaria", "MDR-TB", "syphilis", "tuberculosis"]
file_names = ["hiv_dr_tb", "hiv", "malaria", "mdr_tb", "syphilis", "tuberculosis"]

for i, key in enumerate(causes):
    output_path = f"cwalk/{file_names[i]}_cwalks_to_use.csv"
    if not os.path.exists(output_path):
        cwalk_cause_to_use[key].to_csv(output_path, index=False)
        print("Wrote output to " + output_path)

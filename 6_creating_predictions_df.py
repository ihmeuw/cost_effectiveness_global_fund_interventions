import numpy as np
import pandas as pd
import os
import warnings
from db_queries import get_population, get_outputs, get_location_metadata, get_covariate_estimates, get_age_metadata

try:
    import mr_data_cleaning_functions
except ModuleNotFoundError:
    import modeling_library.mr_data_cleaning_functions as mr_data_cleaning_functions

#
### end of filepath requirements
#

# cause_int combinations
cause_int_combinations = [
    "hiv_aids_art",
    "tuberculosis_prevention",
    "syphilis_diagnostic",
    "hiv_pr_wo_art",
    "tuberculosis_diagnostic",
    "malaria_prevention",
    "malaria_treatment",
    "tuberculosis_treatment",
]

# need a csv for cost data
costs_df = pd.read_csv("summary_keywords_cost_efficacy_230206.csv")
costs_df = costs_df[costs_df["all_causes_per_ratio"].notnull()]

# csv for global fund eligible countries to know which countries we are going to predict for
gf_countries = pd.read_csv("eligibility-by-location-Locations.csv")
gavi_countries = pd.read_csv("gavi_eligible_df.csv")
df = pd.read_csv("cleaned_df.csv")

# cleaning up costs df to split up by cause-int
# make all strings lowercase
costs_cols = ["all_causes_per_ratio", "intervention_type", "intervention_keywords_final"]

for col in costs_cols:
    costs_df[col] = costs_df[col].str.lower()

cause_list = list(costs_df["all_causes_per_ratio"].unique())

for col in gf_countries.columns:
    gf_countries[col] = gf_countries[col].str.lower()

gavi_countries["country"] = gavi_countries["country"].str.lower()

# =========== LOCATIONS TO PREDICT FOR ===========================
# setting eligible country lists by cause
# fixing up the names of gf countries to match gbd names

# getting location data
# this informs other get outputs functions
lmtdta = get_location_metadata(location_set_id=35, gbd_round_id=6)

locations_superregions = pd.read_csv("location_ids_1_6.csv")
lmtdta_to_merge = locations_superregions.loc[locations_superregions["level"] == 3, ["location_id", "location_name"]]
lmtdta_to_merge["location_name"] = lmtdta_to_merge["location_name"].str.lower()

# merging on loction ids to global fund countries
gf_countries = pd.merge(
    left=gf_countries,
    right=lmtdta_to_merge,
    how="left",
    left_on="Location",
    right_on="location_name",
)

# remove all cause-country combinations where they are 'not eligible'
# now using predicting for all countries
gf_countries = gf_countries[gf_countries["location_id"].notnull()]

# make a dict that contains of lists for each cause of GF eligible countries
eligible_countries_dict = dict()

for cause in cause_list:
    eligible_countries_dict[cause] = gf_countries[gf_countries["Component"] == cause]
    eligible_countries_dict[cause] = eligible_countries_dict[cause].drop(columns={"Component", "Location"})

eligible_countries_dict["syphilis"] = eligible_countries_dict["hiv_aids"]

# making a list of all countries that we want to predict at least one cause for
locns = list(gf_countries["location_id"].unique())

# ===========GETTING CAUSES NEEDED ====================
causes_list = [297, 345, 298, 394, 948, 949, 950]
causes_df = pd.DataFrame(columns=["cause"], index=causes_list)
causes_df.loc[297, "cause"] = "tuberculosis"
causes_df.loc[345, "cause"] = "malaria"
causes_df.loc[298, "cause"] = "hiv_aids"
causes_df.loc[394, "cause"] = "syphilis"
causes_df.loc[948, "cause"] = "hiv_aids_dr_tb"
causes_df.loc[949, "cause"] = "hiv_aids_mdr_tb"
causes_df.loc[950, "cause"] = "hiv_aids_xdr_tb"

# ===================ACCESSING SHARED FUNCTIONS =================
# get dalys information across all countries
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

age_groups_needed = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 28, 49, 50]
age_group_ids_2019 = get_age_metadata(gbd_round_id=6)


def get_outputs_fuction(topic, cause_id, locns=None, rei_id=None, metric_id=[1, 3], sex=[3], year_id=2019):
    output_dalys = get_outputs(
        topic,
        cause_id=cause_id,
        age_group_id=age_groups_needed,
        year_id=year_id,
        measure_id=[2, 5, 6],
        metric_id=metric_id,
        location_id=locns,
        sex_id=sex,
        gbd_round_id=6,
        decomp_step="step5",
        rei_id=rei_id,
    )
    return output_dalys


def pull_split_dalys(list_causes, locations, year, sex_list):
    dalys_pull = get_outputs_fuction("cause", cause_id=list_causes, locns=locations, year_id=year, sex=sex_list)
    dalys_pull["age_group_id"] = dalys_pull["age_group_id"].apply(str)
    dalys_per_capita = dalys_pull[dalys_pull["metric_id"] == 3]
    dalys_total = dalys_pull[dalys_pull["metric_id"] == 1]
    return dalys_per_capita, dalys_total


# pulling dalys and prevalence for 2019 - to use in analysis
# pulling dalys and prevalence for 2010 - so can calculate growth rate
dalys_pc_2019, dalys_t_2019 = pull_split_dalys(causes_list, locns, 2019, sex_list=3)
dalys_pc_2010, dalys_t_2010 = pull_split_dalys(causes_list, locns, 2010, sex_list=3)

# pulling female dalys for pregnant women for HIV only
# trialling potential calculation of icers for PMTCT
dalys_pc_2019_female, dalys_t_2019_female = pull_split_dalys(298, locns, 2019, sex_list=2)
dalys_pc_2010_female, dalys_t_2010_female = pull_split_dalys(298, locns, 2010, sex_list=2)

# pulling male dalys for prep
dalys_pc_2019_male, dalys_t_2019_male = pull_split_dalys(298, locns, 2019, sex_list=1)
dalys_pc_2010_male, dalys_t_2010_male = pull_split_dalys(298, locns, 2010, sex_list=1)


def reshape_dalys_pull(dalys_pull):
    """
    Going to reshape the df to get the numbers in the most useful layout
    columns for child (1+6), columns for adults (7+24+25+26), columns for all age burden (22)
    columns for fertile women (7+8+9+10+11+12+13+14)
    """
    dalys_pull = dalys_pull.pivot(
        index=["cause_id", "location_id"], columns=["age_group_id", "measure_id"], values=["val", "upper", "lower"]
    ).reset_index()

    dalys_pull.columns = dalys_pull.columns.to_flat_index()

    dalys_pull["burden_under_10"] = dalys_pull[("val", "1", 2)] + dalys_pull[("val", "6", 2)]
    dalys_pull["burden_under_5"] = dalys_pull[("val", "1", 2)]
    dalys_pull["burden_under_1"] = dalys_pull[("val", "28", 2)]

    dalys_pull["burden_10_49"] = (
        dalys_pull[("val", "7", 2)]
        + dalys_pull[("val", "8", 2)]
        + dalys_pull[("val", "9", 2)]
        + dalys_pull[("val", "10", 2)]
        + dalys_pull[("val", "11", 2)]
        + dalys_pull[("val", "12", 2)]
        + dalys_pull[("val", "13", 2)]
        + dalys_pull[("val", "14", 2)]
    )

    dalys_pull["burden_adult"] = (
        dalys_pull[("val", "24", 2)]
        + dalys_pull[("val", "25", 2)]
        + dalys_pull[("val", "26", 2)]
        + dalys_pull[("val", "7", 2)]
    )
    dalys_pull["prevalence_under_10"] = dalys_pull[("val", "1", 5)] + dalys_pull[("val", "6", 5)]

    dalys_pull["prevalence_adult"] = (
        dalys_pull[("val", "24", 5)]
        + dalys_pull[("val", "25", 5)]
        + dalys_pull[("val", "26", 5)]
        + dalys_pull[("val", "7", 5)]
    )
    dalys_pull["prevalence_under_5"] = dalys_pull[("val", "1", 5)]
    dalys_pull["prevalence_under_1"] = dalys_pull[("val", "28", 5)]
    dalys_pull["prevalence_10_49"] = (
        dalys_pull[("val", "7", 5)]
        + dalys_pull[("val", "8", 5)]
        + dalys_pull[("val", "9", 5)]
        + dalys_pull[("val", "10", 5)]
        + dalys_pull[("val", "11", 5)]
        + dalys_pull[("val", "12", 5)]
        + dalys_pull[("val", "13", 5)]
        + dalys_pull[("val", "14", 5)]
    )
    dalys_pull["burden_all"] = dalys_pull[("val", "22", 2)]
    dalys_pull["prevalence_all"] = dalys_pull[("val", "22", 5)]
    dalys_pull = dalys_pull.rename(columns={("cause_id", "", ""): "cause_id", ("location_id", "", ""): "location_id"})

    return dalys_pull


# collating age groups together
dalys_pc_19_reshaped = reshape_dalys_pull(dalys_pc_2019)
dalys_total_19_reshaped = reshape_dalys_pull(dalys_t_2019)
dalys_pc_10_reshaped = reshape_dalys_pull(dalys_pc_2010)
dalys_total_10_reshaped = reshape_dalys_pull(dalys_t_2010)
dalys_total_19_reshaped_female = reshape_dalys_pull(dalys_t_2019_female)
dalys_pc_19_reshaped_female = reshape_dalys_pull(dalys_pc_2019_female)
dalys_total_10_reshaped_female = reshape_dalys_pull(dalys_t_2010_female)
dalys_pc_10_reshaped_female = reshape_dalys_pull(dalys_pc_2010_female)
dalys_total_19_reshaped_male = reshape_dalys_pull(dalys_t_2019_male)
dalys_pc_19_reshaped_male = reshape_dalys_pull(dalys_pc_2019_male)
dalys_total_10_reshaped_male = reshape_dalys_pull(dalys_t_2010_male)
dalys_pc_10_reshaped_male = reshape_dalys_pull(dalys_pc_2010_male)


def melt_dalys(dalys_pull):
    """
    Remove columns not needed
    Want to melt back into shape so have burden values of all ages, child, adults all stacked in one column
    """
    dalys_pull = dalys_pull[
        [
            "cause_id",
            "location_id",
            "burden_all",
            "burden_under_10",
            "burden_adult",
            "burden_under_5",
            "burden_under_1",
            "burden_10_49",
            "prevalence_all",
            "prevalence_under_10",
            "prevalence_adult",
            "prevalence_under_5",
            "prevalence_under_1",
            "prevalence_10_49",
        ]
    ]

    dalys_pull = dalys_pull.melt(
        id_vars=["cause_id", "location_id"],
        value_vars=[
            "burden_all",
            "burden_under_10",
            "burden_adult",
            "burden_under_5",
            "burden_under_1",
            "burden_10_49",
            "prevalence_all",
            "prevalence_under_10",
            "prevalence_adult",
            "prevalence_under_5",
            "prevalence_under_1",
            "prevalence_10_49",
        ],
    )
    return dalys_pull


dalys_pc_19 = melt_dalys(dalys_pc_19_reshaped)
dalys_total_19 = melt_dalys(dalys_total_19_reshaped)
dalys_pc_10 = melt_dalys(dalys_pc_10_reshaped)
dalys_total_10 = melt_dalys(dalys_total_10_reshaped)
dalys_pc_19_female = melt_dalys(dalys_pc_19_reshaped_female)
dalys_total_19_female = melt_dalys(dalys_total_19_reshaped_female)
dalys_pc_10_female = melt_dalys(dalys_pc_10_reshaped_female)
dalys_total_10_female = melt_dalys(dalys_total_10_reshaped_female)
dalys_pc_19_male = melt_dalys(dalys_pc_19_reshaped_male)
dalys_total_19_male = melt_dalys(dalys_total_19_reshaped_male)
dalys_pc_10_male = melt_dalys(dalys_pc_10_reshaped_male)
dalys_total_10_male = melt_dalys(dalys_total_10_reshaped_male)


def merging_total_per_capita_dalys(dalys_pc, dalys_total):
    """
    Merges together dalys per capita and total dalys for each location and age group.
    """
    dalys_combined = pd.merge(
        left=dalys_pc,
        right=dalys_total,
        how="outer",
        left_on=["cause_id", "location_id", "variable"],
        right_on=["cause_id", "location_id", "variable"],
    )
    dalys_combined = dalys_combined.rename(columns={"value_x": "burden_variable", "value_y": "burden_total"})
    return dalys_combined


dalys_19 = merging_total_per_capita_dalys(dalys_pc_19, dalys_total_19)
dalys_10 = merging_total_per_capita_dalys(dalys_pc_10, dalys_total_10)
dalys_19_female = merging_total_per_capita_dalys(dalys_pc_19_female, dalys_total_19_female)
dalys_10_female = merging_total_per_capita_dalys(dalys_pc_10_female, dalys_total_10_female)
dalys_19_male = merging_total_per_capita_dalys(dalys_pc_19_male, dalys_total_19_male)
dalys_10_male = merging_total_per_capita_dalys(dalys_pc_10_male, dalys_total_10_male)


# merging together female dalys for age_10_49 fertile women
# need to drop these age groups for other datasets
variables_female = [
    "prevalence_10_49",
    "burden_10_49",
]

variables_male = [
    "burden_adult",
    "prevalence_adult",
]

variables_both = [
    "burden_all",
    "burden_under_10",
    "burden_adult",
    "burden_under_5",
    "burden_under_1",
    "prevalence_all",
    "prevalence_under_10",
    "prevalence_adult",
    "prevalence_under_5",
    "prevalence_under_1",
]

dalys_19 = dalys_19[dalys_19["variable"].isin(variables_both)]
dalys_10 = dalys_10[dalys_10["variable"].isin(variables_both)]
dalys_19_female = dalys_19_female[dalys_19_female["variable"].isin(variables_female)]
dalys_10_female = dalys_10_female[dalys_10_female["variable"].isin(variables_female)]
dalys_19_male = dalys_19_male[dalys_19_male["variable"].isin(variables_male)]
dalys_10_male = dalys_10_male[dalys_10_male["variable"].isin(variables_male)]

# need to update variable to be just men for just men
dalys_19_male["variable"] = dalys_19_male["variable"].str.replace("burden_adult", "burden_adult_men")
dalys_19_male["variable"] = dalys_19_male["variable"].str.replace("prevalence_adult", "prevalence_adult_men")


# merging together female with both sexes
# have to remember that age group 10-49 is just women
dalys_19 = pd.concat([dalys_19, dalys_19_female, dalys_19_male], axis=0)
dalys_10 = pd.concat([dalys_10, dalys_10_female, dalys_10_male], axis=0)


# need to add dalys together for women 10-49 and under 1 for art pmtct
def form_pmtct_totals(df_dalys, list_variable_strings, new_variable_name):
    pmtct_df = df_dalys[df_dalys["variable"].isin(list_variable_strings)]
    pmtct_df = pmtct_df[pmtct_df["cause_id"] == 298]
    pmtct_df_agg = pmtct_df.groupby("location_id").sum()[["burden_variable", "burden_total"]].reset_index()
    pmtct_df_agg["cause_id"] = 298
    pmtct_df_agg["variable"] = new_variable_name
    return pmtct_df_agg


dalys_19_pmtct_burden_agg = form_pmtct_totals(dalys_19, ["burden_10_49", "burden_under_1"], "burden_pmtct")
dalys_10_pmtct_burden_agg = form_pmtct_totals(dalys_10, ["burden_10_49", "burden_under_1"], "burden_pmtct")
dalys_19_pmtct_pre_agg = form_pmtct_totals(dalys_19, ["prevalence_10_49", "prevalence_under_1"], "prevalence_pmtct")
dalys_10_pmtct_pre_agg = form_pmtct_totals(dalys_10, ["prevalence_10_49", "prevalence_under_1"], "prevalence_pmtct")

# concat everything together, so then can use appropriate age groups for each intervention
to_concat_19 = [dalys_19, dalys_19_pmtct_burden_agg, dalys_19_pmtct_pre_agg]
to_concat_10 = [dalys_10, dalys_10_pmtct_burden_agg, dalys_10_pmtct_pre_agg]

dalys_19 = pd.concat(to_concat_19, axis=0)
dalys_10 = pd.concat(to_concat_10, axis=0)

# removing age 10-49 because never use this age group by its self
dalys_19 = dalys_19[~dalys_19["variable"].isin(["burden_10_49", "prevalence_10_49"])]
dalys_10 = dalys_10[~dalys_10["variable"].isin(["burden_10_49", "prevalence_10_49"])]


# ============================pulling population=======================================
def get_pop_function(locns, age_group_id, year_id, sex_list=3):
    output = get_population(
        location_id=locns,
        year_id=year_id,
        age_group_id=age_group_id,
        gbd_round_id=6,
        sex_id=sex_list,
        decomp_step="step5",
    )
    return output


# both sexes
pops_19 = get_pop_function(locns=locns, age_group_id=age_groups_needed, year_id=2019, sex_list=[3])
pops_10 = get_pop_function(locns=locns, age_group_id=age_groups_needed, year_id=2010, sex_list=[3])

# just female - just want population groups 7-14 (ages 10-49 y)
female_ages = list(range(7, 15))
pops_19_female = get_pop_function(locns=locns, age_group_id=female_ages, year_id=2019, sex_list=[2])
pops_10_female = get_pop_function(locns=locns, age_group_id=female_ages, year_id=2010, sex_list=[2])

pops_19_male = get_pop_function(locns=locns, age_group_id=[7, 24, 25, 26], year_id=2019, sex_list=[1])
pops_10_male = get_pop_function(locns=locns, age_group_id=[7, 24, 25, 26], year_id=2010, sex_list=[1])

# reformatting population data so have it in the same format as the burden above
# so can add to burden above
pops_19 = pops_19[["location_id", "age_group_id", "population"]]
pops_10 = pops_10[["location_id", "age_group_id", "population"]]
pops_19_female = pops_19_female[["location_id", "age_group_id", "population"]]
pops_10_female = pops_10_female[["location_id", "age_group_id", "population"]]
pops_19_male = pops_19_male[["location_id", "age_group_id", "population"]]
pops_10_male = pops_10_male[["location_id", "age_group_id", "population"]]

# saving file - atm not saving female - because not using saved csvs anywhere
mr_data_cleaning_functions.exporting_csv(pops_19, "shared_functions/population_2019_230413.csv")
mr_data_cleaning_functions.exporting_csv(pops_10, "shared_functions/population_2010_230413.csv")


# reshaping population df so as to be able to merge onto the dalys dataframe
def reshaping_pop_df(pops):
    """
    Reshaping population of both sexes for to have total population for sets of ages
    of interest:
    pops_p = pivoted df
    pops_m - melted df
    """
    pops["age_group_id"] = pops["age_group_id"].apply(str)
    pops_p = pops_19.pivot(index=["location_id"], columns=["age_group_id"], values=["population"]).reset_index()
    pops_p.columns = pops_p.columns.map("_".join)
    pops_p["burden_all"] = pops_p["population_22"]
    pops_p["burden_under_10"] = pops_p["population_1"] + pops_p["population_6"]
    pops_p["burden_under_1"] = pops_p["population_28"]
    pops_p["burden_under_5"] = pops_p["population_1"]
    pops_p["burden_adult"] = pops_p["population_24"] + pops_p["population_25"] + pops_p["population_26"]

    # dropping unneeded columns
    pops_m = pops_p[
        ["location_id_", "burden_all", "burden_under_1", "burden_under_5", "burden_under_10", "burden_adult"]
    ]
    pops_m = pops_m.rename(columns={"location_id_": "location_id"})
    pops_m = pops_m.melt(
        id_vars=["location_id"],
        value_vars=["burden_all", "burden_under_1", "burden_under_5", "burden_under_10", "burden_adult"],
    )
    return pops_m


pops_m_19 = reshaping_pop_df(pops_19)
pops_m_10 = reshaping_pop_df(pops_10)

pops_m_19 = pops_m_19.rename(columns={"value": "population_2019"})
pops_m_10 = pops_m_10.rename(columns={"value": "population_2010"})


pops_m_19_male = reshaping_pop_df(pops_19_male)
pops_m_10_male = reshaping_pop_df(pops_10_male)

pops_m_19_male = pops_m_19_male.rename(columns={"value": "population_2019"})
pops_m_10_male = pops_m_10_male.rename(columns={"value": "population_2010"})

pops_m_19_male = pops_m_19_male[pops_m_19_male["variable"] == "burden_adult"]
pops_m_10_male = pops_m_10_male[pops_m_10_male["variable"] == "burden_adult"]

pops_m_19_male["variable"] = "burden_adult_men"
pops_m_10_male["variable"] = "burden_adult_men"


# ============== getting population for pmtct ==============================
def reshaping_pop_df_female(pops):
    """
    Reshaping just female for adding all age groups together to get fertile women
    10-49 years old
    pops_p - pivoted df
    pops_m - melted df
    """
    pops["age_group_id"] = pops["age_group_id"].apply(str)
    pops_p = pops_19.pivot(index=["location_id"], columns=["age_group_id"], values=["population"]).reset_index()
    pops_p.columns = pops_p.columns.map("_".join)
    pops_p["burden_10_49"] = (
        pops_p["population_7"]
        + pops_p["population_8"]
        + pops_p["population_9"]
        + pops_p["population_10"]
        + pops_p["population_11"]
        + pops_p["population_12"]
        + pops_p["population_13"]
        + pops_p["population_14"]
    )

    # dropping unneeded columns
    pops_m = pops_p[["location_id_", "burden_10_49"]]
    pops_m = pops_m.rename(columns={"location_id_": "location_id"})
    pops_m = pops_m.melt(
        id_vars=["location_id"],
        value_vars=["burden_10_49"],
    )
    return pops_m


pops_m_19_female = reshaping_pop_df_female(pops_19_female)
pops_m_10_female = reshaping_pop_df_female(pops_10_female)

# need to add population together for pmtct under_1 + fertile women
pops_m_19_female = pops_m_19_female.rename(columns={"value": "value_women"})
pops_m_10_female = pops_m_10_female.rename(columns={"value": "value_women"})

pops_1_19 = pops_m_19[pops_m_19["variable"] == "burden_under_1"]
pops_1_10 = pops_m_10[pops_m_10["variable"] == "burden_under_1"]

pops_1_19 = pops_1_19.rename(columns={"population_2019": "value_1"})
pops_1_10 = pops_1_10.rename(columns={"population_2010": "value_1"})

# merge the 19 and 10 age groups together - female + male
pops_pmtct_19 = pops_m_19_female.merge(pops_1_19, how="outer", on="location_id")
pops_pmtct_10 = pops_m_10_female.merge(pops_1_10, how="outer", on="location_id")

pops_pmtct_19["population_2019"] = pops_pmtct_19["value_women"] + pops_pmtct_19["value_1"]
pops_pmtct_10["population_2010"] = pops_pmtct_10["value_women"] + pops_pmtct_10["value_1"]

pops_pmtct_19 = pops_pmtct_19.drop(columns=["variable_x", "variable_y", "value_women", "value_1"])
pops_pmtct_10 = pops_pmtct_10.drop(columns=["variable_x", "variable_y", "value_women", "value_1"])

pops_pmtct_19["variable"] = "burden_pmtct"
pops_pmtct_10["variable"] = "burden_pmtct"

# merging pmtct age groups on to pop
pops_all_19 = pd.concat([pops_m_19, pops_pmtct_19, pops_m_19_male], axis=0)
pops_all_10 = pd.concat([pops_m_10, pops_pmtct_10, pops_m_10_male], axis=0)

# merging dalys and population into one dataframe
# 2019
dalys_pops_19 = pd.merge(
    left=dalys_19,
    right=pops_all_19,
    how="left",
    left_on=["location_id", "variable"],
    right_on=["location_id", "variable"],
)

# 2010
dalys_pops_10 = pd.merge(
    left=dalys_10,
    right=pops_all_10,
    how="left",
    left_on=["location_id", "variable"],
    right_on=["location_id", "variable"],
)


def sep_burden_prevalence(dalys_df, population):
    """
    Function to take long df with prevalence and dalys in one column, and separating out into
    separate columns.

    :params dalys_df: a df with burden and prevalence values with cols for location_ids and age groups
    :returns df: with two separate columns for dalys and prevalence
    """
    dalys_pops_burden = dalys_df[dalys_df["variable"].str.contains("burden")]
    dalys_pops_burden["age"] = dalys_pops_burden["variable"].str.replace("burden_", "")
    dalys_pops_prevalence = dalys_df[dalys_df["variable"].str.contains("prevalence")]
    dalys_pops_prevalence["age"] = dalys_pops_prevalence["variable"].str.replace("prevalence_", "")
    dalys_pops_prevalence = dalys_pops_prevalence.drop(columns=[population])

    dalys_pops_merged = pd.merge(
        left=dalys_pops_burden,
        right=dalys_pops_prevalence,
        how="outer",
        left_on=["cause_id", "location_id", "age"],
        right_on=["cause_id", "location_id", "age"],
    )

    dalys_pops_merged = dalys_pops_merged.rename(
        columns={
            "burden_variable_x": "burden_variable",
            "burden_total_x": "burden_total",
            "burden_variable_y": "prevalence_per_cap",
            "burden_total_y": "prevalence_total",
        }
    )
    dalys_pops_merged = dalys_pops_merged.drop(columns=["variable_x", "variable_y"])

    return dalys_pops_merged


dalys_pops_merged_19 = sep_burden_prevalence(dalys_pops_19, "population_2019")
# per capita values from shared functions cannot be added together
# so removing the per capita ones
dalys_pops_merged_19["burden_variable_wrong"] = dalys_pops_merged_19["burden_variable"]
dalys_pops_merged_19["prevalence_per_cap_wrong"] = dalys_pops_merged_19["prevalence_per_cap"]

# resaving the total/population
dalys_pops_merged_19["burden_variable"] = dalys_pops_merged_19["burden_total"] / dalys_pops_merged_19["population_2019"]
dalys_pops_merged_19["prevalence_per_cap"] = (
    dalys_pops_merged_19["prevalence_total"] / dalys_pops_merged_19["population_2019"]
)

dalys_pops_merged_10 = sep_burden_prevalence(dalys_pops_10, "population_2010")
# per capita values from shared functions cannot be added together
# so removing the per capita ones
dalys_pops_merged_10["burden_variable_wrong"] = dalys_pops_merged_10["burden_variable"]
dalys_pops_merged_10["prevalence_per_cap_wrong"] = dalys_pops_merged_10["prevalence_per_cap"]
# resaving the total/population
dalys_pops_merged_10["burden_variable"] = dalys_pops_merged_10["burden_total"] / dalys_pops_merged_10["population_2010"]
dalys_pops_merged_10["prevalence_per_cap"] = (
    dalys_pops_merged_10["prevalence_total"] / dalys_pops_merged_10["population_2010"]
)

dalys_pops_merged_19 = dalys_pops_merged_19.drop(columns=["burden_variable_wrong", "prevalence_per_cap_wrong"])
dalys_pops_merged_10 = dalys_pops_merged_10.drop(columns=["burden_variable_wrong", "prevalence_per_cap_wrong"])


# saving dalys/prevalence files for 2019 and 2010
mr_data_cleaning_functions.exporting_csv(dalys_pops_merged_19, "shared_functions/dalys_pops_merged_19.csv")
mr_data_cleaning_functions.exporting_csv(dalys_pops_merged_10, "shared_functions/dalys_pops_merged_10.csv")

# saving dalys/prevalence files
mr_data_cleaning_functions.exporting_csv(dalys_10, "shared_functions/dalys_2010.csv")
mr_data_cleaning_functions.exporting_csv(dalys_19, "shared_functions/dalys_2019.csv")


# ======= GETTING DALYS/ BURDEN PER CAPITA ==========
# making a set of dalys-pop dfs by cause id
dalys_dict = dict()

# divides the dalys and pop data by cause_id, so can merge onto the covs_dict later on
for id in causes_list:
    key_name = causes_df.loc[id, "cause"]
    dalys_dict[key_name] = dalys_pops_merged_19[dalys_pops_merged_19["cause_id"] == id]

# including hiv_tb dalys in tb preds df so that we can predict costs for interventions solely targeting PLHIV
all_tb = pd.merge(
    left=dalys_dict["tuberculosis"],
    right=dalys_dict["hiv_aids_dr_tb"][
        ["location_id", "burden_variable", "burden_total", "age", "prevalence_per_cap", "prevalence_total"]
    ].rename(
        columns={
            "burden_variable": "burden_variable_hiv_dr_tb",
            "burden_total": "burden_total_hiv_dr_tb",
            "prevalence_per_cap": "prev_pc_hiv_dr_tb",
            "prevalence_total": "prev_total_hiv_dr_tb",
        }
    ),
    how="left",
    left_on=["location_id", "age"],
    right_on=["location_id", "age"],
)

all_tb = pd.merge(
    left=all_tb,
    right=dalys_dict["hiv_aids_mdr_tb"][
        ["location_id", "burden_variable", "burden_total", "age", "prevalence_per_cap", "prevalence_total"]
    ].rename(
        columns={
            "burden_variable": "burden_variable_hiv_mdr_tb",
            "burden_total": "burden_total_hiv_mdr_tb",
            "prevalence_per_cap": "prev_pc_hiv_mdr_tb",
            "prevalence_total": "prev_total_hiv_mdr_tb",
        }
    ),
    how="left",
    left_on=["location_id", "age"],
    right_on=["location_id", "age"],
)

all_tb = pd.merge(
    left=all_tb,
    right=dalys_dict["hiv_aids_xdr_tb"][
        ["location_id", "burden_variable", "burden_total", "age", "prevalence_per_cap", "prevalence_total"]
    ].rename(
        columns={
            "burden_variable": "burden_variable_hiv_xdr_tb",
            "burden_total": "burden_total_hiv_xdr_tb",
            "prevalence_per_cap": "prev_pc_hiv_xdr_tb",
            "prevalence_total": "prev_total_hiv_xdr_tb",
        }
    ),
    how="left",
    left_on=["location_id", "age"],
    right_on=["location_id", "age"],
)

all_tb["hiv_tb_burden_variable"] = (
    all_tb["burden_variable_hiv_dr_tb"] + all_tb["burden_variable_hiv_mdr_tb"] + all_tb["burden_variable_hiv_xdr_tb"]
)
all_tb["hiv_tb_burden_total"] = (
    all_tb["burden_total_hiv_dr_tb"] + all_tb["burden_total_hiv_mdr_tb"] + all_tb["burden_total_hiv_xdr_tb"]
)
all_tb["hiv_tb_prev_variable"] = (
    all_tb["prev_pc_hiv_dr_tb"] + all_tb["prev_pc_hiv_mdr_tb"] + all_tb["prev_pc_hiv_xdr_tb"]
)
all_tb["hiv_tb_prev_total"] = (
    all_tb["prev_total_hiv_dr_tb"] + all_tb["prev_total_hiv_mdr_tb"] + all_tb["prev_total_hiv_xdr_tb"]
)

dalys_dict["tuberculosis"] = all_tb
del dalys_dict["hiv_aids_dr_tb"]
del dalys_dict["hiv_aids_mdr_tb"]
del dalys_dict["hiv_aids_xdr_tb"]

# =============NEED TO ADD IN OTHER COVS ==================
# start dict to capture all covariates by cause

# sorting dictionaries to just have locations GF eligible
covs_dict_interim = dict()

for key in dalys_dict.keys():
    covs_dict_interim[key] = dalys_dict[key].copy()
    # LOCATION
    covs_dict_interim[key] = pd.merge(
        left=eligible_countries_dict[key],
        right=covs_dict_interim[key],
        how="left",
        left_on="location_id",
        right_on="location_id",
    )

covs_dict = {}

for cause_int in cause_int_combinations:
    for key in covs_dict_interim.keys():
        if cause_int.startswith(key):
            covs_dict[cause_int] = covs_dict_interim[key]

covs_dict["hiv_pr_wo_art"] = covs_dict["hiv_aids_art"]

# ================ GPD =======================
# gdp_df = pd.read_csv(pth.GDP_DF)
gdp_df = pd.read_csv("GDP_DF.csv")
gdp_df = gdp_df[gdp_df["year"] == 2019]

# no longer have  "GDP_2019ppp_per_cap"
gdp_df = gdp_df.loc[
    gdp_df["year"] == 2019,
    [
        "location_id",
        "GDP_2019usd_per_cap",
    ],
].reset_index(drop=True)

for key in covs_dict.keys():
    covs_dict[key] = pd.merge(
        left=covs_dict[key], right=gdp_df, how="left", left_on="location_id", right_on="location_id"
    )

# ======== LOG OF SOME VARIABLES==========
vars_to_log = ["burden_variable", "GDP_2019usd_per_cap", "prevalence_per_cap"]

for key in covs_dict.keys():
    print(key)
    covs_dict[key] = covs_dict[key].assign(**{"log_" + k: np.log(covs_dict[key][k]) for k in vars_to_log})

vars_to_log_tb = ["hiv_tb_burden_variable", "hiv_tb_prev_variable"]

for key in ["tuberculosis_prevention", "tuberculosis_diagnostic"]:
    covs_dict[key] = covs_dict[key].assign(**{"log_" + k: np.log(covs_dict[key][k]) for k in vars_to_log_tb})

# ======= setting values for other covariates -====================================
cov_default_dict = {
    "intercept": 1,
    "CostsDiscountRate": 3,
    "DiscountRate": 3,
    "payer_or_sector": 1,
    "not_lifetime": 1,
    "not_dalys": 1,
    "discount_under_3": 0,
}

for key in covs_dict.keys():
    covs_dict[key] = covs_dict[key].assign(**{k: v for k, v in cov_default_dict.items()})

# adding the mean of timehorizon to use in early predictions
# making all timehorizon 75 years for predictions
for key in covs_dict.keys():
    covs_dict[key]["TimeHorizonMagnitude"] = 75

# removing some columns not relevant for intervention type
# trying to align the columns in the pred_dfs to align with covariates in model
for key in covs_dict.keys():
    covs_dict[key] = covs_dict[key].drop(
        columns=[
            "GDP_2019usd_per_cap",
        ]
    )

# ==== COST AND EFFICACY ===========
# the remainder of the code is structured to read from the cost and efficacy csv
# to add in log_cost and efficacy values to predictions df for different age groups

# making a column to match to each of the cause_intervention combinations
costs_df["cause_int"] = costs_df["all_causes_per_ratio"] + "_" + costs_df["intervention_type"]

# ========= NOTE: ======================
# structure of predictions dfs will be stacked. i.e. so one columns can be turned on and off to run different predictions
# but so one df can hold all info needed for all predictions per cause_int combination

# ============= HIV ART =================
# reference:  art for prevention
# predicting for art for prevention only
# costs for adults and child

costs_efficacy_art = costs_df[costs_df["cause_int"] == "hiv_aids_art"]
costs_efficacy_art_child = costs_efficacy_art[costs_efficacy_art["age_group"] == "child"]
costs_efficacy_art_adult = costs_efficacy_art[costs_efficacy_art["age_group"] == "adult"]

covs_dict["hiv_aids_art"]["efficacy"] = costs_efficacy_art["efficacy"].iloc[0]

# for adults
covs_dict["hiv_aids_art"]["log_per_year_or_full_int_cost"] = np.log(
    costs_efficacy_art_adult["cost_per_year_protection"].iloc[0]
)
# for children
covs_dict["hiv_aids_art"].loc[covs_dict["hiv_aids_art"]["age"] == "under_10", "log_per_year_or_full_int_cost"] = np.log(
    costs_efficacy_art_child["cost_per_year_protection"].iloc[0]
)

# need to update for cost for children
# dropping all, babies and under 5 because don't need for art
covs_dict["hiv_aids_art"] = covs_dict["hiv_aids_art"][covs_dict["hiv_aids_art"]["age"] != "all"]
covs_dict["hiv_aids_art"] = covs_dict["hiv_aids_art"][covs_dict["hiv_aids_art"]["age"] != "under_5"]
covs_dict["hiv_aids_art"] = covs_dict["hiv_aids_art"][covs_dict["hiv_aids_art"]["age"] != "under_1"]

covs_dict["hiv_aids_art"]["cd4_grouping"] = 0
covs_dict["hiv_aids_art"]["second_line"] = 0


# all intervention keyword dummy variables
hiv_art = [
    "antiretroviral therapy for hiv, immunological antiretroviral treatment initiation criteria",
    "antiretroviral therapy for hiv, hiv testing",
    "antiretroviral therapy for hiv for prevention, hiv testing",
    "antiretroviral therapy for hiv",
    "pooled hiv testing, antiretroviral therapy for hiv for prevention",
    "prevention of mother to child hiv transmission, antiretroviral therapy for hiv for prevention",
    "antiretroviral therapy for hiv for prevention, immunological antiretroviral treatment initiation criteria",
    "antiretroviral therapy for hiv for prevention, methadone maintenance therapy",
    "50",
    "200",
    "350",
    "500",
]

for col in hiv_art:
    covs_dict["hiv_aids_art"][col] = 0

# this is to include a column 'log_dalys_per_cap' because we use this column to predict rather than log_burden
# this could be changed in future
covs_dict["hiv_aids_art"]["log_dalys_per_cap"] = covs_dict["hiv_aids_art"]["log_burden_variable"]

covs_dict["hiv_aids_art"].loc[
    covs_dict["hiv_aids_art"]["age"] == "pmtct",
    "prevention of mother to child hiv transmission, antiretroviral therapy for hiv for prevention",
] = 1

# ================ HIV Prevention and wo art = ========================
# reference: pre-exposure prophylaxis for hiv
# change ref to male circumsicion - won't work for alternative model wo male circum
# Predicting for just adults for pep and prep, and potentially male circumcision - adults and babies
costs_efficacy_hiv_pr_wo_art = costs_df[costs_df["cause_int"] == "hiv_aids_pr_wo_art"].set_index(
    "intervention_keywords_final"
)
covs_dict["hiv_pr_wo_art"]["log_dalys_per_cap"] = covs_dict["hiv_pr_wo_art"]["log_burden_variable"]

hiv_pr_wo_art = [
    "education to prevent hiv",
    "treatment of sexually transmitted infections",
    "post-exposure prophylaxis for hiv",
    "voluntary hiv testing and counseling",
    "prevention of mother to child hiv transmission",
    "prevention of mother to child hiv transmission, antiretroviral therapy for hiv for prevention",
    "male circumcision",
    "pooled hiv testing, antiretroviral therapy for hiv for prevention",
    "rotavirus vaccines",
]

for col in hiv_pr_wo_art:
    covs_dict["hiv_pr_wo_art"][col] = 0

# code separates and maps costs and efficacy for each of the individual keywords
# then merges the 3 df together

# 1. pre-exposure prophylaxis PREP (reference)
pre_exposure_cost = float(
    costs_efficacy_hiv_pr_wo_art.loc["pre-exposure prophylaxis for hiv"]["cost_per_year_protection"]
)

prep_df_hetero = covs_dict["hiv_pr_wo_art"].copy()
# cost and efficacy
prep_df_hetero["log_per_year_or_full_int_cost"] = np.log(pre_exposure_cost)
prep_df_hetero["efficacy"] = costs_efficacy_hiv_pr_wo_art.loc["pre-exposure prophylaxis for hiv"]["efficacy"]

# drop everything but adult
prep_df_hetero = prep_df_hetero[prep_df_hetero["age"] == "adult"]
prep_df_hetero["msm_risk_group"] = 0

prep_df_msm = covs_dict["hiv_pr_wo_art"].copy()
prep_df_msm["log_per_year_or_full_int_cost"] = np.log(pre_exposure_cost)
prep_df_msm["efficacy"] = costs_efficacy_hiv_pr_wo_art.loc["pre-exposure prophylaxis for hiv"]["efficacy"]

# drop everything but adult
prep_df_msm = prep_df_msm[prep_df_msm["age"] == "adult_men"]
prep_df_msm["msm_risk_group"] = 1

prep_df_x3 = prep_df_hetero.copy()
prep_df_x3["log_dalys_per_cap"] = np.log(prep_df_x3["burden_variable"] * 3)
prep_df_x3["log_prevalence_per_cap"] = np.log(prep_df_x3["prevalence_per_cap"] * 3)
prep_df_x3["age"] = "adult - x3"

prep_df_x5 = prep_df_hetero.copy()
prep_df_x5["log_dalys_per_cap"] = np.log(prep_df_x5["burden_variable"] * 5)
prep_df_x5["log_prevalence_per_cap"] = np.log(prep_df_x5["prevalence_per_cap"] * 5)
prep_df_x5["age"] = "adult - x5"

prep_df_x10 = prep_df_hetero.copy()
prep_df_x10["log_dalys_per_cap"] = np.log(prep_df_x10["burden_variable"] * 10)
prep_df_x10["log_prevalence_per_cap"] = np.log(prep_df_x10["prevalence_per_cap"] * 10)
prep_df_x10["age"] = "adult - x10"

prep_df_all = pd.concat([prep_df_hetero, prep_df_msm, prep_df_x3, prep_df_x5, prep_df_x10])

# 2. post-exposure prophylaxis PEP
pep_df = covs_dict["hiv_pr_wo_art"].copy()
pep_df["post-exposure prophylaxis for hiv"] = 1
pep_cost = float(costs_efficacy_hiv_pr_wo_art.loc["post-exposure prophylaxis for hiv"]["cost_per_year_protection"])
pep_df["log_per_year_or_full_int_cost"] = np.log(pep_cost)

# efficacy calc - average between 81-99.5%
pep_df["efficacy"] = costs_efficacy_hiv_pr_wo_art.loc["post-exposure prophylaxis for hiv"]["efficacy"]
pep_df = pep_df[pep_df["age"] == "adult"]

pep_kw = [
    "hiv positive - receptive vaginal - heterosexuals",
    "hiv positive - insertive anal - homosexuals",
    "hiv positive - insertive anal/vaginal - heterosexuals",
    "status unknown - receptive anal - homosexuals",
    "status unknown - receptive vaginal - heterosexuals",
    "status unknown - insertive anal - homosexuals",
    "status unknown - insertive anal/vaginal - heterosexuals",
]

for col in pep_kw:
    pep_df[col] = 0

pep_df_list = []

for col in pep_kw:
    sub_df = pep_df.copy()
    sub_df[col] = 1
    print(col)
    if "homosexual" in col:
        sub_df["msm_risk_group"] = 1
    pep_df_list.append(sub_df)

pep_df_risk_groups = pd.concat(pep_df_list)
pep_df_risk_groups = pd.concat([pep_df_risk_groups, pep_df])

pep_df_risk_groups["msm_risk_group"] = pep_df_risk_groups["msm_risk_group"].fillna(0)

# 3. male circumcision
costs_adult = costs_efficacy_hiv_pr_wo_art[costs_efficacy_hiv_pr_wo_art["age_group"] == "adult"]
costs_under_1 = costs_efficacy_hiv_pr_wo_art[costs_efficacy_hiv_pr_wo_art["age_group"] == "0-11 months"]

cir_df = covs_dict["hiv_pr_wo_art"].copy()

# changing ref case to male circu
cir_df["male circumcision"] = 1
ages_mc = ["under_1", "adult"]
cir_df = cir_df[(cir_df["age"].isin(ages_mc))]

cir_cost_adult = float(costs_adult.loc["male circumcision"]["cost_per_year_protection"])
cir_cost_under_1 = float(costs_under_1.loc["male circumcision"]["cost_per_year_protection"])

cir_df.loc[cir_df["age"] == "adult", "log_per_year_or_full_int_cost"] = np.log(cir_cost_adult)
cir_df.loc[cir_df["age"] == "under_1", "log_per_year_or_full_int_cost"] = np.log(cir_cost_under_1)

cir_df["efficacy"] = costs_under_1.loc["male circumcision"]["efficacy"]

# all hiv pr interventions together merging into covs_dict
covs_dict["hiv_pr_wo_art"] = pd.concat([prep_df_all, cir_df])
covs_dict["hiv_pr_wo_art"] = pd.concat([covs_dict["hiv_pr_wo_art"], pep_df])
covs_dict["pep"] = pep_df_risk_groups
covs_dict["prep"] = prep_df_all
covs_dict["male_circum"] = cir_df
covs_dict["prep_pep"] = pd.concat([prep_df_all, pep_df])

# ===================MALARIA ==================
# reference: combination of BED NETS, will predict for this

malaria_keywords_to_add = [
    "malaria intermittent preventive treatment",
    "bed nets",
    # "malaria vaccines", reference case
    "indoor residual spraying, malaria treatment, bed nets",
    "malaria intermittent preventive treatment for pregnant women",
    "indoor residual spraying",
    "rotavirus vaccines",
    "indoor residual spraying, malaria treatment, bed nets",
]

for col in malaria_keywords_to_add:
    covs_dict["malaria_prevention"][col] = 0

covs_dict["malaria_prevention"]["log_dalys_per_cap"] = covs_dict["malaria_prevention"]["log_burden_variable"]

costs_efficacy_malaria_prevention = costs_df[costs_df["cause_int"] == "malaria_prevention"].set_index(
    "intervention_keywords_final"
)

covs_dict["malaria_prevention"] = covs_dict["malaria_prevention"][
    covs_dict["malaria_prevention"]["log_burden_variable"] != -np.inf
]

# bed nets
bed_df = covs_dict["malaria_prevention"].copy()
bed_df["bed nets"] = 1
bed_cost = float(costs_efficacy_malaria_prevention.loc["bed nets"]["cost_per_year_protection"])
bed_df["log_per_year_or_full_int_cost"] = np.log(bed_cost)
bed_df["efficacy"] = costs_efficacy_malaria_prevention.loc["bed nets"]["efficacy"]
bed_df = bed_df[bed_df["age"] == "all"]

# vaccine
vaccine_df = covs_dict["malaria_prevention"].copy()
vaccine_cost = float(costs_efficacy_malaria_prevention.loc["malaria vaccines"]["cost_per_year_protection"])
vaccine_df["log_per_year_or_full_int_cost"] = np.log(vaccine_cost)
vaccine_df["efficacy"] = costs_efficacy_malaria_prevention.loc["malaria vaccines"]["efficacy"]
vaccine_df = vaccine_df[vaccine_df["age"] == "under_5"]

# 'malaria intermittent preventive treatment'
ipt_df = covs_dict["malaria_prevention"].copy()
ipt_cost = float(
    costs_efficacy_malaria_prevention.loc["malaria intermittent preventive treatment for infants"][
        "cost_per_year_protection"
    ]
)
ipt_df["log_per_year_or_full_int_cost"] = np.log(ipt_cost)
ipt_df["efficacy"] = costs_efficacy_malaria_prevention.loc["malaria intermittent preventive treatment for infants"][
    "efficacy"
]
ipt_df["malaria intermittent preventive treatment"] = 1
ipt_df = ipt_df[ipt_df["age"] == "under_1"]

#  'malaria intermittent preventive treatment for pregnant women'
ipt_p_df = covs_dict["malaria_prevention"].copy()
ipt_p_cost = float(
    costs_efficacy_malaria_prevention.loc["malaria intermittent preventive treatment for pregnant women"][
        "cost_per_year_protection"
    ]
)
ipt_p_df["log_per_year_or_full_int_cost"] = np.log(ipt_p_cost)
ipt_p_df["efficacy"] = costs_efficacy_malaria_prevention.loc[
    "malaria intermittent preventive treatment for pregnant women"
]["efficacy"]
ipt_p_df["malaria intermittent preventive treatment for pregnant women"] = 1
ipt_p_df = ipt_p_df[ipt_p_df["age"] == "under_1"]

#  indoor residual spraying
irs_df = covs_dict["malaria_prevention"].copy()
irs_cost = float(costs_efficacy_malaria_prevention.loc["indoor residual spraying"]["cost_per_year_protection"])
irs_df["log_per_year_or_full_int_cost"] = np.log(irs_cost)
irs_df["efficacy"] = costs_efficacy_malaria_prevention.loc["indoor residual spraying"]["efficacy"]
irs_df["indoor residual spraying"] = 1
irs_df = irs_df[irs_df["age"] == "all"]

# merging together
covs_dict["malaria_prevention"] = pd.concat([bed_df, vaccine_df])
covs_dict["malaria_prevention"] = pd.concat([covs_dict["malaria_prevention"], ipt_df])
covs_dict["malaria_prevention"] = pd.concat([covs_dict["malaria_prevention"], ipt_p_df])
covs_dict["malaria_prevention"] = pd.concat([covs_dict["malaria_prevention"], irs_df])

# ===================TB diagnostic ==================
# reference: xpert rapid tuberculosis test
# xpert for all reference case
# LF-LAM for ppl with hiv - tuberculosis testing -

tb_d_costs_eff = costs_df[costs_df["cause_int"] == "tuberculosis_diagnostic"].set_index("intervention_keywords_final")

tb_diagnostic_cols = [
    "tuberculosis screening with tuberculin skin test",
    "tuberculosis screening with smear microscopy",
    "tuberculosis testing",
    "tuberculosis screening with interferon gamma release assays",
]

for col in tb_diagnostic_cols:
    covs_dict["tuberculosis_diagnostic"][col] = 0

covs_dict["tuberculosis_diagnostic"] = covs_dict["tuberculosis_diagnostic"][
    covs_dict["tuberculosis_diagnostic"]["age"] == "all"
]

# xpert - reference
xpert_costs = float(tb_d_costs_eff.loc["xpert rapid tuberculosis test"]["cost"])
covs_dict["tuberculosis_diagnostic"]["log_per_year_or_full_int_cost"] = np.log(xpert_costs)
covs_dict["tuberculosis_diagnostic"]["sensitivity"] = tb_d_costs_eff.loc["xpert rapid tuberculosis test"]["sensitivity"]
covs_dict["tuberculosis_diagnostic"]["specificity"] = tb_d_costs_eff.loc["xpert rapid tuberculosis test"]["specificity"]
covs_dict["tuberculosis_diagnostic"]["hiv_pop"] = 0

# merging dfs back together
covs_dict["tuberculosis_diagnostic"]["log_dalys_per_cap"] = covs_dict["tuberculosis_diagnostic"]["log_burden_variable"]

# ===================TB prevention ==================
# reference: prophylaxis for people without active tb, tuberculosis screening with tuberculin skin test

tb_pr_costs_eff = costs_df[costs_df["cause_int"] == "tuberculosis_prevention"].set_index("intervention_keywords_final")

tb_pr_cols = ["tuberculosis vaccines", "prophylaxis for people without active tb", "rotavirus vaccines"]

for col in tb_pr_cols:
    covs_dict["tuberculosis_prevention"][col] = 0

# confirming what countries are gavi countries vs global fund countries
gavi_countries_list = gavi_countries["location_id"].unique()
covs_dict["tuberculosis_prevention"]["gavi"] = 0
covs_dict["tuberculosis_prevention"].loc[
    covs_dict["tuberculosis_prevention"]["location_id"].isin(gavi_countries_list), "gavi"
] = 1
# nauru and tuvalu are not global fund countries

# 1. tb vaccines
tb_vaccine = covs_dict["tuberculosis_prevention"].copy()
tb_vaccine["tuberculosis vaccines"] = 1

# vaccine costs gavi and non gavi countries
tb_vaccine_cost_gavi = float(tb_pr_costs_eff.loc["tuberculosis vaccines - gavi"]["cost_per_year_protection"])
tb_vaccine_cost = float(tb_pr_costs_eff.loc["tuberculosis vaccines"]["cost_per_year_protection"])

tb_vaccine.loc[tb_vaccine["gavi"] == 1, "log_per_year_or_full_int_cost"] = np.log(tb_vaccine_cost_gavi)
tb_vaccine.loc[tb_vaccine["gavi"] == 0, "log_per_year_or_full_int_cost"] = np.log(tb_vaccine_cost)

tb_vaccine["efficacy"] = tb_pr_costs_eff.loc["tuberculosis vaccines"]["efficacy"]
tb_vaccine = tb_vaccine[tb_vaccine["age"] == "under_5"]
tb_vaccine["hiv_pop"] = 0

# 2. prophylaxis
prophylaxis_cost = float(tb_pr_costs_eff.loc["prophylaxis for people without active tb"]["cost_per_year_protection"])

proph_df = covs_dict["tuberculosis_prevention"].copy()
proph_df = proph_df[(proph_df["age"] == "adult")]
proph_df["log_per_year_or_full_int_cost"] = np.log(prophylaxis_cost)
# only predicting out for prophylaxis by itself
proph_df["prophylaxis for people without active tb"] = 1

proph_df["efficacy"] = tb_pr_costs_eff.loc["prophylaxis for people without active tb"]["efficacy"]
proph_df["hiv_pop"] = 0

# reconfiguring burden to have just burden for hiv pop
hiv_tb = proph_df.copy()
hiv_tb = hiv_tb[(hiv_tb["age"] == "adult")]
hiv_tb["log_burden_variable"] = hiv_tb["log_hiv_tb_burden_variable"]
hiv_tb["log_prevalence_per_cap"] = hiv_tb["log_hiv_tb_prev_variable"]
hiv_tb["hiv_pop"] = 1

# merging on set of data so can predict for hiv populations as well
covs_dict["tuberculosis_prevention"] = pd.concat([proph_df, hiv_tb])
covs_dict["tuberculosis_prevention"] = pd.concat([covs_dict["tuberculosis_prevention"], tb_vaccine])

# final updates to the entire predictions df
covs_dict["tuberculosis_prevention"]["log_dalys_per_cap"] = covs_dict["tuberculosis_prevention"]["log_burden_variable"]


# ========================= TB TREATMENT ===================
# reference case tb treatment : antibiotics for tuberculosis
# covs_dict['tuberculosis_treatment']

# costs and efficacy for tb treatment
tb_tr_costs_eff = costs_df[costs_df["cause_int"] == "tuberculosis_treatment"].set_index("intervention_keywords_final")

# cleaning up tb treatment and adding keyword columns prior to add specific costs for each keyword
covs_dict["tuberculosis_treatment"]["antibiotics for multidrug resistant tuberculosis"] = 0
covs_dict["tuberculosis_treatment"] = covs_dict["tuberculosis_treatment"][
    covs_dict["tuberculosis_treatment"]["age"] == "all"
]

covs_dict["tuberculosis_treatment"]["log_dalys_per_cap"] = covs_dict["tuberculosis_treatment"]["log_burden_variable"]

# 1. antibiotics for tb
# reference case
ab_tb = covs_dict["tuberculosis_treatment"].copy()
ab_tb_cost = float(tb_tr_costs_eff.loc["antibiotics for tuberculosis"]["cost_per_year_protection"])
ab_tb["log_per_year_or_full_int_cost"] = np.log(ab_tb_cost)
ab_tb["efficacy"] = tb_tr_costs_eff.loc["antibiotics for tuberculosis"]["efficacy"]

# 2. antibiotics for MDR TB
ab_mdr = covs_dict["tuberculosis_treatment"].copy()
ab_mdr["antibiotics for multidrug resistant tuberculosis"] = 1
ab_mdr_cost = float(tb_tr_costs_eff.loc["antibiotics for multidrug resistant tuberculosis"]["cost_per_year_protection"])
ab_mdr["log_per_year_or_full_int_cost"] = np.log(ab_mdr_cost)
ab_mdr["efficacy"] = tb_tr_costs_eff.loc["antibiotics for multidrug resistant tuberculosis"]["efficacy"]

covs_dict["tuberculosis_treatment"] = pd.concat([ab_tb, ab_mdr])

# ===================Syphilis diagnostic ==================
# antibiotics for syphilis, syphilis testing
syp_costs_eff = costs_df[costs_df["cause_int"] == "syphilis_diagnosis"].set_index("intervention_keywords_final")
syp_cost = float(syp_costs_eff.loc["antibiotics for syphilis, syphilis testing"]["cost_per_year_protection"])

covs_dict["syphilis_diagnosis"]["per_year_or_full_int_cost"] = syp_cost
covs_dict["syphilis_diagnosis"]["log_per_year_or_full_int_cost"] = np.log(syp_cost)
covs_dict["syphilis_diagnosis"]["efficacy"] = syp_costs_eff.loc["antibiotics for syphilis, syphilis testing"][
    "efficacy"
]
covs_dict["syphilis_diagnosis"]["log_dalys_per_cap"] = covs_dict["syphilis_diagnosis"]["log_burden_variable"]
covs_dict["syphilis_diagnosis"]["rotavirus vaccines"] = 0
covs_dict["syphilis_diagnosis"] = covs_dict["syphilis_diagnosis"][covs_dict["syphilis_diagnosis"]["age"] == "under_1"]

# saving file
for key in covs_dict.keys():
    mr_data_cleaning_functions.exporting_csv(covs_dict[key], "{key}_predictions_df.csv")

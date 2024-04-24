import numpy as np
import pandas as pd
import os


# imports cleaned df from data cleaning step
df = pd.read_csv("CLEANED_DF.csv")

# diagnostic is the reference level
dtp_df = pd.get_dummies(df["diagnostic_treatment_prevention"])
df = df.drop("diagnostic", axis=1).join(dtp_df)

# Construct a list of variables that may have sensitivity-reference pairs in the data set
sens_vars = [
    "DiscountRate",
    "CostsDiscountRate",
    "qalys",
    "TimeHorizonMagnitude",
    "ltd_or_societal",
    "ltd_societal",
    "sector",
    "treatment",
    "prevention",
    "log_per_year_or_full_int_cost",
    "coverage",
    "sensitivity",
    "specificity",
    "efficacy",
]


# Construct a dict of baseline values used for selecting reference analyses
# these are the values that are the ones we want to predict for
base_vals = {
    "log_burden_variable": np.nan,
    "DiscountRate": 3,
    "CostsDiscountRate": 3,
    "qalys": 0,
    "TimeHorizonMagnitude": 100,
    "ltd_or_societal": 0,
    "ltd_societal": 0,
    "sector": 0,
    "log_GDP_per_cap": np.nan,
    "treatment": 0,
    "prevention": 0,
    "log_per_year_or_full_int_cost": np.nan,
    "coverage": np.nan,
    "sensitivity": np.nan,
    "specificity": np.nan,
    "efficacy": np.nan,
}


# Construct a list of columns of df that must be identical between sensitivity and reference pairs
# and are not included in sens_vars.
identical_vars = [
    "all_causes_per_ratio",
    "standardized_intervention_phrase",
    "ihme_intervention_describe",
    "Random_Effects_ID",
    "TargetPopulation",
    "age_group_id_updated_ihme",
    "risk_group_code",
    "cd4_count",
    "ComparatorID",
    "ComparatorPhrase",
    "pop_prevalence",
    "target_prevalence",
    "target_transmission_prob",
    "sex_id",
    "location_id",
    "YearOfCurrency",
    "ArticleID",
    "PubMedID",
    "intervention_keywords_final",
]

## Alternative columns to use for a given crosswalk
# i.e. have different versions of cost, denominator discounting
# important to exclude the log_discounted_rate, because only want one covariate present in covs list
# (log_cost, discount_rate, log_dicounted_dalys - all on different rates that would be bad)
alt_cov_versions = {
    f"{a}log_discounted_dalys_from_{b}_per_cap": f"{a}DiscountRate"
    for a in ["", "CostsDiscountRate,"]
    for b in ["start", "birth"]
}
alt_cov_versions.update(
    {
        i: "log_per_year_or_full_int_cost"
        for i in [
            "log_cost_diff_int_comp_pc_usd",
            "log_pr_year_cost_diff_int_comp_pc_usd",
            "log_tr_year_cost_diff_int_comp_pc_usd",
            "log_total_cost_per_cap_usd",
        ]
    }
)

# Merge df with itself to construct every pair of analyses from the same article & location.
pairs_df = df[["ArticleID", "location_id", "RatioID"]]
pairs_df = pairs_df.merge(
    pairs_df, on=["ArticleID", "location_id"], how="left", suffixes=["_sens", "_ref"]
)

# Drop rows where a ratio has been paired with itself.
pairs_df = pairs_df[pairs_df["RatioID_sens"] != pairs_df["RatioID_ref"]]
pairs_df = pairs_df.drop(["ArticleID", "location_id"], axis=1)

# list of all columns other than the columns that we are checking to pair sens and ref analysis
additional_cols = [
    "log_icer_usd",
    "log_icer_se",
    "log_discounted_dalys_from_start_per_cap",
    "log_discounted_dalys_from_birth_per_cap",
    "log_pr_protection_year_cost_usd",
    "log_cost_diff_int_comp_pc_usd",
    "log_pr_year_cost_diff_int_comp_pc_usd",
    "log_tr_year_cost_diff_int_comp_pc_usd",
    "log_total_cost_per_cap_usd",
    "total_population",
    "population",
]

resp_and_covs = sens_vars + identical_vars + additional_cols


# Construct data frames for the sensitivity and reference analyses' covariate values and log ICERs
# and use a merge to filter down to only the ratios present in pairs_df.
sens_pairs = df[["RatioID", "reference_scenario"] + resp_and_covs]
sens_pairs = sens_pairs.rename({"RatioID": "RatioID_sens"}, axis=1)
sens_pairs = sens_pairs.merge(pairs_df, on="RatioID_sens")
ref_pairs = df[["RatioID", "reference_scenario"] + resp_and_covs]
ref_pairs = ref_pairs.rename({"RatioID": "RatioID_ref"}, axis=1)
ref_pairs = ref_pairs.merge(pairs_df, on="RatioID_ref")


ratioid_cols = ["RatioID_sens", "RatioID_ref"]
sens_pairs = sens_pairs.set_index(ratioid_cols)
ref_pairs = ref_pairs.set_index(ratioid_cols)

# Construct a data frame with multiindex columns that contains covariate, icer,
# and identical_vars values for sensitivity and reference analyses.
sr_df = pd.concat([sens_pairs, ref_pairs], axis=1, keys=["sens", "ref"])


def get_all_sr_pairs(sens_ref_df, cvt, sens_variables, suffixes=["sens", "ref"]):
    """
    Finds all pairs of ratios that differ in all and only covariates in a given list.

    Example
    3 different values of cost, this function determines that those differences are just
    in the cost columns (i.e. difference is just in cost and in no other covariates)
    one row for each pair of ratios in that group of 3.
    This function creates the df with 6
    other functions later on, drop the extra info that we don't need

    Args:
        sens_ref_df (pd.DataFrame): DataFrame with multiindex columns with values of suffixes
            in the outer level and values of sens_ref_variables in the inner level.
        cvt (list of str or str): Column labels of covariate. This is the covariate that differs btw sens and ref analysis.
        sens_variables (list of str): Column labels of all covariates. List of covariates that aren't meant to be the different.
        suffixes (list of str): List with 2 elements that specify sensitivity or reference.

    Returns:
        sr_pairs: (pd.DataFrame): DataFrame consisting of the subset of rows of sens_ref_df where
            the entries for cvt differ between sensitivity and reference and where the entries
            for all other entries in sens_variables are identical between sensitivity and reference.
    """
    # checking if string or list and making sure it is a string or list of strings
    if isinstance(cvt, list):
        assert all([isinstance(i, str) for i in cvt])
        cvts = cvt
    else:
        assert isinstance(cvt, str)
        cvts = [cvt]

    # creating a mask of where covariates are unequal
    cvt_unequal = (
        sens_ref_df[suffixes[0]][cvts] != sens_ref_df[suffixes[1]][cvts]
    ).any(axis=1)

    # creating list of variables that are not in cvts
    other_cvts = [j for j in sens_variables if j not in cvts]

    # creating another mask  of where covariates are meant to be the same
    # checking that they are the same
    other_cvts_equal = (
        sens_ref_df[suffixes[0]][other_cvts] == sens_ref_df[suffixes[1]][other_cvts]
    )

    other_cvts_equal = (other_cvts_equal) | (
        (sens_ref_df[suffixes[0]][other_cvts].isnull())
        & (sens_ref_df[suffixes[1]][other_cvts].isnull())
    )

    other_cvts_equal = other_cvts_equal.all(axis=1)

    # creating df where from original input df where one cov is equal and rest are equal
    sr_pairs = sens_ref_df.loc[(cvt_unequal) & (other_cvts_equal)].copy()

    return sr_pairs


def filter_cost_saving_pairs(
    sr_df,
    cvt,
    resp_name="log_icer_usd",
    suffixes=["sens", "ref"],
    ratio_id_name="RatioID",
):
    """
    Drops sensitivity-reference pairs where either analysis is not in the ICER quadrant unless
    the sensitivity analysis is in the ICER quadrant but cannot be paired with a reference
    analysis that is. Also drop sensitivity-reference pairs where either analysis has a null
    value for any of the listed covariates.

    Args:
        sr_df (pd.DataFrame): DataFrame of sensitivity-reference pairs. Must have column
            multiindex whose outer level has values equal to the entries of suffixes and
            whose inner level has values resp_name and ratio_id_name.
        cvt (str or list of str): Column label(s) of the covariate(s) that must not be NA
            for both sensitivity and reference.
        resp_name (str): Column label of the response that must not be NA/infinite
        suffixes (list of str): List of 2 elements that specify sensitivity or reference.
            Sensitivity must be first. Corresponds to the outer level of the column multiindex.
        ratio_id_name (str): Column label of the ratio identifier.

    Returns:
        Dataframe
    """

    # confirming type of input and dealing with different types for cvt i.e. string or list of strings
    if isinstance(cvt, list):
        assert all([isinstance(i, str) for i in cvt])
        cvts = cvt
    else:
        assert isinstance(cvt, str)
        cvts = [cvt]

    # making resp_name column nan when the value equal infinity
    sr_df.loc[np.isinf(sr_df[(suffixes[0], resp_name)]), (suffixes[0], resp_name)] = (
        np.nan
    )
    sr_df.loc[np.isinf(sr_df[(suffixes[1], resp_name)]), (suffixes[1], resp_name)] = (
        np.nan
    )

    # selecting only rows where resp_name is not null
    sr_df = sr_df.loc[(sr_df[suffixes[0]][resp_name].notnull())].copy()

    #
    all_refs_null_icer = sr_df.groupby(level=ratio_id_name + "_" + suffixes[0])[
        (suffixes[1], resp_name),
    ].agg(lambda x: (x.isnull() | np.isinf(x)).all())

    sr_df = sr_df.reset_index(ratio_id_name + "_" + suffixes[1])
    sr_df["all_refs_null_icer"] = all_refs_null_icer
    sr_df = sr_df.set_index(ratio_id_name + "_" + suffixes[1], append=True)

    #
    sr_df = sr_df[
        (sr_df["all_refs_null_icer"]) | (sr_df[("ref", "log_icer_usd")].notnull())
    ].copy()
    idx = pd.IndexSlice
    sr_df = sr_df[sr_df.loc[:, idx[:, cvts]].notnull().all(axis=1)]

    return sr_df


def get_conn_comps(tpls):
    """
    Takes the edge-set of a graph, represented by a list of 2-tuples, and partitions
    the set of non-isolated vertices into its connected components.

    This function returns a list of sets
    If paired_df compares r1 to r2, and r1 to r3, then r1, r2, r3 should all be
    in the same set

    Args:
        tpls: (list of tuples): List of tuples that represent the set of edges of a
            graph. Each tuple must have exactly 2 entries, which must be distinct.
            Entries of each tuple must also be hashable.

    Returns:
        ccomps (list of sets): list of sets of vertices such that each set forms a
            connected component of the graph represented by tpls.
    """
    ccomps = {}
    assignments = {}
    cc_counter = 0
    for x in tpls:
        assg0 = assignments.get(x[0])
        assg1 = assignments.get(x[1])
        if assg0 is None and assg1 is None:
            # If neither sens nor ref ratio has been assigned a clique,
            # create a new clique and assign them to it
            ccomps[cc_counter] = {x[0], x[1]}
            assignments.update({i: cc_counter for i in x})
            cc_counter += 1
        elif assg0 is not None and assg1 is not None:
            # If both sens and ref have been assigned a clique, if they have
            # been assigned different cliques, merge those cliques and reassign
            # the deleted clique's ratios to the new clique. Otherwise do nothing
            if assg0 != assg1:
                assignments.update({i: assg0 for i in ccomps[assg1]})
                ccomps[assg0] = ccomps[assg0].union(ccomps[assg1])
                del ccomps[assg1]
        else:
            # If only one has been assigned a clique, assign the other to the
            # same clique
            assert assg1 is None or assg0 is None
            not_none_assg = assg0 if assg0 is not None else assg1
            assignments.update({i: not_none_assg for i in x})
            ccomps[not_none_assg].update(set(x))
    return list(ccomps.values())


def select_ref_one_ccomp(ccomp, df, cvts, base_vals=None):
    """
    This takes a set of ratioIDs and chooses one of those to be a ref case.

    ccomp (list of sets of strings):
    df (dataframe): The df that has paired potential ref and sens ratios together.
    cvts (list of strings): List of covariate name (usually 1 in list, occassionally 2)
    base_vals (): Single base value when single covariate, for multivariate
        it will be a list of base values.

    Returns:
        Dataframe with rows where the chosen reference cases from function ,
        are the reference cases.
    """
    if base_vals is None:
        base_vals = {i: np.nan for i in cvts}

    ccomp_df = df.loc[df.index.isin(pd.MultiIndex.from_product([ccomp, ccomp]))]
    ix = pd.IndexSlice

    uniq_vals = np.unique(ccomp_df.loc[:, ix["ref", cvts]], axis=0)
    uniq_vals = uniq_vals[~np.isnan(uniq_vals).any(axis=1), :]

    bvs = np.array([base_vals[c] for c in cvts])

    for i, c in enumerate(cvts):
        if np.isnan(bvs[i]):
            bvs[i] = np.mean(uniq_vals[:, i])

    ref_vals = uniq_vals[np.argmin(np.abs(uniq_vals - bvs[None, :]).sum(axis=1)), :]

    return ccomp_df.loc[(ccomp_df["ref"][cvts] == ref_vals).all(axis=1)]


def get_paired_df(
    sr_df,
    cvt,
    sens_vars,
    base_vals,
    suffixes=["sens", "ref"],
    resp_name="log_icer_usd",
    ratio_id_name="RatioID",
):
    """
    This is the main function and it calls all the other ones

    Args:
        sr_df (df):
        cvt (str or list of strings):
        sens_vars (string or list of strings):
        base_vals (list of strings):
        suffixes (list of strings): Will almost always be set value ['sens', 'ref']
        resp_name (string): the response variable that we are trying to predict
        ratio_id_name (string): Usually ratioID

    Returns:
        paired_df with ....
    """
    # checking data type of input
    if isinstance(cvt, list):
        assert all([isinstance(i, str) for i in cvt])
        cvts = cvt
    else:
        assert isinstance(cvt, str)
        cvts = [cvt]

    pairs = get_all_sr_pairs(sr_df, cvts, sens_vars, suffixes)

    # filtering out sets of ref_sens that don't have a match or contain null covariates
    pairs = filter_cost_saving_pairs(
        pairs, cvts, resp_name=resp_name, suffixes=suffixes, ratio_id_name=ratio_id_name
    )
    # dropping the null icers (i think)
    pairs = pairs[~pairs["all_refs_null_icer"]]

    if pairs.shape[0] == 0:
        return pairs

    ccomps = get_conn_comps(pairs.index)
    pairs = pd.concat(
        [select_ref_one_ccomp(c, pairs, cvts, base_vals) for c in ccomps], axis=0
    )
    pairs = pairs.assign(sens_variable=",".join(cvts))

    return pairs


# designed to capture ratios where all covariates are identical
def identical_covariate_pairs(
    df, sens_vars, id_vars, suffixes=["_sens", "_ref"], ratio_id_name="RatioID"
):
    paired_df = df[[ratio_id_name] + sens_vars + id_vars].merge(
        df[[ratio_id_name] + sens_vars + id_vars],
        on=sens_vars + id_vars,
        how="outer",
        suffixes=suffixes,
    )
    paired_df = paired_df.loc[
        (
            paired_df[ratio_id_name + suffixes[0]]
            != paired_df[ratio_id_name + suffixes[1]]
        )
    ]
    paired_df = paired_df.set_index([ratio_id_name + i for i in suffixes])
    ccomps = get_conn_comps(paired_df.index)
    paired_df = paired_df.loc[
        paired_df.index.get_level_values(0).isin([list(cc)[0] for cc in ccomps])
    ]

    return paired_df[id_vars].reset_index().drop_duplicates()


# creates a list of df
sr_pairs = [
    get_paired_df(
        sr_df,
        i,
        sens_vars + identical_vars,
        base_vals,
        suffixes=["sens", "ref"],
        resp_name="log_icer_usd",
        ratio_id_name="RatioID",
    ).assign(sens_variable=i)
    for i in sens_vars
]

# adds to list of dfs the df where at least one of the two discountrates varies
# checks at least one thing in the list changes, and at least one thing in list differs between sensivity and ref analysis
# also checks that everything else in sens vars and identical_vars don't differ
sr_pairs.append(
    get_paired_df(
        sr_df,
        ["CostsDiscountRate", "DiscountRate"],
        sens_vars + identical_vars,
        base_vals,
        suffixes=["sens", "ref"],
        resp_name="log_icer_usd",
        ratio_id_name="RatioID",
    )
)

# takes a list dfs and stacks them vertically when axis =0, making one big df
sr_pairs = pd.concat(sr_pairs, axis=0)

# adding more rows to df sr_pairs
# goal to have a copy of the df that is for sens analysis for discount rates,
# but want to change the name of the sens_variable
alt_names_sr_pairs_list = [
    sr_pairs[sr_pairs["sens_variable"] == v].assign(sens_variable=k)
    for k, v in alt_cov_versions.items()
]
sr_pairs = pd.concat([sr_pairs] + alt_names_sr_pairs_list, axis=0)

# redefining column names to get rid of the multi-index
sr_pairs.columns = [pref + "_" + suff for suff, pref in sr_pairs.columns]
sr_pairs.columns = sr_pairs.columns.str.replace("^_", "", regex=True)
# sr_pairs contains duplicate columns that add suff for identical_vars
sr_pairs = sr_pairs.reset_index()

# Adding some columns from original df onto the sr_pairs df
sr_pairs = sr_pairs.merge(
    df[
        [
            "PubMedID",
            "ArticleID",
            "diagnostic_treatment_prevention",
            "cause_id",
            "cause",
            "RatioID",
            "intervention_keywords_final",
        ]
    ].rename({"RatioID": "RatioID_sens"}, axis=1),
    on="RatioID_sens",
    how="left",
    validate="many_to_one",
)

# writing paired_df to file
if not os.path.exists("PAIRED_DF.csv"):
    sr_pairs.to_csv("PAIRED_DF.csv", index=False)
    print(f"wrote output to {'PAIRED_DF.csv'}")

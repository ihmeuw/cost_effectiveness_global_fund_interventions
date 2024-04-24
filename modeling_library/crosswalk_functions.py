import numpy as np
import pandas as pd
import sys
import os
import warnings
import mrtool


def create_diff_variables(df, resp_name, se_name, cov_name, suffixes=("_sens", "_ref")):
    """
    This function calculates the difference between the response variable (usually the log_icer)
        and the covariate that has been modified for the sensivity analysis.

    Args:
        df (dataframe): Usually takes in the paired_df
        resp_name (str): Response variable, usually log_icer (column name)
        se_name (str): column name for where the standard error is saved
        cov_names (str or list of strings): Name of covariate that changes in the sensivity analysis

    Returns:
        Dataframe with additional columns that have the suffix _diff
    """
    df[resp_name + "_diff"] = df[resp_name + suffixes[0]] - df[resp_name + suffixes[1]]

    # The standard error of the difference of two independent ratios is the pythagorean
    # sum of those ratios' standard errors, SE_s = sqrt( SE1^2 + SE2^2 )
    df[se_name + "_diff"] = np.sqrt(df[se_name + suffixes[0]] ** 2 + df[se_name + suffixes[1]] ** 2)

    if isinstance(cov_name, list):
        for c in cov_name:
            df[c + "_diff"] = df[c + suffixes[0]] - df[c + suffixes[1]]
    else:
        df[cov_name + "_diff"] = df[cov_name + suffixes[0]] - df[cov_name + suffixes[1]]

    return df


def cwalk(df, resp_name, se_name, cov_name, study_id, interaction_covs=None, monotonicity=None, random_effect=False):
    """
    Fitting crosswalk model using df and using differences in cov_name to explain differences in resp_name.

    Args:
        df (dataframe): usually is the paired_df, filtered down to where col sens_variable = cov_name,
            paired_df includes pairs of sens and reference pairs.
            It needs to have cols with labels that match cov_name, se_name and resp_name with suffix; _diff.
            Also needs a col for study_id wo suffixes.
        resp_name (str): name of response variable, wo suffixes, usually log_icer
        se_name (str): standard error of response variable. This is a fixed value in our current use of MR BRT.
            We would need better information to change this assumption.
        cov_name (str):  name of covariate that running the crosswalk for.
        interaction_covs (list of strings): that we want estimate different slopes for different values of the covariates
        monotonicity (str or None): if it is a string, has to be either 'increasing' or 'decreasing',
            if it is 'increasing' that we want to assume that larger values of cov_name results larger values of resp_name,
            if it is 'decreasing' then we want to assume smaller values of cov_name results in larger values of resp_name.
        random_effect (boolean): True if we want to estimate random slopes for different values of study_id
            (For us in cwalks this is likely to be cause or intervention type).

    Returns:
        MRBRT model object
    """

    cov_dict = {cov_name + "_diff": df[cov_name + "_diff"].to_numpy()}

    if monotonicity is not None:
        if monotonicity == "increasing":
            prior_beta_uniform = np.array([0.0, np.inf])
        elif monotonicity == "decreasing":
            prior_beta_uniform = np.array([-np.inf, 0.0])
    else:
        prior_beta_uniform = None

    cvt_models = [
        mrtool.LinearCovModel("intercept", use_re=False, prior_beta_uniform=np.array([0.0, 0.0])),
        mrtool.LinearCovModel(alt_cov=cov_name + "_diff", use_re=random_effect, prior_beta_uniform=prior_beta_uniform),
    ]

    if interaction_covs is not None:
        cov_dict.update({f"{i}_x_{cov_name}": (df[i] * df[cov_name + "_diff"]).to_numpy() for i in interaction_covs})
        cvt_models.extend(
            [
                mrtool.LinearCovModel(f"{i}_x_{cov_name}", use_re=False, prior_beta_uniform=prior_beta_uniform)
                for i in interaction_covs
            ]
        )

    cwd = mrtool.MRData(
        obs=df[resp_name + "_diff"].to_numpy(),
        obs_se=df[se_name + "_diff"].to_numpy(),
        covs=cov_dict,
        study_id=df[study_id].to_numpy(),
        data_id=df.index.to_numpy(),
    )

    mr = mrtool.MRBRT(data=cwd, cov_models=cvt_models)
    mr.fit_model(inner_max_iter=1000, outer_max_iter=2000)

    return mr


def cwalk_multivar(
    df, resp_name, se_name, cov_names, study_id, interaction_covs=None, monotonicity=None, random_effect=False
):

    cov_dict = {cov_name + "_diff": df[cov_name + "_diff"].to_numpy() for cov_name in cov_names}

    prior_beta_uniform = {"intercept": np.array([0.0, 0.0])}

    if monotonicity is not None:
        for cov_name, monot in monotonicity.items():
            if monot == "increasing":
                prior_beta_uniform.update({cov_name + "_diff": np.array([0.0, np.inf]) for cov_name in cov_names})
            elif monot == "decreasing":
                prior_beta_uniform.update({cov_name + "_diff": np.array([-np.inf, 0.0]) for cov_name in cov_names})

    cvt_models = [mrtool.LinearCovModel("intercept", use_re=False, prior_beta_uniform=np.array([0.0, 0.0]))]

    cvt_models = cvt_models + [
        mrtool.LinearCovModel(
            alt_cov=cov_name + "_diff", use_re=random_effect, prior_beta_uniform=prior_beta_uniform.get(cov_name)
        )
        for cov_name in cov_names
    ]

    if interaction_covs is not None:
        cov_dict.update(
            {
                f"{i}_x_{cov_name}": (df[i] * df[cov_name + "_diff"]).to_numpy()
                for i in interaction_covs
                for cov_name in cov_names
            }
        )
        cvt_models.extend(
            [
                mrtool.LinearCovModel(
                    f"{i}_x_{cov_name}", use_re=False, prior_beta_uniform=prior_beta_uniform.get(cov_name)
                )
                for i in interaction_covs
                for cov_name in cov_names
            ]
        )

    cwd = mrtool.MRData(
        obs=df[resp_name + "_diff"].to_numpy(),
        obs_se=df[se_name + "_diff"].to_numpy(),
        covs=cov_dict,
        study_id=df[study_id].to_numpy(),
        data_id=df.index.to_numpy(),
    )

    mr = mrtool.MRBRT(data=cwd, cov_models=cvt_models)
    mr.fit_model(inner_max_iter=1000, outer_max_iter=2000)

    return mr


def summarize_cwalk(mr, cov_name, n_samples=1000, seed=24601):
    """
    Creates a df with one column for each of several of statistics about model fit and parameters.

    Args:
        mr (model object): This is output from cwalk function
        cov_name (string or list of strings): covariate of the sens analysis
        n_samples (integer): Usually 1000, but this can be changed if wanted
        seed (integer): Specified here, so as to help our estimates not change too much due to changing random sample starting at a different value

    Returns:
        Dataframe with model statistics
    """

    if seed is not None:
        np.random.seed(seed)

    beta_samples, _ = mr.sample_soln(sample_size=n_samples)

    # for handling potential differences in cov_name - either a string or list of strings
    if isinstance(cov_name, str):
        cov_names = [cov_name]
    elif isinstance(cov_name, list):
        cov_names = cov_name

    par_summary = pd.DataFrame(
        {
            "beta": mr.beta_soln[1:],
            "se_beta": beta_samples[:, 1:].std(axis=0),
            "sample_size": np.full(len(cov_names), mr.data.obs.shape[0]),
        },
        index=cov_names,
    )

    if (mr.gamma_soln != 0).any():
        par_summary = par_summary.assign(**mr.re_soln)
        par_summary = par_summary.rename({v: "beta_" + v for v in mr.re_soln.keys()}, axis=1)

        par_summary["gamma"] = mr.gamma_soln

        par_summary["covariate"] = mr.cov_model_names[1:]

        # returns beta_samples, but might not ever use the beta_samples from crosswalks
    return (beta_samples[:, 1:], par_summary)


def compare_sd_versions(mr, beta_samples=None, n_samples=1000, seed=24601):
    if beta_samples is None:
        beta_samples, _ = mr.sample_soln(sample_size=n_samples)
        beta_samples = beta_samples[:, 1]

    lme_specs = mrtool.core.other_sampling.extract_simple_lme_specs(mr)
    hessn = mrtool.core.other_sampling.extract_simple_lme_hessian(lme_specs)
    sigma = np.linalg.inv(hessn)

    f_info_se = np.sqrt(sigma[1, 1] - (sigma[0, 1] ** 2 / sigma[0, 0]))
    resample_se = beta_samples.std()

    return {"Fisher Information": f_info_se, "Fit-Refit": resample_se}

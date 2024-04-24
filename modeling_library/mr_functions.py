import os
import warnings
import dill
import pickle as pkl
import logging

import sys
import numpy as np
import pandas as pd
import mrtool

log = logging.getLogger()

# ========
# STEP 4: COVARIATE SELECTION


def select_covariates(df, candidate_covs, include_covs, resp_name, se_name, study_id_name, beta_gprior=None):
    """
    Selects and covariates from candidate_covs and returns them as a list.
    Mixture of certain covariates that are specified by the team i.e. ones that have crosswalk information
    Some covariates determined using lasso covariate selection through MRTOOL

    Args:
        df (pd.DataFrame): the data.
        candidate_covs (list of strings): each of which is the column index
            of a column of df whose inclusion in the model will be determined
            by lasso.
        include_covs (list of strings): each of which is the column index in df
            of a covariate that will definitely be included in the model.
        resp_name (string): denoting the column index in df of the response.
        se_name (string): denotes the column index in df of the SE of the response.
        study_id_name (string): denotes the column index in df of the grouping variable
            that defines groups for random effects.
    Returns:
        list of strings denoting covariates selected by CovFinder.
    """

    if not any([v == "intercept" for v in include_covs]):
        include_covs = include_covs + ["intercept"]
    if not any([v == "intercept" for v in df.columns.values]):
        df["intercept"] = np.ones(df.shape[0])

    norm_df = df.copy()
    for cov in candidate_covs:
        norm_df[cov] = (df[cov] - df[cov].mean()) / df[cov].std()

    all_covs = include_covs + candidate_covs

    mrd = mrtool.MRData(
        obs=norm_df[resp_name].to_numpy(),
        obs_se=norm_df[se_name].to_numpy(),
        covs={v: norm_df[v].to_numpy() for v in all_covs},
        study_id=norm_df[study_id_name].to_numpy(),
    )

    cfinder = mrtool.CovFinder(
        data=mrd,
        covs=candidate_covs,
        pre_selected_covs=include_covs,
        beta_gprior=beta_gprior,
        normalized_covs=True,
        num_samples=5000,
        laplace_threshold=1e-5,
        power_range=(-8, 8),
        power_step_size=0.5,
    )

    cfinder.select_covs(verbose=True)

    # returns the selected covs as list
    return cfinder.selected_covs


# ===========
# STEP 5: MIXED EFFECTS MODEL


def fit_with_covs(
    df,
    covs,
    resp_name,
    se_name,
    study_id_name,
    data_id_name=None,
    z_covs=["intercept"],
    trim_prop=0.0,
    spline_cov=None,
    spline_degree=None,
    spline_knots_type=None,
    spline_knots=None,
    spline_monotonicity=None,
    ensemble=False,
    full_spline_basis=False,
    # Do specify from here down
    gprior_dict=None,
    uprior_dict=None,
    inner_max_iter=2000,
    outer_max_iter=1000,
):
    """
    Fits a model using the specified covariates.
    This is used in the final step. The final models.
    It includes random_effects on AritcleID

    Args:
        df (pd.DataFrame): the data.

        covs (list of strings): names of columns of df containing the model covariates

        resp_name (string): name of the column of df containing the response

        se_name (string): name of the column of df containing the SE of the response

        study_id_name (string): name of the column of df that defines random effect groups

        z_covs (list of strings): list of elements of covs that have random effects

        trim_prop (float between 0.0 and 1.0): proportion of data points to trim

        spline_cov (string or None): If a string, then it is the covariate fit with a
            spline. If None, then it indicates that no spline should be fit

        spline_degree (int > 0): the degree of the spline. None if spline_cov is None

        spline_knots_type (string, either 'domain' or 'frequency'):

        spline_knots (np.ndarray): location of knots either as quantiles of the spline
            covariate (if spline_knots_type == 'frequency') or as the proportion of the
            distance from the min to the max (if spline_knots_type == 'domain')

        ensemble (boolean): whether to use a knot-ensemble. Default False, not currently
            working for True.

        gprior_dict (None or dictionary of np.arrays or lists of np.arrays): dictionary
            whose keys are the entries of covs and whose values specify priors to be
            used for the covariates (**crosswalk priors**). Priors for spline_cov are specified using a numpy.array
            of shape (2, k) where k is the number of parameters estimated for spline_cov.
            Priors for covariates that are not in z_covs must be specified using a np.array
            or shape (2,) whose entries are the mean and sd of the gaussian prior, respectively.
            Priors for z_covs must be lists of length 2 whose entries are numpy arrays of
            shape (2,).

        uprior_dict (None or dictionary of np.arrays or lists of np.arrays): dictionary.
            We are not using at this point. If there is a covariate we want to assume have a positive or relationship
            with the ICER then we can input information in uprior to constrain the relationship of the covariate
            to the ICER (either be positive or negative)

        inner_max_iter (int): maximum number of iterations for the inner loop.

        outer_max_iter (int): maximum number of iterations for the outer loop.

    Returns:
        MRBRT object

    """

    z_covs = list(set(z_covs + ["intercept"]))
    if not any(df.columns.values == "intercept"):
        df["intercept"] = 1.0
    elif not all(df["intercept"] == 1.0):
        sys.exit("Columns of df labeled intercept must only have entries of 1.0")

    if full_spline_basis:
        if uprior_dict is None:
            uprior_dict = {}
        uprior_dict.update({"intercept": np.array([0.0, 0.0])})

    if data_id_name is None:
        data_id_name = "DataID"
        df = df.assign(DataID=np.arange(df.shape[0]))

    mrd = mrtool.MRData(
        obs=df[resp_name].to_numpy(),
        obs_se=df[se_name].to_numpy(),
        covs={v: df[v].to_numpy() for v in covs},
        study_id=df[study_id_name].to_numpy(),
        data_id=df[data_id_name].to_numpy(),
    )

    cov_model_args = {v: {"alt_cov": v, "use_re": (v in z_covs)} for v in covs if v != spline_cov}

    if gprior_dict is not None:
        for key, val in gprior_dict.items():
            if key in z_covs:
                cov_model_args[key].update({"prior_beta_gaussian": val[0], "prior_gamma_gaussian": val[1]})
            else:
                cov_model_args[key].update({"prior_beta_gaussian": val})

    if uprior_dict is not None:
        for key, val in uprior_dict.items():
            cov_model_args[key].update({"prior_beta_uniform": val})

    cov_model_list = [mrtool.LinearCovModel(**x) for x in cov_model_args.values()]

    if spline_cov is not None:
        n_knots = spline_knots.shape[0] if spline_degree is not None else None
        spline_degree = int(spline_degree)
        prior_spline_maxder = None if gprior_dict is None else gprior_dict.get(spline_cov)
        spline_cov_model = mrtool.LinearCovModel(
            spline_cov,
            use_re=False,
            use_spline=True,
            spline_degree=spline_degree,
            spline_knots_type=spline_knots_type,
            spline_knots=spline_knots,
            spline_r_linear=(spline_degree > 1),
            spline_l_linear=(spline_degree > 1),
            use_spline_intercept=full_spline_basis,
            prior_spline_monotonicity=spline_monotonicity,
            prior_spline_maxder_gaussian=prior_spline_maxder,
        )

        cov_model_list.append(spline_cov_model)

    if spline_cov is not None or not ensemble:
        mr = mrtool.MRBRT(data=mrd, cov_models=cov_model_list, inlier_pct=1 - trim_prop)
        mr.fit_model(inner_print_level=5, inner_max_iter=inner_max_iter, outer_max_iter=outer_max_iter)

    return mr


# =============== STEP 5: MIXED EFFECTS MODEL FIT ===============
def summarize_parameters(mr, spline_cov=None, beta_samples=None, num_draws=10000):
    cov_names = [v + "_" + str(i) for v in mr.fe_soln.keys() for i in range(mr.fe_soln.get(v).shape[0])]
    for i in range(len(cov_names)):
        if cov_names[i][:-2] != spline_cov:
            cov_names[i] = cov_names[i][:-2]
        betas = np.concatenate(list(mr.fe_soln.values()))
    use_draws_method = beta_samples is not None

    if use_draws_method:
        beta_ses = beta_samples.std(axis=0)
        beta_var = beta_ses**2
    else:
        lme_specs = mrtool.core.other_sampling.extract_simple_lme_specs(mr)
        hessn = mrtool.core.other_sampling.extract_simple_lme_hessian(lme_specs)

        beta_var = np.diag(np.linalg.inv(hessn))
        beta_ses = np.sqrt(beta_var)

    summary_df = pd.DataFrame(
        [
            (mr.cov_model_names[i], bta, bta_se)
            for i, x_var_idx in enumerate(mr.x_vars_indices)
            for (bta, bta_se) in zip(mr.beta_soln[x_var_idx], beta_ses[x_var_idx])
        ],
        columns=["covariate", "beta", "beta_se"],
    )

    summary_df["beta_variance"] = summary_df["beta_se"] ** 2
    summary_df["gamma"] = np.concatenate(
        [
            (
                mr.gamma_soln[mr.z_vars_indices[mr.get_cov_model_index(cov_name)]]
                if mr.cov_models[mr.get_cov_model_index(cov_name)].use_re
                else np.repeat(np.nan, mr.cov_models[mr.get_cov_model_index(cov_name)].num_x_vars)
            )
            for cov_name in mr.cov_model_names
        ]
    )
    return summary_df


def summarize_parameters_no_spline(mr, spline_cov=None, beta_samples=None, num_draws=10000):
    cov_names = [v + "_" + str(i) for v in mr.fe_soln.keys() for i in range(mr.fe_soln.get(v).shape[0])]
    use_draws_method = beta_samples is not None

    if use_draws_method:
        beta_ses = beta_samples.std(axis=0)
        beta_var = beta_ses**2
    else:
        lme_specs = mrtool.core.other_sampling.extract_simple_lme_specs(mr)
        hessn = mrtool.core.other_sampling.extract_simple_lme_hessian(lme_specs)

        beta_var = np.diag(np.linalg.inv(hessn))
        beta_ses = np.sqrt(beta_var)

    summary_df = pd.DataFrame(
        [
            (mr.cov_model_names[i], bta, bta_se)
            for i, x_var_idx in enumerate(mr.x_vars_indices)
            for (bta, bta_se) in zip(mr.beta_soln[x_var_idx], beta_ses[x_var_idx])
        ],
        columns=["covariate", "beta", "beta_se"],
    )

    summary_df["beta_variance"] = summary_df["beta_se"] ** 2
    summary_df["gamma"] = np.concatenate(
        [
            (
                mr.gamma_soln[mr.z_vars_indices[mr.get_cov_model_index(cov_name)]]
                if mr.cov_models[mr.get_cov_model_index(cov_name)].use_re
                else np.repeat(np.nan, mr.cov_models[mr.get_cov_model_index(cov_name)].num_x_vars)
            )
            for cov_name in mr.cov_model_names
        ]
    )
    return summary_df


# =================
# STEP 6: PREDICTIONS
def predict(pred_df, mr):
    """
    Predicts response values for the data frame pred_df using the model fit mr.
    """
    pred_mrdata = mrtool.MRData(covs={v: pred_df[v].to_numpy() for v in mr.cov_model_names})
    preds = mr.predict(data=pred_mrdata, predict_for_study=False)
    return preds


# ================
# STEP 4: GUASSIAN PRIORS


def k_fold_cv_gaussian_prior(
    k,
    df,
    resp_name,
    se_name,
    covs,
    data_id_name,
    study_id_name,
    constrained_covs=None,
    beta_gpriors=None,
    combine_gpriors=False,
    fold_id_name=None,
    initial_upper_prior_sd=1.0,
    inner_max_iter=2000,
    outer_max_iter=1000,
    sd_tol=1e-6,
    num_sds_per_step=10,
    dev=False,
):
    np.random.seed(1)

    df["fold_id"] = np.random.randn(df.shape[0])

    df["fold_id"] = pd.qcut(df["fold_id"], k, labels=list(range(k)))
    return cv_gaussian_prior(
        df,
        resp_name,
        se_name,
        covs,
        data_id_name,
        study_id_name,
        constrained_covs=constrained_covs,
        beta_gpriors=beta_gpriors,
        combine_gpriors=combine_gpriors,
        initial_upper_prior_sd=initial_upper_prior_sd,
        fold_id_name="fold_id",
        inner_max_iter=inner_max_iter,
        outer_max_iter=outer_max_iter,
        sd_tol=sd_tol,
        num_sds_per_step=num_sds_per_step,
        dev=dev,
    )


# ====================
# feeds into function above
# STEP 4 CONTINUED: GUASSIAN PRIORS
def cv_gaussian_prior(
    df,
    resp_name,
    se_name,
    covs,
    data_id_name,
    study_id_name,
    constrained_covs=None,
    beta_gpriors=None,
    combine_gpriors=False,
    fold_id_name=None,
    initial_upper_prior_sd=1.0,
    inner_max_iter=2000,
    outer_max_iter=1000,
    sd_tol=1e-6,
    num_sds_per_step=10,
    dev=False,
):

    if not any(df.columns.values == "intercept"):
        df["intercept"] = 1.0
    elif not all(df["intercept"] == 1.0):
        sys.exit("Columns of df labeled intercept must only have entries of 1.0")

    if fold_id_name is None:
        fold_id_name = data_id_name

    stdized_df = df[[fold_id_name, data_id_name, study_id_name, resp_name, se_name] + covs].copy()
    log.info(f"covs into gaus function {covs}")
    log.info(f"shape of df with all covs: {stdized_df.shape}")

    if constrained_covs is None:
        constrained_covs = {}

    unstdized_covs = ["intercept"] + list(constrained_covs.keys())
    if beta_gpriors is not None:
        if combine_gpriors:
            covs_to_stdize = list(set(covs) - set(unstdized_covs))
            for v in beta_gpriors.keys():
                if v in covs_to_stdize:
                    beta_gpriors[v] = beta_gpriors[v] * stdized_df[v].std()
        else:
            unstdized_covs = unstdized_covs + list(beta_gpriors.keys())
            unstdized_covs = list(set(unstdized_covs))
            covs_to_stdize = list(set(covs) - set(unstdized_covs))
    else:
        covs_to_stdize = list(set(covs) - set(unstdized_covs))
        beta_gpriors = {}

    stdized_df[covs_to_stdize] = (stdized_df[covs_to_stdize] - stdized_df[covs_to_stdize].mean(axis=0)) / stdized_df[
        covs_to_stdize
    ].std(axis=0)

    log.info(f"standardized df shape {stdized_df[covs_to_stdize].shape}")

    # Create a list of pandas series for the train-test splitting.
    mask_list = [stdized_df[fold_id_name] == fid for fid in stdized_df[fold_id_name].unique()]

    # Use the list of masks to create a list of tuples of MRData objects.
    train_test_mrd_list = [
        tuple(
            mrtool.MRData(
                obs=stdized_df.loc[m == truth_val, resp_name].values,
                obs_se=stdized_df.loc[m == truth_val, se_name].values,
                covs={v: stdized_df.loc[m == truth_val, v].values for v in covs},
                study_id=stdized_df.loc[m == truth_val, study_id_name].values,
                data_id=stdized_df.loc[m == truth_val, data_id_name].values,
            )
            for truth_val in [False, True]
        )
        for m in mask_list
    ]
    if not combine_gpriors:
        gprior_cov_models = [
            mrtool.LinearCovModel(key, use_re=False, prior_beta_gaussian=val) for key, val in beta_gpriors.items()
        ]

    # Initialize empty numpy arrays for the prior SDs & mses
    prior_sds = np.array([], dtype=np.float64)
    mses = np.array([], dtype=np.float64)

    # Initialize bounds for the set of prior SDs for a single iteration
    lower_prior_sd = 1e-4
    upper_prior_sd = initial_upper_prior_sd

    # Create a copy of the number of prior SDs per step variable.
    n_sds_per_step = num_sds_per_step
    if dev:
        import time
    while upper_prior_sd - lower_prior_sd > sd_tol:
        if dev:
            tm = time.time()

        # Generate new values of the prior SD, evenly spaced on a log-scale
        new_prior_sds = np.geomspace(lower_prior_sd, upper_prior_sd, n_sds_per_step)

        if n_sds_per_step == num_sds_per_step:
            n_sds_per_step += 2
        new_prior_sds = np.setdiff1d(new_prior_sds, prior_sds)

        if combine_gpriors:
            gprior_cov_model_lists = [
                [
                    mrtool.LinearCovModel(
                        key,
                        use_re=False,
                        prior_beta_gaussian=np.array(
                            [val[0] / (1 + (val[1] ** 2 / sd**2)), val[1] * sd / np.sqrt(val[1] ** 2 + sd**2)]
                        ),
                    )
                    for key, val in beta_gpriors.items()
                ]
                for sd in new_prior_sds
            ]
        else:
            gprior_cov_model_lists = [gprior_cov_models for sd in new_prior_sds]

        const_cov_model_lists = [
            [
                mrtool.LinearCovModel(key, use_re=False, prior_beta_uniform=np.full((2,), val))
                for key, val in constrained_covs.items()
            ]
            for sd in new_prior_sds
        ]

        cov_model_lists = [
            [mrtool.LinearCovModel("intercept", use_re=True)]
            + [
                mrtool.LinearCovModel(v, use_re=False, prior_beta_gaussian=np.array([0, sd]))
                for v in covs
                if v not in ["intercept"] + list(constrained_covs.keys()) + list(beta_gpriors.keys())
            ]
            + gprior_cov_model_lists[i]
            + const_cov_model_lists[i]
            for i, sd in enumerate(new_prior_sds)
        ]

        print(train_test_mrd_list)
        print(cov_model_lists)

        new_mses = np.array(
            [
                np.array(
                    [get_mse(train_mrd, test_mrd, cmod_list) for train_mrd, test_mrd in train_test_mrd_list]
                ).mean()
                for cmod_list in cov_model_lists
            ]
        )

        prior_sds = np.hstack([prior_sds, new_prior_sds])
        mses = np.hstack([mses, new_mses])
        ordr = np.argsort(prior_sds)
        prior_sds = prior_sds[ordr]
        mses = mses[ordr]

        min_index = np.argmin(mses)
        lower_prior_sd = prior_sds[min_index - 2 if min_index > 1 else 0]
        upper_prior_sd = prior_sds[min_index + 2 if min_index < prior_sds.shape[0] - 2 else prior_sds.shape[0] - 1]

        if dev:
            print("iteration took " + str(time.time() - tm))
            print("just ran for " + str(new_mses.shape[0]) + " values of lambda")
    return (prior_sds, mses)


# ======== STEP 4:  GUASSIAN PRIORS CONTINUED ========
# Used in above function (step 4 still)
def get_mse(train_mrd, test_mrd, cov_model_list, inner_max_iter=1000, outer_max_iter=2000):

    mr = mrtool.MRBRT(data=train_mrd, cov_models=cov_model_list)
    mr.fit_model(
        inner_print_level=5,
        inner_max_iter=inner_max_iter,
        outer_max_iter=outer_max_iter,
    )
    preds = mr.predict(test_mrd, predict_for_study=False)
    mse = ((preds - test_mrd.obs) ** 2).mean()
    return mse


# =============== STEP 2: FITTING SPLINES ===============
# This does implement ensembles
def fit_signal_model(
    df,
    resp_name,
    se_name,
    spline_cov,
    study_id_name,
    data_id_name,
    other_cov_names,
    other_cov_gpriors=None,
    h=0.1,
    num_samples=20,
    deg=2,
    n_i_knots=2,
    knots_type="frequency",  # domain
    prior_spline_monotonicity=None,
    knot_bounds=np.array([[0.1, 0.6], [0.4, 0.9]]),
    interval_sizes=np.array([[0.1, 0.7], [0.1, 0.7], [0.1, 0.7]]),
):
    """
    Fits a signal model calculating spline for one variable and applying a linear regression model to others.

    Returns:
        Metaregression object saved to a pickle file, which can be used to calculate spline values for model fit and icer predictions
    """

    np.random.seed(1)

    cov_dict = {spline_cov: df[spline_cov].to_numpy()}
    print(other_cov_names)
    cov_dict = {w: df[w].to_numpy() for w in other_cov_names + [spline_cov]}

    covs_with_prior = list(other_cov_gpriors.keys())

    cov_model_list = []
    for v in other_cov_names:
        if v in covs_with_prior:
            cov_model_list.append(mrtool.LinearCovModel(v, use_re=False, prior_beta_gaussian=other_cov_gpriors[v]))
        else:
            cov_model_list.append(mrtool.LinearCovModel(v, use_re=False))

    mrd = mrtool.MRData(
        obs=df[resp_name].to_numpy(),
        obs_se=df[se_name].to_numpy(),
        study_id=df[study_id_name].to_numpy(),
        data_id=df[data_id_name].to_numpy(),
        covs=cov_dict,
    )

    ensemble_cov_model = mrtool.LinearCovModel(
        alt_cov=spline_cov,
        use_spline=True,
        spline_degree=deg,
        spline_knots_type=knots_type,
        spline_knots=np.linspace(0.0, 1.0, n_i_knots + 2),
        spline_r_linear=True,
        spline_l_linear=True,
        use_spline_intercept=False,
        # this is variable depending on experiment
        prior_spline_monotonicity=prior_spline_monotonicity,
        use_re=False,
    )
    knot_samples = mrtool.core.utils.sample_knots(
        num_intervals=n_i_knots + 1, knot_bounds=knot_bounds, interval_sizes=interval_sizes, num_samples=num_samples
    )

    signal_mr = mrtool.MRBeRT(
        data=mrd,
        ensemble_cov_model=ensemble_cov_model,
        ensemble_knots=knot_samples,
        cov_models=cov_model_list,
        inlier_pct=1.0 - h,
    )

    signal_mr.fit_model(inner_max_iter=1000, outer_max_iter=2000)

    return signal_mr


# used this as a trial
def fit_signal_model_not_ensemble(
    df,
    resp_name,
    se_name,
    spline_cov,
    study_id_name,
    data_id_name,
    other_cov_names,
    other_cov_gpriors=None,
):
    """
    Fits a signal model calculating spline wo ensemble for one variable and applying a linear regression model to others.

    Returns:
        Metaregression object saved to a pickle file, which can be used to calculate spline values for model fit and icer predictions
    """

    np.random.seed(1)

    cov_dict = {spline_cov: df[spline_cov].to_numpy()}
    cov_dict = {w: df[w].to_numpy() for w in other_cov_names + [spline_cov]}

    covs_with_prior = list(other_cov_gpriors.keys())

    cov_model_list = []
    for v in other_cov_names:
        if v in covs_with_prior:
            cov_model_list.append(mrtool.LinearCovModel(v, use_re=False, prior_beta_gaussian=other_cov_gpriors[v]))
        else:
            cov_model_list.append(mrtool.LinearCovModel(v, use_re=False))

    mrd = mrtool.MRData(
        obs=df[resp_name].to_numpy(),
        obs_se=df[se_name].to_numpy(),
        study_id=df[study_id_name].to_numpy(),
        data_id=df[data_id_name].to_numpy(),
        covs=cov_dict,
    )

    signal_mr = mrtool.MRBRT(
        data=mrd,
        cov_models=cov_model_list
        + mrtool.LinearCovModel(
            alt_cov=spline_cov,
            use_spline=True,
            spline_knots=np.array([0, 0.25, 0.5, 0.75, 1]),
            spline_degree=2,
            spline_knots_type="frequency",
            spline_r_linear=True,
            spline_l_linear=True,
        ).tolist(),
    )

    signal_mr.fit_model(inner_max_iter=1000, outer_max_iter=2000)

    return signal_mr


# =============== STEP 2: FITTING SPLINES CONTINUED ===============


# Create signal is used twice in entire process once to create new spline covariates for model
# (training data set) and one for the predictions
def create_signal(signal_mr, spline_cov, spline_cov_values, data_id_name, data_ids):
    """
    Takes a signal_mr object and df, and adds a column to the df new_spline_cov column, which includes transformed log_gdp.

    """

    pred_covs = {spline_cov: spline_cov_values}
    cov_model_names = [x.name for x in signal_mr.cov_models]
    pred_covs.update({v: np.zeros(spline_cov_values.shape[0]) for v in cov_model_names if v != spline_cov})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_mrd = mrtool.MRData(
            obs=np.ones(data_ids.shape),
            obs_se=np.ones(data_ids.shape),
            study_id=np.arange(data_ids.shape[0]),
            data_id=data_ids,
            covs=pred_covs,
        )

    new_spline_cov = signal_mr.predict(pred_mrd, predict_for_study=False)

    pred_df = pred_mrd.to_df().assign(new_spline_cov=new_spline_cov, data_id=pred_mrd.data_id)
    pred_df = pred_df[["data_id", "new_spline_cov"]]
    pred_df = pred_df.rename(
        {
            "data_id": data_id_name,
        },
        axis=1,
    )
    return pred_df


# =============== STEP 2: FITTING SPLINES CONTINUED ===============


def get_ws(signal_mr, data_id_name):
    """
    This to trim outliers
    The w values that MR BRT returns are continuous numbers btw 0-1,
    most of data points should be if inliers =90% are 1, approx 9.5% 0, and others
    may be in between 0-1
    """
    w = np.hstack([mdl.w_soln[:, None] for mdl in signal_mr.sub_models]).dot(signal_mr.weights)
    pred_df = signal_mr.data.to_df().assign(**{data_id_name: signal_mr.data.data_id, "w": w})
    pred_df = pred_df[[data_id_name, "w"]]
    pred_df = pred_df.rename(
        {
            "data_id": data_id_name,
        },
        axis=1,
    )
    return pred_df


# ===============
# STEP 5: Mixed effects model


def create_fit_df(mr, df, resp_name, study_id_name, other_id_col_names, data_id_name):
    """
    Wsed by r2 and maybe for summarize_parameters
    Here we create two versions of fitted values
    'fitted_fe_only' - doesn't include re
    'fitted_fe_and_re' - does include re
    """

    fit_mrd = mrtool.MRData(
        obs=np.zeros((df.shape[0],)),
        obs_se=np.zeros((df.shape[0],)),
        covs={v: df[v].to_numpy() for v in mr.data.covs.keys()},
        study_id=df[study_id_name].to_numpy(),
        data_id=df[data_id_name].to_numpy(),
    )
    fit_df = fit_mrd.to_df().rename({"study_id": study_id_name}, axis=1)
    fit_df[data_id_name] = fit_mrd.data_id
    fit_df = fit_df.merge(
        df[[study_id_name, data_id_name, resp_name] + other_id_col_names], on=[study_id_name, data_id_name]
    )

    fit_df["fitted_fe_only"] = mr.predict(data=fit_mrd, predict_for_study=False)
    fit_df["fitted_fe_and_re"] = mr.predict(data=fit_mrd, predict_for_study=True)

    fit_df = fit_df[
        [study_id_name, data_id_name] + other_id_col_names + [resp_name, "fitted_fe_only", "fitted_fe_and_re"]
    ]

    return fit_df


# =============== STEP 5: MIXED EFFECTS MODEL FIT ===============


def r2(mr, fit_df, resp_name):
    """
    Calculates the R^2 values for the model fit.
    """
    rmses = fit_df[[resp_name, "fitted_fe_and_re", "fitted_fe_only"]].copy()
    rmses["fitted_fe_and_re"] = rmses["fitted_fe_and_re"] - rmses[resp_name]
    rmses["fitted_fe_only"] = rmses["fitted_fe_only"] - rmses[resp_name]

    rmses = np.sqrt(rmses[["fitted_fe_and_re", "fitted_fe_only"]].var(axis=0))
    rmses.name = "RMSE"

    r2s = (fit_df[[resp_name, "fitted_fe_and_re", "fitted_fe_only"]].corr() ** 2).loc[
        resp_name, ["fitted_fe_and_re", "fitted_fe_only"]
    ]
    r2s.name = "R_squared"
    r2s = pd.DataFrame(r2s, columns=["R_squared"])
    r2s = r2s.join(rmses)
    r2s.loc["fitted_fe_only", "Sample_size"] = fit_df.shape[0]

    return r2s


# =============== STEP 6 PREDICTIONS ===============
def create_predictions(
    mr,
    signal_mr,
    preds_df,
    resp_name,
    se_name,
    selected_covs,
    study_id_name,
    data_id_name,
    beta_samples=None,
    n_samples=1000,
    seed=24601,
):

    preds_df["idx"] = np.arange(preds_df.shape[0])
    spline_cov = signal_mr.ensemble_cov_model_name
    cov_model_names = [x.name for x in signal_mr.cov_models]

    if "new_spline_cov" not in preds_df.columns:
        signal_preds_df = create_signal(
            signal_mr, spline_cov, preds_df[spline_cov].to_numpy(), "idx", preds_df["idx"].to_numpy()
        )
        preds_df = preds_df.merge(signal_preds_df, on="idx")

    if any([v not in preds_df.columns for v in selected_covs]):
        print([v for v in selected_covs if v not in preds_df.columns])
        print(spline_cov)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds_mrd = mrtool.MRData(
            obs=np.zeros((preds_df.shape[0],)),
            obs_se=np.zeros((preds_df.shape[0],)),
            data_id=preds_df["idx"].to_numpy(),
            study_id=preds_df["idx"].to_numpy(),
            covs={v: preds_df[v].to_numpy() for v in selected_covs},
        )

    # Predictions on the log scale
    preds_df["predicted_" + resp_name] = mr.predict(preds_mrd, predict_for_study=False, sort_by_data_id=True)

    np.random.seed(seed)
    if beta_samples is None:
        beta_samples = mrtool.core.other_sampling.sample_simple_lme_beta(n_samples, mr)
    preds_draws = mr.create_draws(
        data=preds_mrd, beta_samples=beta_samples, gamma_samples=np.full((beta_samples.shape[0], 1), mr.gamma_soln)
    )

    ci_preds = np.quantile(preds_draws, [0.5, 0.025, 0.975, 0.25, 0.75, 0.05, 0.95], axis=1).T

    ci_suffixes = ["_median", "_lower", "_upper", "_25th_per", "_75th_per", "_90_lower", "_90_upper"]
    preds_df[["predicted_" + resp_name + v for v in ci_suffixes]] = ci_preds

    preds_df["predicted_" + resp_name.replace("log_", "")] = np.exp(preds_draws).mean(axis=1)
    log_columns = ["predicted_" + resp_name + v for v in ci_suffixes]
    lin_columns = [v.replace("log_", "") for v in log_columns]
    preds_df[lin_columns] = np.exp(preds_df[log_columns])

    preds_df = preds_df.drop("idx", axis=1)
    preds_df = preds_df.reset_index()

    return preds_df


def create_predictions_no_spline(
    mr,
    preds_df,
    resp_name,
    se_name,
    selected_covs,
    study_id_name,
    data_id_name,
    beta_samples=None,
    n_samples=1000,
    seed=24601,
):

    preds_df["idx"] = np.arange(preds_df.shape[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds_mrd = mrtool.MRData(
            obs=np.zeros((preds_df.shape[0],)),
            obs_se=np.zeros((preds_df.shape[0],)),
            data_id=preds_df["idx"].to_numpy(),
            study_id=preds_df["idx"].to_numpy(),
            covs={v: preds_df[v].to_numpy() for v in selected_covs},
        )

    preds_df["predicted_" + resp_name] = mr.predict(preds_mrd, predict_for_study=False, sort_by_data_id=True)

    np.random.seed(seed)
    if beta_samples is None:
        beta_samples = mrtool.core.other_sampling.sample_simple_lme_beta(n_samples, mr)
    preds_draws = mr.create_draws(
        data=preds_mrd, beta_samples=beta_samples, gamma_samples=np.full((beta_samples.shape[0], 1), mr.gamma_soln)
    )

    ci_preds = np.quantile(preds_draws, [0.5, 0.025, 0.975, 0.05, 0.95], axis=1).T

    ci_suffixes = ["_median", "_lower", "_upper", "_90_lower", "_90_upper"]
    preds_df[["predicted_" + resp_name + v for v in ci_suffixes]] = ci_preds

    preds_df["predicted_" + resp_name.replace("log_", "")] = np.exp(preds_draws).mean(axis=1)
    log_columns = ["predicted_" + resp_name + v for v in ci_suffixes]
    lin_columns = [v.replace("log_", "") for v in log_columns]
    preds_df[lin_columns] = np.exp(preds_df[log_columns])

    preds_df = preds_df.drop("idx", axis=1)
    preds_df = preds_df.reset_index()

    return preds_df

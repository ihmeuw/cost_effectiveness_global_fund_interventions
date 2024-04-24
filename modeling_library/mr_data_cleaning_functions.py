# Includes functions used in the modeling code steps, to clean
# and transform the datasets to be used in the various modeling functions

import pandas as pd
import numpy as np
import os


def exporting_csv(df, filepath):
    """
    Writes csv to file, but checks if it exists first

    :param df: Dataframe to be exported to csv
    :param filepath: A filepath, which will usually be in the format of f"{pth.HIV_MALARIA}/shared_functions/dalys_2010.csv"
    """
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
        print("Wrote file to " + filepath)
    else:
        print("Output path already exists.")


def drops_kw_not_meeting_criteria_flexible_column(df_modeling, column_to_count):
    """
    This function counts articles and ratios by intervention keywords, and returns a list of intervention keywords that do not meet
    the 2 articles, 3 icer criteria for inclusion in the analysis. This list of keywords can then be dropped from the dataset to be modeled

    :param df_modeling: Dataframe that contains all icers for a specific cause intervention combination i.e. for malaria_prevention.
                        Needs to have columns: intervention_keywords_final, ArticleID, RatioID
    :return: Dataframe that drops any intervention keywords that do not meet the 2 articles, 3 icer requirements

    """

    kw_drop_list = []

    article_ratio_counts = df_modeling.groupby(column_to_count)[["ArticleID", "RatioID"]].agg("nunique").reset_index()

    # drop rows that have only 1 Article
    kw_to_drop_df_articles = article_ratio_counts[article_ratio_counts["ArticleID"] <= 1]
    kw_drop_list.extend(list(kw_to_drop_df_articles[column_to_count].unique()))

    # drop rows that have only 2 or less ratios
    kw_to_drop_df_ratios = article_ratio_counts[article_ratio_counts["RatioID"] <= 2]
    kw_drop_list.extend(list(kw_to_drop_df_ratios[column_to_count]))

    # # using set to get just the unique
    kw_drop_list = list(set(kw_drop_list))

    # # dropping those from df
    df_meets_criteria = df_modeling[~df_modeling[column_to_count].isin(kw_drop_list)]

    return df_meets_criteria


def drops_kw_not_meeting_criteria(df_modeling):
    """
    This function counts articles and ratios by intervention keywords, and returns a list of intervention keywords that do not meet
    the 2 articles, 3 icer criteria for inclusion in the analysis. This list of keywords can then be dropped from the dataset to be modeled

    :param df_modeling: Dataframe that contains all icers for a specific cause intervention combination i.e. for malaria_prevention.
                        Needs to have columns: intervention_keywords_final, ArticleID, RatioID
    :return: Dataframe that drops any intervention keywords that do not meet the 2 articles, 3 icer requirements

    """

    kw_drop_list = []

    article_ratio_counts = (
        df_modeling.groupby("intervention_keywords_final")[["ArticleID", "RatioID"]].agg("nunique").reset_index()
    )

    # drop rows that have only 1 Article
    kw_to_drop_df_articles = article_ratio_counts[article_ratio_counts["ArticleID"] <= 1]
    kw_drop_list.extend(list(kw_to_drop_df_articles["intervention_keywords_final"].unique()))

    # drop rows that have only 2 or less ratios
    kw_to_drop_df_ratios = article_ratio_counts[article_ratio_counts["RatioID"] <= 2]
    kw_drop_list.extend(list(kw_to_drop_df_ratios["intervention_keywords_final"]))

    # # using set to get just the unique
    kw_drop_list = list(set(kw_drop_list))

    # # dropping those from df
    df_meets_criteria = df_modeling[~df_modeling["intervention_keywords_final"].isin(kw_drop_list)]

    return df_meets_criteria


def test_kw_dropped(df1):
    """
    Function to confirm that for all intervention keywords there are at least 2 artiles, 3 icers.

    :param df1: Dataframe with intervention_keywords_final, 'ArticleID', 'RatioID' - columns
    :return: Statement if conditions are met, or ValueError message if conditions are not met
    """
    revised_count = df1.groupby("intervention_keywords_final")[["ArticleID", "RatioID"]].agg("nunique").reset_index()
    min_articles = revised_count["ArticleID"].min()
    min_ratios = revised_count["RatioID"].min()

    if (min_articles > 1) & (min_ratios > 2):
        return "meets criteria"
    else:
        print(revised_count)
        raise ValueError(f"article, ratio counts do not meet criteria: {min_articles} and {min_ratios}")


def create_df_ratio_counts_per_cause_int(dctionary_cause_int):
    """
    Creates a dataframe if want to view the counts of ratios by cause_int combination.

    :param dictionary_cause_int: A dictionary of all the various dataframes where the keys are cause_int combinations
    :return: dataframe with counts of all ratios of all cause_int combinations that were in the dictionary input
    """

    rows_per_cause_int = pd.DataFrame(
        index=range(0, len(dctionary_cause_int.keys())), columns=["cause_int_comb", "ratio_count"]
    )

    for i, key in enumerate(dctionary_cause_int.keys()):
        x = dctionary_cause_int[key].shape[0]
        rows_per_cause_int.iloc[i]["cause_int_comb"] = key
        rows_per_cause_int.iloc[i]["ratio_count"] = x

    return rows_per_cause_int


def remove_from_list(input_list, value):
    """
    Removes item from my list, without returning an error.
    Because if the value is not there, it is not a big deal.

    :param input_list: List to check that the item is in or not
    :param value: the item that needs to be removed from the list
    :return: new list with value removed
    """
    try:
        new_list = [x for x in input_list if x != value]
    except ValueError:
        # that value was not in the list
        pass
    return new_list


def is_non_unique_values_col(df, col_name):
    """
    Test whether a column in a dataframe has more than one unique value

    :param df: a dataframe to look a column into
    :param col_name: the column name in the above dataframe
    :return: True if the column in the dataframe has more than one unique value, False otherwise
    """
    return len(df[col_name].unique()) > 1


def cleaning_preds(predictions_to_summarize):
    """
    Function takes in the predictions df and then adds columns ui_upper_lower, ui_90, ratio_ul

    The ratio_ul is the column that is one key decision making variable in choice of model.
    """
    predictions_to_summarize = predictions_to_summarize[
        [
            "location_id",
            "location_name",
            "age",
            "burden_variable",
            "new_spline_cov",
            "predicted_icer_usd",
            "predicted_icer_usd_median",
            "predicted_icer_usd_lower",
            "predicted_icer_usd_upper",
            "predicted_icer_usd_90_lower",
            "predicted_icer_usd_90_upper",
            "log_GDP_2019usd_per_cap",
            "log_dalys_per_cap",
            "log_per_year_or_full_int_cost",
        ]
    ].copy()

    predictions_to_summarize["ui_upper_lower"] = (
        predictions_to_summarize["predicted_icer_usd_upper"] - predictions_to_summarize["predicted_icer_usd_lower"]
    )
    predictions_to_summarize["ui_90"] = (
        predictions_to_summarize["predicted_icer_usd_90_upper"]
        - predictions_to_summarize["predicted_icer_usd_90_lower"]
    )
    predictions_to_summarize["ratio_ul"] = (
        predictions_to_summarize["predicted_icer_usd_upper"] / predictions_to_summarize["predicted_icer_usd_lower"]
    )

    return predictions_to_summarize


def cleaning_preds_broad_cat(predictions_to_summarize):
    """
    Function takes in the predictions df and then adds columns ui_upper_lower, ui_90, ratio_ul

    The ratio_ul is the column that is one key decision making variable in choice of model.
    """
    predictions_to_summarize = predictions_to_summarize[
        [
            "location_id",
            "location_name",
            "new_spline_cov",
            "predicted_icer_usd",
            "predicted_icer_usd_median",
            "predicted_icer_usd_lower",
            "predicted_icer_usd_upper",
            "predicted_icer_usd_90_lower",
            "predicted_icer_usd_90_upper",
            "log_GDP_2019usd_per_cap",
            "log_dalys_per_cap",
        ]
    ].copy()

    predictions_to_summarize["ui_upper_lower"] = (
        predictions_to_summarize["predicted_icer_usd_upper"] - predictions_to_summarize["predicted_icer_usd_lower"]
    )
    predictions_to_summarize["ui_90"] = (
        predictions_to_summarize["predicted_icer_usd_90_upper"]
        - predictions_to_summarize["predicted_icer_usd_90_lower"]
    )
    predictions_to_summarize["ratio_ul"] = (
        predictions_to_summarize["predicted_icer_usd_upper"] / predictions_to_summarize["predicted_icer_usd_lower"]
    )

    return predictions_to_summarize


def summarizing_script_outputs(df_input, index_to_calc):
    """
    Function provides two summary statistics of the predictions as a way to quickly see the key output from the model run
    but also to test if the model run outputs changed between runs.
    """
    summary_df = pd.DataFrame(columns=["summary_value"], index=index_to_calc)

    # assuming last one in the list will be a count
    range_len = len(index_to_calc) - 1

    for i in range(range_len):
        summary_df.loc[index_to_calc[i]] = df_input[index_to_calc[i]].mean()

    summary_df.loc[index_to_calc[-1]] = df_input.shape[0]

    summary_df = summary_df.reset_index()
    return summary_df

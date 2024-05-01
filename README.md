# Cost Effectiveness Meta Regression of HIV/AIDS, Malaria, Syphilis and Tuberculosis
This repository contains the code that performs the cost effectiveness meta-regression analysis of interventions to treat global fund priorities. This analysis has been accepted for publication in The Lancet Global Health under the title, "Cost-effectiveness of interventions for HIV/AIDS, malaria, syphilis, and tuberculosis in 128 countries: A meta-regression approach".

## Purpose
The purpose for this repository is to make the analytic code for this project available as part of [Guidelines for Accurate and Transparent Health Estimates Reporting (GATHER)](http://gather-statement.org/) compliance.

## Organization
All input, intermediate, and output files are defined in the file paths.py, and are replaced with "FILEPATH".

Scripts should be run in the following order:

1. create_paired_df.py

   Constructs a data set consisting of pairs of ratios from the same article and location that differ only in that they use different values of one covariate.

2. crosswalks.py

   Analyses of differences between ICERs for sensitivity-reference pairs of ratios. Uses functions defined in crosswalk_functions.py. Step 1 of the modeling pipeline

3. sorting_cwalk_results.py

    Filters the output from the crosswalk.py to pull out the comparisons and results that are of importance for the next steps.

4. data_prep_mr.py

   Prepares the cleaned df, and then splits the clean dataset into the different cause intervention-type sets of datasets. 

5. step_2_cov.py

    This is the step 2 of the modeling pipeline and this code, calculates the spline variable for GDP per capita and removes outliers from the datasets. This step is run by a cli. 

6. creating_predictions_df.py

   Creates the predictions datasets for each of the cause-intervention-type pairs, for use in the step of the pipeline. 

7. steps_3_5_mr.py

    Completes steps 3 through 5 of the modeling pipeline, selects covariates, calculates guassian priors for some variables, fits the final model, and calculates the predicted icers for each country using the prediction datasets made in previous step. This script is also run through a cli, with a number of inputs that are used at the top of the cli function. 

8. {cause_intervention-type}_logistic_reg.R

   Runs a logistic regression to predict the probability that our predicted ICERs are cost-saving and adjusts the predicted ICERs accordingly. There is code for only HIV ART and Malaria Prevention models as they were the only datasets that met the requirements for adjusting for cost saving ratios.

9. The code to make the plots for the publication. 

## Inputs
Inputs required for the code to run are:

1. Valid paths to directories and files particularly "CLEANED_DF.csv", which is then subset to make each cause-intervention-type data subset.

2. A file specifying the values of all covariates for each prediction. Its path is specified as f"{pred_dir}/predictions/{cause_intervention_combinations}_predictions_df.csv".


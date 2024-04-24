---
title: "hiv_art_logistic_reg"
---
library(data.table)
library(lme4)
library(corrplot)
library(Hmisc)
library(reticulate)

# Reading the files from the hiv_aids_art version folder:
preds_df_path <- read.csv("predictions/hiv_aids_art_predictions.csv")
metareg_param_path <- read.csv("hiv_aids_art_model_parameters.csv" )
df <- read.csv("hiv_aids_art_logistic_regression_data_updated_230522.csv")

df = df[!df$random_effects_id %in% c('2007-01-03079', '2013-01-11266', '2015-01-20625', '2016-01-19908'),]

# Checking if the covariates' names correspond to df column names, need to replace spaces with "."
indx <- grepl('effect', colnames(df))

# We need to replace " " and "," with "." for the covariates:
metareg_param_path$covariate <- gsub(",", ".", metareg_param_path$covariate)
metareg_param_path$covariate <- gsub(" ", ".", metareg_param_path$covariate)
metareg_param_path

# dropping keywords from parameters that do not have any cost saving ratios so can be ignored
metareg_param_path_reduced <- subset(metareg_param_path, covariate!='antiretroviral.therapy.for.hiv.for.prevention..methadone.maintenance.therapy')
metareg_param_path_reduced <- subset(metareg_param_path_reduced, covariate!='antiretroviral.therapy.for.hiv..hiv.testing')
metareg_param_path_reduced <- subset(metareg_param_path_reduced, covariate!='prevention.of.mother.to.child.hiv.transmission..antiretroviral.therapy.for.hiv.for.prevention')
metareg_param_path_reduced

#A glmer_wrapper function is defined
glmer_wrapper <- function(frmla,
                          dat,
                          max_refits = 10,
                          fam = "binomial"
){
  ## Stubbornly tries to fit the model, specified by the lme4-format formula frmla, on the
  ## data.frame dat. It first fits the model, then checks convergence. If it did not
  ## converge, it then retries with lower tolerances both for the change in the log-likelihood
  ## and for the change in the parameter values, and starting where the previous attempt
  ## stopped.
  require(lme4)
  mdl <- suppressWarnings(
    glmer(frmla, family = fam, data = dat))
  converged <- is.null(mdl@optinfo$conv$lme4$code)
  if(!converged){
    i <- 1
    last_params <- mdl@theta
    last_step_sizes <- 1
    tol <- 10^(-8) # Lower tolerance will prevent stopping before the gradient is close enough to 0
    while(!converged){
      if(i > max_refits & !converged){
        
        print(paste0("model never converged."))
        return(NULL)
      }
      
      mdl <- suppressWarnings(glmer(frmla,
                                    family = fam,
                                    data = dat,
                                    start = mdl@theta,
                                    control = lmerControl(optCtrl = list("maxit" = 10000,
                                                                         "ftol_abs" = tol,
                                                                         "xtol_abs" = tol))))
      converged <- is.null(mdl@optinfo$conv$lme4$code)
      i <- i + 1
    }
  }
  return(mdl)
}

indx <- grepl('cost_saving', colnames(df)) #Checked that the cost_saving column is already in df
df[indx]

metareg_param_path_reduced

# covariates' names are extracted from the data frame
metareg_cvts <- metareg_param_path_reduced[, 'covariate']
spline_cov <- "log_GDP_2019usd_per_cap" #changed the name of the column

c(spline_cov, setdiff(metareg_cvts, c("new_spline_cov", "intercept",  "log_GDP_2019usd_per_cap" )))
metareg_cvts <- c(spline_cov, setdiff(metareg_cvts, c("new_spline_cov", "intercept",  "log_GDP_2019usd_per_cap" ))) 


# Checking if random_effect column is in df:
indx <- grepl('effect', colnames(df))

# creating a new data frame with the response variable and covariates. Adding random_effects_id:
dfx <- df[c("cost_saving", "random_effects_id", metareg_cvts)] 

# creating another dataframe for the correlation matrix, so that there are only numeric columns (w/o Article ID)
dfx_corr<- dfx[,-2] 
M <-cor(dfx_corr, use="pairwise.complete.obs")
corrplot(M, method="number", tl.cex = 0.4) 

frmla <- paste0("cost_saving ~ ", paste(metareg_cvts, collapse = " + "))
frmla <- paste0(frmla, " + (1 | random_effects_id)")
frmla

# running model
mdl <- glmer_wrapper(as.formula(frmla), dat = df)
summary(mdl)

# adjusting predictions
preds <-preds_df_path
preds$random_effects_id = "1"

preds$pred_prob <- predict(mdl, newdata =preds,
                             type="response",
                             allow.new.levels=TRUE)

df$pred_prob <- predict(mdl, newdata=df,
                          type="response",
                          re.form=~0)

df$pred_prob_with_re <- predict(mdl, newdata=df,
                                  type="response", re.form=~ (1 | random_effects_id))
#df$pred_val = df$pred_prob >= 0.5

#confus_mtx <- df[, table(df$cost_saving, df$pred_val)]
#confus_mtx <- table(df$cost_saving, factor(df$pred_val, c(FALSE, TRUE)))
#confus_mtx <- df[, table(c('cost_saving', 'pred_val'))]
#confus_mtx <- confus_mtx[sort(rownames(confus_mtx)), sort(colnames(confus_mtx))]

mean(preds$predicted_log_icer_usd_25th_per)

preds$predicted_icer_usd <- (1-preds$pred_prob) * preds$predicted_icer_usd
preds$predicted_icer_usd_median <- (1-preds$pred_prob) * preds$predicted_icer_usd_median
preds$predicted_icer_usd_lower <- (1-preds$pred_prob) * preds$predicted_icer_usd_lower
preds$predicted_icer_usd_upper <- (1-preds$pred_prob) * preds$predicted_icer_usd_upper
preds$predicted_icer_usd_25th_per <- (1-preds$pred_prob) * preds$predicted_icer_usd_25th_per
preds$predicted_icer_usd_75th_per <- (1-preds$pred_prob) * preds$predicted_icer_usd_75th_per

mean(preds$predicted_log_icer_usd_25th_per)

metrics_df <- data.table(value=metrics)
metrics_df[, metric := names(..metrics)]
setcolorder(metrics_df, c("metric", "value"))

# exporting the results
write.csv(preds, 'predictions/hiv_aids_art_predictions_adjusted.csv', sep = '')
write.csv(metrics_df, 'predictions/hiv_aids_art_metrics.csv', sep = '')

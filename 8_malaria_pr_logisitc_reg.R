---
title: "malaria_pr_logisitc_reg"
---

library(data.table)
library(lme4)
library(corrplot)
library(Hmisc)
library(reticulate)

# Reading the files from the malaria_prevention folder:
preds_df_path<- read.csv("predictions/malaria_prevention_predictions.csv")
metareg_param_path <- read.csv("malaria_prevention_model_parameters.csv" )
df <- read.csv("malaria_prevention_logistic_regression_data_updated.csv")


# Checking if the covariates' names corresponf to df columnnames, need to replace spaces with "."
indx <- grepl('effect', colnames(df))

# replace " " with "." for the covariates:
metareg_param_path$covariate <- gsub(" ", ".", metareg_param_path$covariate)
metareg_param_path

# A glmer_wrapper function is defined
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

# covariates' names are extracted from the data frame
metareg_cvts <- metareg_param_path[, 'covariate']
spline_cov <- "log_GDP_2019usd_per_cap" #changed the name of the column

metareg_cvts <- c(spline_cov, setdiff(metareg_cvts, c("new_spline_cov", "intercept",  "log_GDP_2019usd_per_cap" ))) # we removed the same covariates as Jonah removed and log_GDP_2019usd_per_cap because it is the same as spline_cov
metareg_cvts <- metareg_cvts[!metareg_cvts %in% c("indoor.residual.spraying,.malaria.treatment,.bed.nets")]

df_most <- subset(df, intervention_keywords_final == 'bed nets' | intervention_keywords_final == 'malaria intermittent preventive treatment')
NROW(df_most)

df_other <- subset(df, intervention_keywords_final != 'bed nets' & intervention_keywords_final != 'malaria intermittent preventive treatment')
NROW(df_other)


# creating a new data frame with the response variable and covariates
dfx <- df[c("cost_saving", "ArticleID", metareg_cvts)] 
dfx

# creating another dataframe for the correlation matrix, so that there are only numeric columns (w/o Article ID)
dfx_corr<- dfx[,-2] 
dfx_corr<- na.omit(dfx_corr)
M<-cor(dfx_corr)
corrplot(M, method="number", tl.cex = 0.5) 

# confirming that there are no null values
dfx<- na.omit(dfx)
NROW(dfx)

# dropping interventions without cost saving ratios
metareg_cvts_reduced <- metareg_cvts
metareg_cvts_reduced <- metareg_cvts_reduced[! metareg_cvts_reduced %in% c('malaria.intermittent.preventive.treatment.for.pregnant.women')]
# bed nets is the reference
metareg_cvts_reduced <- metareg_cvts_reduced[! metareg_cvts_reduced %in% c('bed.nets')]
# dropping efficacy because does not converge with efficacy
metareg_cvts_reduced <- metareg_cvts_reduced[! metareg_cvts_reduced %in% c('efficacy')]
metareg_cvts_reduced


# running the model
frmla <- paste0("cost_saving ~ ", paste(metareg_cvts_reduced, collapse = " + "))
frmla <- paste0(frmla, " + (1 | ArticleID)")


mdl <- glmer_wrapper(as.formula(frmla), dat = dfx)
summary(mdl)

# adjusting predictions
preds <-preds_df_path

preds$ArticleID = "1"

preds_most  <- subset(preds, bed.nets == 1 | malaria.intermittent.preventive.treatment == 1)
preds_other <- subset(preds, bed.nets != 1 & malaria.intermittent.preventive.treatment != 1)
NROW(preds_other)

# changing the bed nets column to zeros because it is the reference case for LR model
# which is diferent from the main model where vaccines are ref case
# vaccines are not being adjusted here

preds_other$pred_val = FALSE
preds_other$ArticleID = 1
preds_other$pred_prob = 0
NCOL(preds_other)
NROW(preds_other)


preds_most$pred_prob = predict(mdl, newdata = preds_most,
                             type = "response",
                             allow.new.levels = TRUE)


preds_most$pred_val <- preds_most$pred_prob >= 0.5

dfx$pred_prob = predict(mdl, newdata = df_most,
                          type = "response",
                          re.form = ~0)


dfx$pred_prob_with_re = predict(mdl, newdata=dfx,
                                  type="response", re.form = ~(1 | ArticleID))

dfx$pred_val = dfx$pred_prob >= 0.5


confus_mtx <- table(dfx$cost_saving, dfx$pred_val)
confus_mtx <- confus_mtx[sort(rownames(confus_mtx)), sort(colnames(confus_mtx))]

metrics <- sum(diag(confus_mtx)) / sum(confus_mtx)
metrics <- c(metrics, diag(confus_mtx)/rowSums(confus_mtx))
names(metrics) <- c("accuracy", "specificity", "sensitivity")

# getting before mean
mean(preds_most$predicted_icer_usd_median)

preds_most$predicted_icer_usd <- (1-preds_most$pred_prob) * preds_most$predicted_icer_usd
preds_most$predicted_icer_usd_median <- (1-preds_most$pred_prob) * preds_most$predicted_icer_usd_median
preds_most$predicted_icer_usd_lower <- (1-preds_most$pred_prob) * preds_most$predicted_icer_usd_lower
preds_most$predicted_icer_usd_upper <- (1-preds_most$pred_prob) * preds_most$predicted_icer_usd_upper
preds_most$predicted_icer_usd_25th_per <- (1-preds_most$pred_prob) * preds_most$predicted_icer_usd_25th_per
preds_most$predicted_icer_usd_75th_per <- (1-preds_most$pred_prob) * preds_most$predicted_icer_usd_75th_per

# getting after mean
mean(preds_most$predicted_icer_usd_median)


# merging adjusted predictions and non-adjusted predictions together
preds <- rbind(preds_other, preds_most)
NROW(preds)

metrics_df <- data.table(value=metrics)
metrics_df[, metric := names(..metrics)]
setcolorder(metrics_df, c("metric", "value"))

setnames(preds, old=c("malaria.intermittent.preventive.treatment", "bed.nets", "indoor.residual.spraying..malaria.treatment..bed.nets", "malaria.intermittent.preventive.treatment.for.pregnant.women", "indoor.residual.spraying"), new=c("malaria intermittent preventive treatment", "bed nets", "indoor residual spraying, malaria treatment, bed nets", "malaria intermittent preventive treatment for pregnant women", "indoor residual spraying"))
write.csv(preds, 'predictions/malaria_prevention_predictions_adjusted.csv', sep = '')



# Load necessary libraries
library(dplyr)       # For data manipulation
library(lme4)        # For mixed-effects models (if considering individual-level variation within households/assignments)
library(estimatr)    # For robust standard errors (lm_robust)
library(ivreg)       # For instrumental variable regression
library(ggplot2)     # For visualizations

# load data 
n2n_test_data <- readRDS("~/n2n_test_data.rds")
df <- n2n_test_data

# Ensure categorical variables are factors
df$race <- as.factor(df$race)
colnames(df)[colnames(df) == 'stae'] <- 'state'
df$state <- as.factor(df$state)
df$urbanicity <- as.factor(df$urbanicity)
df$canvass_result <- as.factor(df$canvass_result)
df$treatment <- as.factor(df$treatment) # 0 and 1 should be treated as factors for regression
df$voted <- as.factor(df$voted)         # 0 and 1 should be treated as factors for logistic regression

# Create a binary variable for "Canvass Attempted" for TOT analysis
df <- df %>%
  mutate(canvass_attempted = ifelse(canvass_result %in% c("NOT_HOME", "SPOKE_TO"), 1, 0))


# Raw ITT effect (difference in means)
mean_voted_treatment <- df %>%
  filter(treatment == 1) %>%
  summarise(mean_voted = mean(as.numeric(as.character(voted))))

mean_voted_control <- df %>%
  filter(treatment == 0) %>%
  summarise(mean_voted = mean(as.numeric(as.character(voted))))

raw_itt <- mean_voted_treatment$mean_voted - mean_voted_control$mean_voted
# Raw ITT = 0.0033

# Logistic Regression (more appropriate for binary outcome)
# need to manually calculate clustered standard errors or use a package
itt_logistic_model <- glm(
  voted ~ treatment + turnout_score + partisan_score + age + race + state + urbanicity,
  data = df,
  family = binomial(link = "logit")
)
summary(itt_logistic_model)

# get clustered standard errors:
library(clubSandwich)
library(lmtest)
coeftest(itt_logistic_model, vcov = vcovCR(itt_logistic_model, cluster = df$household_id, type = "CR1"))


# ITT provides overall program effect; what is the effect of ACTUALLY being canvassed? ToT analysis
# Ensure canvass_attempted is numeric (0/1)
df$canvass_attempted <- as.numeric(df$canvass_attempted)

# Two-Stage Least Squares (2SLS) with ivreg
# Instrument: treatment
# Endogenous variable: canvass_attempted
# Outcome: voted
library(ivreg)

tot_iv_model <- ivreg(
  as.numeric(as.character(voted)) ~ canvass_attempted + turnout_score + partisan_score + age + race + state + urbanicity |
    treatment + turnout_score + partisan_score + age + race + state + urbanicity,
  data = df
)
summary(tot_iv_model, diagnostics = TRUE)

# The default summary for ivreg does not include clustered SEs.
# An alternative is to manually perform 2SLS with lm_robust for each stage and then adjust for clustering.
# Manual 2SLS with clustered standard errors
# Stage 1: Predict Canvass_Attempted using Treatment and covariates
first_stage_model <- lm_robust(
   canvass_attempted ~ treatment + turnout_score + partisan_score + age + race + state + urbanicity,
   data = df,
   clusters = household_id,
   se_type = "stata"
 )
df$Canvass_Attempted_Predicted <- predict(first_stage_model, newdata=df)

# Stage 2: Regress Voted on predicted Canvass_Attempted and covariates
second_stage_model <- lm_robust(
  as.numeric(as.character(voted)) ~ Canvass_Attempted_Predicted + turnout_score + partisan_score + age + race + state + urbanicity,
  data = df,
  clusters = household_id,
  se_type = "stata"
)
summary(second_stage_model)




###
# Subgroup analyses: race, age, state, urbanicity 
# ITT Race Model
itt_race_interaction_model <- lm_robust(
  as.numeric(as.character(voted)) ~ treatment * race + turnout_score + partisan_score + age + state + urbanicity,
  data = df,
  clusters = household_id,
  se_type = "stata"
)
summary(itt_race_interaction_model)

# interpret the coefficients for Treatment:Race categories.
# "Treatment1" would be the effect for the reference Race category (e.g., White),
# and the interaction terms would show the *additional* effect for other racial groups.

# ITT Age Model
# First, group age into buckets
table(df$age)
age_breaks <- c(15, 25, 40, 60, 80, 100, 115)
age_labels <- c("16-25", "26-40", "41-60", "61-80", "81-100", "101+")
df$AgeCat <- cut(
  df$age,
  breaks = age_breaks,
  labels = age_labels,
  right = TRUE,
  include.lowest = TRUE
)
table(df$AgeCat)

# then run interaction model for age group
itt_age_interaction_model <- lm_robust(
  as.numeric(as.character(voted)) ~ treatment * AgeCat + turnout_score + partisan_score + race + state + urbanicity,
  data = df,
  clusters = household_id,
  se_type = "stata"
)
summary(itt_age_interaction_model)

# ITT State Model
itt_state_interaction_model <- lm_robust(
  as.numeric(as.character(voted)) ~ treatment * state + turnout_score + partisan_score + age + race + urbanicity,
  data = df,
  clusters = household_id,
  se_type = "stata"
)
summary(itt_state_interaction_model)

# ITT Urbanicity Model
itt_urbanicity_interaction_model <- lm_robust(
  as.numeric(as.character(voted)) ~ treatment * urbanicity + turnout_score + partisan_score + age + race + state,
  data = df,
  clusters = household_id,
  se_type = "stata"
)
summary(itt_urbanicity_interaction_model)


###
# Addressing complexities and challenges 
#cluster SE at canvasser ID
# You might need to decide whether to cluster at Household_ID, Canvasser_ID, or both
# Example: Clustered at Canvasser_ID
itt_canvasser_cluster_model <- lm_robust(
  as.numeric(as.character(voted)) ~ treatment + turnout_score + partisan_score + age + race + state + urbanicity,
  data = df,
  clusters = canvasser_id, # Or list(Household_ID, Canvasser_ID) if both levels are relevant
  se_type = "stata"
)
summary(itt_canvasser_cluster_model)
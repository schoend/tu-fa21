# Chapter 10   https://mdsr-book.github.io/mdsr2e/
# 10.1 Predictive Modeling
# 10.2 Simple classification models
# 10.2.1 Example: High-earners in the 1994 United State Census
# 10.2.1.x The null model, log_1 model, log_all model, autoplot
# 10.3.x   Hint at ROC curves

## ----
library(tidyverse)
library(mdsr)
library(yardstick)
library(tidymodels)
url <-
  "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
census <- read_csv(
  url,
  col_names = c(
    "age", "workclass", "fnlwgt", "education", 
    "education_1", "marital_status", "occupation", "relationship", 
    "race", "sex", "capital_gain", "capital_loss", "hours_per_week", 
    "native_country", "income"
  )
) %>%
  mutate(income = factor(income))
glimpse(census)

## ----
set.seed(364)
n <- nrow(census)
census_parts <- census %>%
  initial_split(prop = 0.8)

train <- census_parts %>%
  training()

test <- census_parts %>%
  testing()

list(train, test) %>%
  map_int(nrow)

## ----
pi_bar <- train %>%
  count(income) %>%
  mutate(pct = n / sum(n)) %>%
  filter(income == ">50K") %>%
  pull(pct)
pi_bar

## ----
train %>%
  count(income) %>%
  mutate(pct = n / sum(n))

## ---- Using logistic_reg to get null model;

mod_null <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(income ~ 1, data = train)

## ----
pred <- train %>%
  select(income, capital_gain) %>%
  bind_cols(
    predict(mod_null, new_data = train, type = "class")
  ) %>%
  rename(income_null = .pred_class)
accuracy(pred, income, income_null)

## ----
confusion_null <- pred %>%
  conf_mat(truth = income, estimate = income_null)
confusion_null

autoplot(confusion_null) +
  geom_label(
    aes(
      x = (xmax + xmin) / 2, 
      y = (ymax + ymin) / 2, 
      label = c("TN", "FP", "FN", "TP")
    )
  )

#Repeating with log_1 model
mod_log_1 <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(income ~ capital_gain, data = train)

## ----
pred <- train %>%
  select(income, capital_gain) %>%
  bind_cols(
    predict(mod_log_1, new_data = train, type = "class")
  ) %>%
  rename(income_log_1 = .pred_class)
accuracy(pred, income, income_log_1)

## ----
confusion_log_1 <- pred %>%
  conf_mat(truth = income, estimate = income_log_1)
confusion_log_1

autoplot(confusion_log_1) +
  geom_label(
    aes(
      x = (xmax + xmin) / 2, 
      y = (ymax + ymin) / 2, 
      label = c("TN", "FP", "FN", "TP")
    )
  )

#Repeating again with log_all model

mod_log_all <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(income ~ age + workclass + education + marital_status + 
        occupation + relationship + race + sex + 
        capital_gain + capital_loss + hours_per_week, 
      data = train)

## ----
pred <- pred %>%
  bind_cols(
    predict(mod_log_all, new_data = train, type = "class")
  ) %>%
  rename(income_log_all = .pred_class)

accuracy(pred, income, income_log_all)

## ----
confusion_log_all <- pred %>%
  conf_mat(truth = income, estimate = income_log_all)

confusion_log_all

autoplot(confusion_log_all) +
  geom_label(
    aes(
      x = (xmax + xmin) / 2, 
      y = (ymax + ymin) / 2, 
      label = c("TN", "FP", "FN", "TP")
    )
  )


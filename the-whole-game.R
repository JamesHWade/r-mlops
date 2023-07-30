# Load Packages & Set Options ---------------------------------------------

library(tidyverse)         # data wrangling and cleaning
library(tidymodels)        # modeling and machine learning
library(ranger)            # random forest model engine
library(brulee)            # neural network with torch
library(pins)              # sharing resources across sessions and users
library(vetiver)           # model versioning and deployment
library(plumber)           # API creation
library(palmerpenguins)    # penguin dataset
library(gt)                # creating table objects for data
library(conflicted)        # handling function conflicts
tidymodels_prefer()        # handle common conflicts with tidymodels and others

# resolve conflict in favor of palmerpenguins::penguins
conflict_prefer("penguins", "palmerpenguins")

# set the default ggplot2 theme to theme_bw
theme_set(theme_bw())
# enable tidymodels dark mode for console messages
options(tidymodels.dark = TRUE)

# Exploratory Data Analysis -----------------------------------------------

body_bg = "#111233"
body_color = "#ffa275"
link_color = "#99D9DD"

# Create custom theme
theme_custom <- function(base_size = 16, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Specify plot background
      plot.background = element_rect(fill = body_bg, color = NA),
      # Specify panel background
      panel.background = element_rect(fill = body_bg, color = NA),
      # Specify text and title colors
      text = element_text(color = body_color),
      title = element_text(color = body_color),
      strip.background = element_rect(fill = body_bg, color = NA),
      
      # Specify axis line colors
      axis.line = element_line(color = link_color),
      axis.ticks = element_line(color = link_color),
      # Specify axis text colors
      strip.text = element_text(color = link_color),
      axis.text = element_text(color = link_color),
      axis.title = element_text(color = link_color),
      legend.background = element_rect(fill = body_bg, color = NA)
    )
}

penguins %>%
  filter(!is.na(sex)) %>%
  ggplot(aes(x     = flipper_length_mm,
             y     = bill_length_mm,
             color = sex,
             size  = body_mass_g)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~species) +
  scale_color_manual(values = c(body_color, link_color)) +
  theme_custom() +
  guides(size = guide_legend(override.aes = list(color = body_color)))

# Prepare & Split Data ----------------------------------------------------

# remove any rows with missing sex data, and exclude the 'year' and 'island'
penguins_df <-
  penguins |>
  drop_na(sex) |>
  select(-year, -island)

# seet the seed for reproducibility
set.seed(1234)

# Split the data into training and testing datasets stratified by 'sex'
penguin_split <- initial_split(penguins_df, strata = sex)

# extract the training data from the split
penguin_train <- training(penguin_split)

# extract the testing data from the split
penguin_test  <- testing(penguin_split)

# create folds for cross validation on the training data
penguin_folds <- vfold_cv(penguin_train)

# Create Recipe & Specify Models ------------------------------------------

# create our tidymodels recipe
penguin_rec <-
  recipe(sex ~ ., data = penguin_train) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_dummy(species)

# specify models with parsnip
# logistic regression with glm
glm_spec <-
  logistic_reg(penalty = 1) |>
  set_engine("glm")

# random forest
tree_spec <-
  rand_forest(min_n = tune()) |>
  set_engine("ranger") |>
  set_mode("classification")

# simple neural network
mlp_brulee_spec <-
  mlp(
    hidden_units = tune(),
    epochs       = tune(),
    penalty      = tune(),
    learn_rate   = tune()
  ) %>%
  set_engine("brulee") %>%
  set_mode("classification")

# Fit Models & Tune Hyperparameters ---------------------------------------

# specify parameters for hyperparameter tuning with Bayesian optimization
bayes_control <- control_bayes(no_improve = 10L,
                               time_limit = 20,
                               save_pred  = TRUE,
                               verbose    = TRUE)

# create a workflow set with recipe and models
workflow_set <-
  workflow_set(
    preproc = list(penguin_rec),
    models = list(glm   = glm_spec,
                  tree  = tree_spec,
                  torch = mlp_brulee_spec)
  ) |>
  workflow_map("tune_bayes",
               iter = 50L,
               resamples = penguin_folds,
               control = bayes_control
  )

# rank our model results
# create table of best models defined using roc_auc metric
rank_results(workflow_set,
             rank_metric = "roc_auc",
             select_best = TRUE) |>
  gt()

# use autoplot to compare models
workflow_set |> autoplot()

best_model_id <- "recipe_glm"

# Finalize Model ----------------------------------------------------------

# select best model
best_fit <-
  workflow_set |>
  extract_workflow_set_result(best_model_id) |>
  select_best(metric = "accuracy")

# create workflow for best model
final_workflow <-
  workflow_set |>
  extract_workflow(best_model_id) |>
  finalize_workflow(best_fit)

# fit final model with all data
final_fit <-
  final_workflow |>
  last_fit(penguin_split)

# show model performance
final_fit |>
  collect_metrics() |>
  gt()

final_fit |>
  collect_predictions() |>
  roc_curve(sex, .pred_female) |>
  autoplot()

# Create Vetiver Model & API ----------------------------------------------

# create a vetiver model from final fit
final_fit_to_deploy <- final_fit |> extract_workflow()

v <- vetiver_model(final_fit_to_deploy, model_name = "penguins_model")

model_board <- board_folder(path = "pins-r", versioned = TRUE)
model_board |> vetiver_pin_write(v)
write_board_manifest(model_board)

# use board_url
pin_loc     <- pins:::github_raw("JamesHWade/r-mlops/main/pins-r/_pins.yaml")
model_board <- board_url(pin_loc)
model_board |> vetiver::vetiver_write_plumber("penguins_model")

# create a model API with plumber
pr() |>
  vetiver_api(v) |> 
  plumber::pr_run(port = 8080)

# Create Dockerfile & Deploy ----------------------------------------------

# create a dockerfile
model_board |> vetiver_prepare_docker("penguins_model")


endpoint <- vetiver::vetiver_endpoint("https://jameshwade-penguins-model.hf.space:8080/predict")
predict(endpoint, new_data = 1)

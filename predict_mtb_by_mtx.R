#===================================
# Library
#===================================

library(tidyverse)
library(tidymodels)
library(stringr)
library(ggplot2)
library(magrittr)
library(readr)
library(fastshap)

print(str_c(Sys.time(), "- Library loaded"))

#===================================
# Import data
#===================================


args = commandArgs(trailingOnly=TRUE)

outcome <- args[[1]]
group_filter <- args[[2]]
feature_table_path <- args[[3]]
out_path <- args[[4]]

feature_table <- read_delim(feature_table_path, "\t")

print(str_c(Sys.time(), "- Data loaded"))

#===================================
# Organize data for predictions
#===================================

data_for_single_model <- feature_table%>%
  #filter data according to the feature_table_parameters
  filter(group == group_filter)%>%
  mutate(outcome_name = outcome)%>%
  rename("outcome" = all_of(outcome))%>%
  select(subject_visit_id, subject_id, visit, group, contains("taxa_"), outcome, outcome_name)%>%
  filter(outcome>0)

sample_map <- data_for_single_model%>%
  select(subject_id, visit, group, outcome_name)%>%
  mutate(row_id=row_number()) 

data_cv <-data_for_single_model%>%
  group_vfold_cv(group = subject_id, v = 10, repeats = 10)

print(str_c(Sys.time(), "- Data preprocessed"))

#===================================
# Prep and run model 
#===================================

model_recipe <- recipe(outcome  ~  ., data = data_for_single_model)%>%
  update_role(subject_visit_id,subject_id, visit, group, outcome_name,new_role = "ID")%>%
  step_zv(all_numeric())
# step_normalize(all_numeric())

cores <- parallel::detectCores()
rf_model <-rand_forest() %>%
  set_engine("ranger" ,  num.threads = cores, importance = "impurity") %>%
  set_mode("regression")

#workflow
model_rf_workflow <- workflow()%>%
  add_recipe(model_recipe) %>%
  add_model(rf_model)

print(str_c(Sys.time(), "- Defined model specs"))


#run model
model_results <- fit_resamples(model_rf_workflow,
                               data_cv,
                               control = control_resamples(save_pred = TRUE),
                               metrics = metric_set(rmse))

print(str_c(Sys.time(), "- Finished running models"))

#===================================
# Calculate performance 
#===================================

#collect tidy predictions
model_predictions <-  collect_predictions(model_results)%>%
  rename("repeats" = id, resample = id2, pred = .pred, row_id = .row)%>%
  arrange(row_id)%>%
  select(-.config)%>%
  mutate(outcome = outcome,
         group_filter = group_filter)%>%
  left_join(sample_map)%>%
  mutate(outcome = outcome,
         group_filter = group_filter,
         feature_table_path = feature_table_path)

#calculate correlations
mean_model_predictions <- model_predictions%>%
  group_by(row_id)%>%
  mutate(.pred = mean(pred))%>%
  select(-repeats , -resample     )%>%
  distinct()

spearman_correlation<-cor.test(x=mean_model_predictions$.pred,
                               y= mean_model_predictions$outcome,
                               method = c( "spearman"))%>%
  tidy()%>%
  rename(value = estimate, p = p.value)%>%
  mutate(metric = "spearman")%>%
  select(metric, value, p )

#collect metrics
model_metrics <- collect_metrics(model_results)%>%
  rename(metric = .metric, value = mean)%>%
  select(metric, value, std_err)%>%
  bind_rows(spearman_correlation)%>%
  mutate(outcome = outcome,
         group_filter = group_filter,
         feature_table_path = feature_table_path)


#===================================
# SHAP
#===================================

# if model is reliable (spearman corr > 0.2 ) - compute SHAP values for the features
if(spearman_correlation$value >=0.2){
  calc_shap <- TRUE
  
  print(str_c(Sys.time(), "- Spearman_correlation>=0.2, calculate SHAP"))
  
  # fastshap doesn't work well with recipes pre-processing steps, therefore, we first bake that data according to the recipe and then fit the model on baked data
  # bake data
  # recipe_prep <- prep(recipe)
  # baked_feature_table <- bake(recipe_prep, new_data = NULL)%>%
  #   select(-sample_id, -genome_id)
  # data_ids <- bake(recipe_prep, new_data = NULL)%>%
  #   select(sample_id, genome_id)
  
  data_outcome_and_features <- data_for_single_model%>%
    select(-c(subject_visit_id, subject_id, visit, group))
  
  #fit the model
  model_fit <- fit(formula = outcome ~ ., rf_model, data = data_outcome_and_features)
  
  #fastshap need also table with only the features
  data_only_features <- data_outcome_and_features%>%
    select(-c(outcome))%>%
    as.data.frame()
  
  # calculate shap
  predict_function <-  function(model, newdata) {predict(model, newdata) %>% pluck(.,1)}
  shap_values <- fastshap::explain(model_fit, X = data_only_features, pred_wrapper = predict_function)
  
  #tidy data  
  shap_values_tidy <- shap_values%>%
    as_tibble()%>%
    select(-outcome_name)%>%
    bind_cols(sample_map, .)%>%
    select(-row_id)%>%
    pivot_longer(-c("subject_id", "visit", "group", "outcome_name" ), names_to = "taxa", values_to = "shap_value")%>%
    mutate(group_filter = group_filter,
           feature_table_path = feature_table_path)

  print(str_c(Sys.time(), "- SHAP calculated"))
  
}else{
  print(str_c(Sys.time(), "- Spearman_correlation<0.2, skip SHAP calculation"))
  calc_shap <- FALSE
}

#===================================
# Save files
#===================================

#defin path
path_metric <- str_c(out_path,"/" , outcome,"_", group_filter, ".metric.txt")
path_predictions <- str_c(out_path,"/", outcome,"_", group_filter, ".predictions.txt")
path_shap <- str_c(out_path,"/", outcome,"_", group_filter, ".shap.txt")

#save files

if(calc_shap== TRUE){
  write_delim(shap_values_tidy, path_shap)
}

write_delim(model_metrics, path_metric)
write_delim(model_predictions, path_predictions)

print(str_c(Sys.time(), "- Files saved"))
print("Done!")

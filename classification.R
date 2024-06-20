## PACOTES ##

library(tidyverse)	# manipulacao de dados
library(tidymodels) # ferramentas de ML
library(extrasteps) # complemento para pre-processamento dos dados
library(cutpointr)  # ponto de corte
library(plsmod)     # necessario para usar modelo pls
library(tabnet)     # necessario para usar modelo tabnet
library(stacks)     # stacking
library(tictoc)     # registrar o tempo de execução de comandos
library(janitor)    # limpeza de dados
library(vip)        # extrair a importancia de cada variavel
library(DALEX)      # extrair a importancia de cada variavel via permutacao
library(DALEXtra)   # complemento do pacote DALEX



##### CARREGANDO/LIMPANDO OS DADOS #####

df<- read.csv("winequality_red.csv") %>% 
  data.frame()

df %>% glimpse()

df<- df %>% 
  dplyr::rename("y"="quality") %>% 
  dplyr::mutate(y=ifelse(y>=7,"good","bad"))
  
df$y<- df$y %>% as.factor()

glimpse(df)






##### SPLIT TRAIN/TEST/VALIDATION #####

set.seed(0)
split<- initial_split(df, prop=0.8, strata=y)

df_train<- training(split)    # usado para cross-validation
df_test<- testing(split)      # usado para verificar desempenho

folds<- vfold_cv(df_train, v=2, repeats=5, strata=y)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(y ~ . , data = df_train) %>%
  #step_rm(...) %>%                                           # variaveis removidas
  step_filter_missing(all_predictors(), threshold = 0.3) %>%  # variaveis +30% de faltantes
  step_zv(all_predictors()) %>%                               # variaveis sem variabilidade
  #step_nzv(all_predictors(), freq_cut=tune()) %>%            # variaveis quase sem variabilidade
  step_YeoJohnson(all_numeric_predictors()) %>%               # normalizar variaveis
  #step_impute_bag(all_predictors()) %>%                      # imputando faltantes
  #step_naomit() %>%                                          # deletando faltantes
  step_normalize(all_numeric_predictors()) %>%                # padronizar variaveis
  #step_robust(all_numeric_predictors()) %>%                  # padronizacao robusta
  #step_corr(all_numeric_predictors(),threshold=tune()) %>%   # removendo variaveis correlacionadas
  #step_other(all_nominal_predictors(),threshold=tune()) %>%  # cria a categoria "outros"
  step_novel(all_nominal_predictors()) %>%                    # novas categorias
  step_dummy(all_nominal_predictors())                        # variaveis dummy


receita_pls<- recipe(y ~ . , data = df_train) %>%
  #step_rm(...) %>%                                           # variaveis removidas
  step_filter_missing(all_predictors(), threshold = 0.3) %>%  # variaveis +30% de faltantes
  step_zv(all_predictors()) %>%                               # variaveis sem variabilidade
  #step_nzv(all_predictors(), freq_cut=tune()) %>%            # variaveis quase sem variabilidade
  step_YeoJohnson(all_numeric_predictors()) %>%               # normalizar variaveis
  #step_impute_bag(all_predictors()) %>%                      # imputando faltantes
  #step_naomit() %>%                                          # deletando faltantes
  step_normalize(all_numeric_predictors()) %>%                # padronizar variaveis
  #step_robust(all_numeric_predictors()) %>%                  # padronizacao robusta
  #step_corr(all_numeric_predictors(),threshold=tune()) %>%   # removendo variaveis correlacionadas
  #step_other(all_nominal_predictors(),threshold=tune()) %>%  # cria a categoria "outros"
  step_novel(all_nominal_predictors()) %>%                    # novas categorias
  step_dummy(all_nominal_predictors()) %>%                    # variaveis dummy
  step_pls(all_numeric_predictors(),
           outcome="y",
           num_comp=tune(),
           predictor_prop = tune())                           # reducao de dimensao




##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune(),
                             dist_power = tune(),
                             weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")


model_pls<- parsnip::pls(num_comp = tune(),
                         predictor_prop = tune()) %>%
  set_engine("mixOmics") %>%
  set_mode("classification")


model_net<- logistic_reg(penalty = tune(),
                         mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")


model_rfo<- rand_forest(mtry = tune(),
                        trees = 10000,
                        min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")


model_xgb<- boost_tree(mtry = tune(),
                       trees = 10000,
                       min_n = tune(),
                       loss_reduction = tune(),
                       learn_rate = tune(),
                       stop_iter = 50) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


model_svm<- svm_rbf(cost = tune(),
                    margin = tune(),
                    rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")


# callback_list <- list(keras::callback_early_stopping(monitor = "val_loss", 
#                                                      min_delta = 0, 
#                                                      patience = 10))
# 
# model_mlp <- mlp(epochs = 200,
#                  hidden_units = tune(),
#                  dropout = tune(),
#                  activation = "relu") %>% 
#   set_engine("keras",
#              verbose = 1,
#              seeds = 0, 
#              #metrics = c("accuracy" ), 
#              #validation_split = 1/6,
#              callbacks = callback_list) %>% 
#   set_mode("classification")


model_mlp <- mlp(epochs = 30,
                 hidden_units = tune(),
                 dropout = tune(),
                 learn_rate = tune(),
                 activation = "relu") %>%
  set_engine("brulee") %>%
  set_mode("classification")


model_tbn <- tabnet(epochs = 200,
                    penalty = tune(),
                    learn_rate = tune(),
                    decision_width = tune(),
                    attention_width = tune(),
                    num_steps = tune(),
                    early_stopping_tolerance = 0.001,
                    early_stopping_patience = 10) %>% 
  set_engine("torch") %>% 
  set_mode("classification")




##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(receita_pls) %>%
  add_model(model_knn)

wf_pls<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_pls)

wf_net<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_net)

wf_rfo<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rfo)

wf_xgb<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_xgb)

wf_svm<- workflow() %>%
  add_recipe(receita_pls) %>%
  add_model(model_svm)

wf_mlp<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_mlp)

wf_tbn<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_tbn)




##### TUNAGEM DE HIPERPARAMETROS - BAYESIAN SEARCH #####

## KNN - K NEAREST NEIGHBORS

tic()
tune_knn<- tune_bayes(wf_knn,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(neighbors(range=c(1,min(200,trunc(0.25*nrow(df_train))))),
                                              dist_power(range=c(1,2)),
                                              weight_func(c("epanechnikov",
                                                            "rectangular",
                                                            "triangular")),#,
                                                            #"gaussian",
                                                            #"cos",
                                                            #"rank",
                                                            #"optimal",
                                                            #"biweight",
                                                            #"triweight",
                                                            #"inv",)),
                                              num_comp(range=c(1,min(100,trunc(0.75*ncol(df_train))))),
                                              predictor_prop(range=c(0,1)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 223.33 sec elapsed (~ 4 min)




## PLS - PARTIAL LEAST SQUARE

tic()
tune_pls<- tune_bayes(wf_pls,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(num_comp(range=c(1,min(100,trunc(0.75*ncol(df_train))))),
                                              predictor_prop(range=c(0,1)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 101.03 sec elapsed (~ 2 min)



## NET - ELASTIC NET

tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,5)),
                                              mixture(range=c(0,1)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 31.97 sec elapsed (~ 0.5 min)




## RFO - RANDOM FOREST

tic()
tune_rfo<- tune_bayes(wf_rfo,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(mtry(range=c(1,trunc(0.9*ncol(df_train)))),
                                              min_n(range=c(1,min(200,trunc(0.25*nrow(df_train))))))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 405.44 sec elapsed (~ 7 min)




## XGB - XGBOOSTING

tic()
tune_xgb<- tune_bayes(wf_xgb,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(mtry(range=c(1,trunc(0.9*ncol(df_train)))),
                                              min_n(range=c(1,min(200,trunc(0.25*nrow(df_train))))),
                                              loss_reduction(range=c(-10,5)),
                                              learn_rate(range=c(-10,0)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 906.62 sec elapsed (~ 15 min)




## SVM - SUPPORT VECTOR MACHINE

tic()
tune_svm<- tune_bayes(wf_svm,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(cost(range=c(-10,5)),
                                              svm_margin(range=c(0,0.5)),
                                              rbf_sigma(range=c(-10,5)),
                                              num_comp(range=c(1,min(100,trunc(0.75*ncol(df_train))))),
                                              predictor_prop(range=c(0,1)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 108.87 sec elapsed (~ 2 min)




## MLP - MULTILAYER PERCEPTRON

tic()
tune_mlp<- tune_bayes(wf_mlp,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(hidden_units(range=c(8,1024)),
                                              dropout(range=c(0.2,0.8)),
                                              learn_rate(range=c(-10,0)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 778.31 sec elapsed (~ 13 min)




## TBN - TABNET

tic()
tune_tbn<- tune_bayes(wf_tbn,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,5)),
                                              learn_rate(range=c(-10,0)),
                                              decision_width(range=c(4,80)),
                                              attention_width(range=c(4,80)),
                                              num_steps(range=c(2,12)))#,
                                              #threshold(range=c(0.7,1)),
                                              #freq_cut(range=c(5,50)))
)
toc()
# 2711.58 sec elapsed (~ 45 min)






## VISUALIZANDO OS MELHORES MODELOS (BEST ROC AUC)

show_best(tune_knn, metric="roc_auc", n=3)
show_best(tune_pls, metric="roc_auc", n=3)
show_best(tune_net, metric="roc_auc", n=3)
show_best(tune_rfo, metric="roc_auc", n=3)
show_best(tune_xgb, metric="roc_auc", n=3)
show_best(tune_svm, metric="roc_auc", n=3)
show_best(tune_mlp, metric="roc_auc", n=3)
show_best(tune_tbn, metric="roc_auc", n=3)




##### PREPARANDO STACKING #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_pls) %>% 
  add_candidates(tune_net) %>% 
  add_candidates(tune_rfo) %>% 
  add_candidates(tune_xgb) %>% 
  add_candidates(tune_svm) %>% 
  add_candidates(tune_mlp) %>% 
  add_candidates(tune_tbn)

stack_ensemble_data


##### AJUSTANDO STACKING #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = 10^(-9:3),
                    mixture = seq(0,1,by=0.1), # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = TRUE,
                    metric = metric_set(roc_auc))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights")

stack_ensemble_model$penalty


##### REFINANDO O AJUSTE DOS PARÂMETROS DO STACKING  #####

p<- as.numeric(stack_ensemble_model$penalty[1])
grid.p<- c(0.5*p,0.75*p,p,1.25*p,1.5*p)

m<- as.numeric(stack_ensemble_model$penalty[2])
if(m==0) {
  grid.m <- c(0, 0.01, 0.025, 0.05, 0.075)
} else{
  if (m == 1) {
    grid.m <- c(0.925, 0.95, 0.975, 0.99, 1)
  } else{
    inv.m <- log(m / (1 - m))
    grid.m <-
      1 / (1 + exp(-c(
        0.5 * inv.m, 0.75 * inv.m, inv.m, 1.25 * inv.m, 1.5 * inv.m
      )))
  }
}

##### REAJUSTANDO O STACKING  #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = grid.p,
                    mixture = grid.m, # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = TRUE,
                    metric = metric_set(roc_auc))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights")

stack_ensemble_model$penalty


##### REFINANDO O AJUSTE DOS PARÂMETROS DO STACKING  #####

p<- as.numeric(stack_ensemble_model$penalty[1])
grid.p<- c(0.9*p,0.95*p,p,1.05*p,1.1*p)

m<- as.numeric(stack_ensemble_model$penalty[2])
if(m==0) {
  grid.m <- c(0, 0.005, 0.01, 0.015, 0.02)
} else{
  if (m == 1) {
    grid.m <- c(0.98, 0.985, 0.99, 0.995, 1)
  } else{
    inv.m <- log(m / (1 - m))
    grid.m <-
      1 / (1 + exp(-c(
        0.9 * inv.m, 0.95 * inv.m, inv.m, 1.05 * inv.m, 1.1 * inv.m
      )))
  }
}

##### REAJUSTANDO O STACKING  #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = grid.p,
                    mixture = grid.m, # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = TRUE,
                    metric = metric_set(roc_auc))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights") + ggtitle("")

stack_ensemble_model$penalty


##### FINALIZANDO O MODELO #####

stack_ensemble_trained<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_trained





##### FINALIZANDO MODELOS INDIVIDUAIS #####

wf_knn_trained<- wf_knn %>% finalize_workflow(select_best(tune_knn,metric="roc_auc")) %>% fit(df_train)
wf_pls_trained<- wf_pls %>% finalize_workflow(select_best(tune_pls,metric="roc_auc")) %>% fit(df_train)
wf_net_trained<- wf_net %>% finalize_workflow(select_best(tune_net,metric="roc_auc")) %>% fit(df_train)
wf_rfo_trained<- wf_rfo %>% finalize_workflow(select_best(tune_rfo,metric="roc_auc")) %>% fit(df_train)
wf_xgb_trained<- wf_xgb %>% finalize_workflow(select_best(tune_xgb,metric="roc_auc")) %>% fit(df_train)
wf_svm_trained<- wf_svm %>% finalize_workflow(select_best(tune_svm,metric="roc_auc")) %>% fit(df_train)
wf_mlp_trained<- wf_mlp %>% finalize_workflow(select_best(tune_mlp,metric="roc_auc")) %>% fit(df_train)
wf_tbn_trained<- wf_tbn %>% finalize_workflow(select_best(tune_tbn,metric="roc_auc")) %>% fit(df_train)

# mlp_best<- tune_mlp %>% select_best(metric="roc_auc")
# mlp_best_list<- mlp_best %>% as.list()
# mlp_best_list$hidden_units <- mlp_best_list$hidden_units %>% unlist()
# wf_mlp_trained<- wf_mlp %>% finalize_workflow(mlp_best_list) %>% fit(df_train)




## SALVANDO OS MODELOS

# saveRDS(wf_pls_trained,"wf_pls_trained.rds")
# saveRDS(wf_net_trained,"wf_net_trained.rds")
# saveRDS(wf_rfo_trained,"wf_rfo_trained.rds")
# saveRDS(wf_xgb_trained,"wf_xgb_trained.rds")
# saveRDS(wf_svm_trained,"wf_svm_trained.rds")
# saveRDS(wf_mlp_trained,"wf_mlp_trained.rds")
# saveRDS(stack_ensemble_model,"stack_ensemble_model.rds")
# saveRDS(stack_ensemble_trained,"stack_ensemble_trained.rds")



## CARREGANDO OS MODELOS SALVOS

# wf_pls_trained<- readRDS("wf_pls_trained.rds")
# wf_net_trained<- readRDS("wf_net_trained.rds")
# wf_rfo_trained<- readRDS("wf_rfo_trained.rds")
# wf_xgb_trained<- readRDS("wf_xgb_trained.rds")
# wf_svm_trained<- readRDS("wf_svm_trained.rds")
# wf_mlp_trained<- readRDS("wf_mlp_trained.rds")
# stack_ensemble_model<- readRDS("stack_ensemble_model.rds")
# stack_ensemble_trained<- readRDS("stack_ensemble_trained.rds")



####################################################
#####  ESCOLHENDO O PONTO DE CORTE - F1-SCORE  #####
####################################################

## CORTE PARTIAL LEAST SQUARE (PLS)

cut_pls<- tune_pls %>% 
  dplyr::select(id, .predictions) %>% 
  unnest(.predictions) %>% 
  filter(num_comp==as.numeric(show_best(tune_pls,metric="roc_auc",n=1)[1]),
         predictor_prop==as.numeric(show_best(tune_pls,metric="roc_auc",n=1)[2]),
         threshold==as.numeric(show_best(tune_pls,metric="roc_auc",n=1)[3]),
         freq_cut==as.numeric(show_best(tune_pls,metric="roc_auc",n=1)[4])) %>% 
  dplyr::select(.pred_Sim, auto_suficiencia_financeira) %>% 
  dplyr::rename(prob=.pred_Sim) %>% 
  cutpointr(prob, auto_suficiencia_financeira, method=minimize_metric, metric=roc01)

cut_pls<- cut_pls$optimal_cutpoint


## CORTE ELASTIC-NET

cut_net<- tune_net %>% 
  dplyr::select(id, .predictions) %>% 
  unnest(.predictions) %>% 
  filter(penalty==as.numeric(show_best(tune_net,metric="roc_auc",n=1)[1]),
         mixture==as.numeric(show_best(tune_net,metric="roc_auc",n=1)[2]),
         threshold==as.numeric(show_best(tune_net,metric="roc_auc",n=1)[3]),
         freq_cut==as.numeric(show_best(tune_net,metric="roc_auc",n=1)[4])) %>% 
  dplyr::select(.pred_Sim, auto_suficiencia_financeira) %>% 
  dplyr::rename(prob=.pred_Sim) %>% 
  cutpointr(prob, auto_suficiencia_financeira, method=minimize_metric, metric=roc01)

cut_net<- cut_net$optimal_cutpoint



## CORTE RANDOM FOREST

cut_rfo<- tune_rfo %>% 
  dplyr::select(id, .predictions) %>% 
  unnest(.predictions) %>% 
  filter(mtry==as.numeric(show_best(tune_rfo,metric="roc_auc",n=1)[1]),
         min_n==as.numeric(show_best(tune_rfo,metric="roc_auc",n=1)[2]),
         threshold==as.numeric(show_best(tune_rfo,metric="roc_auc",n=1)[3]),
         freq_cut==as.numeric(show_best(tune_rfo,metric="roc_auc",n=1)[4])) %>% 
  dplyr::select(.pred_Sim, auto_suficiencia_financeira) %>% 
  dplyr::rename(prob=.pred_Sim) %>% 
  cutpointr(prob, auto_suficiencia_financeira, method=minimize_metric, metric=roc01)

cut_rfo<- cut_rfo$optimal_cutpoint



## CORTE XGBOOSTING

cut_xgb<- tune_xgb %>% 
  dplyr::select(id, .predictions) %>% 
  unnest(.predictions) %>% 
  filter(mtry==as.numeric(show_best(tune_xgb,metric="roc_auc",n=1)[1]),
         min_n==as.numeric(show_best(tune_xgb,metric="roc_auc",n=1)[2]),
         loss_reduction==as.numeric(show_best(tune_xgb,metric="roc_auc",n=1)[3]),
         learn_rate==as.numeric(show_best(tune_xgb,metric="roc_auc",n=1)[4]),
         threshold==as.numeric(show_best(tune_xgb,metric="roc_auc",n=1)[5]),
         freq_cut==as.numeric(show_best(tune_xgb,metric="roc_auc",n=1)[6])) %>% 
  dplyr::select(.pred_Sim, auto_suficiencia_financeira) %>% 
  dplyr::rename(prob=.pred_Sim) %>% 
  cutpointr(prob, auto_suficiencia_financeira, method=minimize_metric, metric=roc01)

cut_xgb<- cut_xgb$optimal_cutpoint



## CORTE SVM

cut_svm<- tune_svm %>% 
  dplyr::select(id, .predictions) %>% 
  unnest(.predictions) %>% 
  filter(cost==as.numeric(show_best(tune_svm,metric="roc_auc",n=1)[1]),
         margin==as.numeric(show_best(tune_svm,metric="roc_auc",n=1)[2]),
         rbf_sigma==as.numeric(show_best(tune_svm,metric="roc_auc",n=1)[3]),
         threshold==as.numeric(show_best(tune_svm,metric="roc_auc",n=1)[4]),
         freq_cut==as.numeric(show_best(tune_svm,metric="roc_auc",n=1)[5])) %>% 
  dplyr::select(.pred_Sim, auto_suficiencia_financeira) %>% 
  dplyr::rename(prob=.pred_Sim) %>% 
  cutpointr(prob, auto_suficiencia_financeira, method=minimize_metric, metric=roc01)

cut_svm<- cut_svm$optimal_cutpoint



## CORTE STACKING

cut_stc<- stack_ensemble_trained %>% predict(df_train, type="prob") %>% 
  cutpointr(prob, auto_suficiencia_financeira, method=minimize_metric, metric=roc01)

cut_stc<- cut_stc$optimal_cutpoint


cut_stc<- summary(df_train$auto_suficiencia_financeira)[2]/nrow(df_train)



cbind(cut_pls,
      cut_net,
      cut_rfo,
      cut_xgb,
      cut_svm,
      cut_stc)




# PREDIZENDO CLASSES (CLASSIFICACAO - DADOS TESTE)

prob_pls<- wf_pls_trained %>% predict.model_fit(df_test, type="prob")
prob_net<- wf_net_trained %>% predict(df_test, type="prob")
prob_rfo<- wf_rfo_trained %>% predict(df_test, type="prob")
prob_xgb<- wf_xgb_trained %>% predict(df_test, type="prob")
prob_svm<- wf_svm_trained %>% predict(df_test, type="prob")
prob_stc<- stack_ensemble_trained %>% predict(df_test, type="prob")

df_prob<- cbind.data.frame(df_test$DEATH_EVENT,
                           prob_net[,2],
                           prob_rfo[,2],
                           prob_xgb[,2],
                           prob_svm[,2])

colnames(df_prob)<- c("DEATH_EVENT",
                      "net",
                      "rfo",
                      "xgb",
                      "svm")

df_prob %>% head()    # VISUALIZANDO PROBABILIDADES

df_pred_class<- df_prob %>% 
  mutate(net=ifelse(net>cut_net,1,0)) %>% 
  mutate(rfo=ifelse(rfo>cut_rfo,1,0)) %>% 
  mutate(xgb=ifelse(xgb>cut_xgb,1,0)) %>% 
  mutate(svm=ifelse(svm>cut_svm,1,0)) %>% 
  mutate(across(!DEATH_EVENT, as.factor))

df_pred_class %>% head()    # VISUALIZANDO CLASSES





#####  VERIFICANDO MEDIDAS DE CLASSIFICAÇÃO  #####

medidas<- cbind(summary(conf_mat(df_pred_class, DEATH_EVENT, net))[,-2],
                summary(conf_mat(df_pred_class, DEATH_EVENT, rfo))[,3],
                summary(conf_mat(df_pred_class, DEATH_EVENT, xgb))[,3],
                summary(conf_mat(df_pred_class, DEATH_EVENT, svm))[,3])                     

colnames(medidas)<- c("medida",
                      "net",
                      "rfo",
                      "xgb",
                      "svm")

medidas



# MATRIZ DE CONFUSAO

conf_mat(df_pred_class, DEATH_EVENT, rfo)



# CURVA ROC

cbind(roc_curve(df_prob, DEATH_EVENT, net, event_level="second"),modelo="Elastic-Net") %>% 
  rbind(cbind(roc_curve(df_prob, DEATH_EVENT, rfo, event_level="second"),modelo="Random Forest")) %>% 
  rbind(cbind(roc_curve(df_prob, DEATH_EVENT, xgb, event_level="second"),modelo="XGBoosting")) %>% 
  rbind(cbind(roc_curve(df_prob, DEATH_EVENT, svm, event_level="second"),modelo="Support Vector Machine")) %>% 
  ggplot(aes(x=1-specificity, y=sensitivity, color=modelo)) + 
  geom_path() + 
  geom_abline(lty=3) + 
  coord_equal() + 
  xlab("1-Especificidade") + 
  ylab("Sensibilidade") +
  theme_bw()




# AREA ABAIXO DA CURVA ROC

auc_net<- roc_auc(df_prob, DEATH_EVENT, net, event_level="second")[3] %>% as.numeric()
auc_rfo<- roc_auc(df_prob, DEATH_EVENT, rfo, event_level="second")[3] %>% as.numeric()
auc_xgb<- roc_auc(df_prob, DEATH_EVENT, xgb, event_level="second")[3] %>% as.numeric()
auc_svm<- roc_auc(df_prob, DEATH_EVENT, svm, event_level="second")[3] %>% as.numeric()

auc<- cbind(auc_net,
            auc_rfo,
            auc_xgb,
            auc_svm)

colnames(auc)<- c("net",
                  "rfo",
                  "xgb",
                  "svm")

auc


#####  FEATURE/VARIABLE IMPORTANCE  #####

## ELASTIC-NET

explainer_net<- explain_tidymodels(model=wf_net_trained,
                                   data=dplyr::select(df_train,-ROA),
                                   y=df_train$ROA,
                                   label="Elastic-Net")

vi_net<- model_parts(explainer_net,
                     type="variable_importance")

vi_net %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))



## RANDOM FOREST

explainer_rfo<- explain_tidymodels(model=wf_rfo_trained,
                                   data=dplyr::select(df_train,-ROA),
                                   y=df_train$ROA,
                                   label="Random Forest")

vi_rfo<- model_parts(explainer_rfo,
                     type="variable_importance")

vi_rfo %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))



## XGBOOSTING

explainer_xgb<- explain_tidymodels(model=wf_xgb_trained,
                                   data=dplyr::select(df_train,-ROA),
                                   y=df_train$ROA,
                                   label="XGBoosting")

vi_xgb<- model_parts(explainer_xgb,
                     type="variable_importance")

vi_xgb %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))




## SUPPORT VECTOR MACHINE

explainer_svm<- explain_tidymodels(model=wf_svm_trained,
                                   data=dplyr::select(df_train,-ROA),
                                   y=df_train$ROA,
                                   label="Support Vector Machine")

vi_svm<- model_parts(explainer_svm,
                     type="variable_importance")

vi_svm %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))



## STACKING

tic()
explainer_stc<- explain_tidymodels(model=stack_ensemble_trained,
                                   data=dplyr::select(df_train,-auto_suficiencia_financeira),
                                   y=df_train$auto_suficiencia_financeira=="Sim",
                                   label="Stacking")
toc()


tic()
vi_stc<- model_parts(explainer_stc,
                     type="variable_importance")
toc()
# 50075.43 sec elapsed (~ 14 hr)

# saveRDS(vi_stc,"vi_stc.rds")
# vi_stc<- readRDS("vi_stc.rds")



vi_stc %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))

vi_stc %>% plot(show_boxplots=FALSE,
                title="Variable Importance",
                subtitle="",
                max_vars=10)


tic()
profile_stc<- explainer_stc %>% model_profile()
toc()


profile_stc %>% plot(geom = "aggregates")
profile_stc %>% plot(geom = "profiles")
profile_stc %>% plot(geom = "points")

profile_stc %>% plot(geom = "aggregates",
                     #title="",
                     subtitle="")

explainer_stc %>% model_profile(variables=c("SIZE","SCORE")) %>% plot(geom = "profiles")


profile_stc$agr_profiles %>% 
  data.frame() %>% 
  filter(X_vname_ %in% c("SIZE")) %>% 
  ggplot(aes(x = X_x_,
             y = X_yhat_)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  #geom_smooth(method = "loess", se = FALSE) + 
  #geom_line(data = spline_int, aes(x = x, y = y)) +
  ylab("ROA estimado") +
  xlab("SIZE") +
  #ylim(c(5,9)) +
  theme_bw()


df_SIZE<- profile_stc$agr_profiles %>% 
  data.frame() %>% 
  dplyr::filter(X_vname_ %in% c("SIZE")) %>% 
  dplyr::rename(ROA=X_yhat_) %>% 
  dplyr::rename(SIZE=X_x_) %>% 
  dplyr::select(c(ROA,SIZE))

df_SIZE %>% 
  ggplot(aes(x = SIZE,y = ROA)) +
  geom_point() +
  #geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  #geom_smooth(method = "loess", se = FALSE) + 
  geom_smooth(formula = y ~ s(x, k = 20), method = "gam", se = FALSE) + 
  ylab("ROA estimado") +
  xlab("SIZE") +
  ggtitle("Ceteris Paribus profile") + 
  theme_bw() 


df_SCORE<- profile_stc$agr_profiles %>% 
  data.frame() %>% 
  dplyr::filter(X_vname_ %in% c("SCORE")) %>% 
  dplyr::rename(ROA=X_yhat_) %>% 
  dplyr::rename(SCORE=X_x_) %>% 
  dplyr::select(c(ROA,SCORE))

df_SCORE %>% 
  ggplot(aes(x = SCORE,y = ROA)) +
  geom_point() +
  #geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  #geom_smooth(method = "loess", se = FALSE) + 
  geom_smooth(formula = y ~ s(x, k = 4), method = "gam", se = FALSE) + 
  ylab("ROA estimado") +
  xlab("SCORE") +
  ggtitle("Ceteris Paribus profile") + 
  theme_bw() 



df_HHI_s<- profile_stc$agr_profiles %>% 
  data.frame() %>% 
  dplyr::filter(X_vname_ %in% c("HHI_s")) %>% 
  dplyr::rename(ROA=X_yhat_) %>% 
  dplyr::rename(HHI_s=X_x_) %>% 
  dplyr::select(c(ROA,HHI_s))

df_HHI_s %>% 
  ggplot(aes(x = HHI_s,y = ROA)) +
  geom_point() +
  #geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  #geom_smooth(method = "loess", se = FALSE) + 
  geom_smooth(formula = y ~ s(x, k = 15), method = "gam", se = FALSE) + 
  ylab("ROA estimado") +
  xlab("HHI_s") +
  ggtitle("Ceteris Paribus profile") + 
  theme_bw() 

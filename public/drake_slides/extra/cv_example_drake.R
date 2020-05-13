library(pacman)

pacman::p_load(magrittr, tidyverse, tidymodels, broom, drake)

office_plan <- drake_plan(
  ratings_raw = target(
    readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-03-17/office_ratings.csv")),
  remove_regex = target(
  "[:punct:]|[:digit:]|parts |part |the |and"),
  office_ratings = target(
    ratings_raw %>%
    transmute(
      episode_name = str_to_lower(title),
      episode_name = str_remove_all(episode_name, remove_regex),
      episode_name = str_trim(episode_name),
      imdb_rating)),
  office_info = target(
    schrute::theoffice %>%
      mutate(
        season = as.numeric(season),
        episode = as.numeric(episode),
        episode_name = str_to_lower(episode_name),
        episode_name = str_remove_all(episode_name, remove_regex),
        episode_name = str_trim(episode_name)) %>%
    select(
      season, episode, episode_name,
      director, writer, character)),
  characters = target(
    office_info %>%
      count(episode_name, character) %>%
      add_count(character, wt = n, name = "character_count") %>%
      filter(character_count > 800) %>%
      select(-character_count) %>%
      pivot_wider(
        names_from = character,
        values_from = n,
        values_fill = list(n = 0))),
  creators = target(
    office_info %>%
      distinct(episode_name, director, writer) %>%
      pivot_longer(
        director:writer,
        names_to = "role", values_to = "person") %>%
      separate_rows(person, sep = ";") %>%
      add_count(person) %>%
      filter(n > 10) %>%
      distinct(episode_name, person) %>%
      mutate(person_value = 1) %>%
      pivot_wider(
      names_from = person,
      values_from = person_value,
      values_fill = list(person_value = 0))),
  office = target(
    office_info %>%
      distinct(season, episode, episode_name) %>%
      inner_join(characters) %>%
      inner_join(creators) %>%
      inner_join(office_ratings %>%
        select(episode_name, imdb_rating)) %>%
      janitor::clean_names()),
  office_boxplot = target(
    office %>%
      ggplot(aes(episode, imdb_rating, fill = as.factor(episode))) +
      geom_boxplot(show.legend = FALSE)),
  office_split = target(
    initial_split(office, strata = season)),
  office_train = target(training(office_split)),
  office_test = target(testing(office_split)),
  office_rec = target(
    recipe(imdb_rating ~ ., data = office_train) %>%
      update_role(episode_name, new_role = "ID") %>%
      step_zv(all_numeric(), -all_outcomes()) %>%
      step_normalize(all_numeric(), -all_outcomes())),
  office_prep = target(
    office_rec %>%
      prep(strings_as_factors = FALSE)),
  lasso_spec = target(
    linear_reg(penalty = 0.1, mixture = 1) %>%
      set_engine("glmnet")),
  wf = target(
    workflows::workflow() %>%
      add_recipe(office_rec)),
  lasso_fit = target(
    wf %>%
      add_model(lasso_spec) %>%
      parsnip::fit(data = office_train)  ),
  office_boot = target(bootstraps(office_train, strata = season)),
  office_vcv = target(
    vfold_cv(office_train, strate = season, v = 10, repeats = 3)),
  tune_spec = target(
    linear_reg(penalty = tune(), mixture = 1) %>%
      set_engine("glmnet")),
  lambda_grid = target(grid_regular(penalty(), levels = 50)),
  # alpha_lambda_grid = target(
  #   grid_regular(penalty(), mixture(), levels = 50)),
  lasso_grid = target(
    tune_grid(
      wf %>% add_model(tune_spec),
      resamples = office_boot,
      grid = lambda_grid)),
  metrics = target(lasso_grid %>% collect_metrics()),
  error_plot = target(
    metrics %>%
      ggplot(aes(penalty, mean, color = .metric)) +
      geom_errorbar(aes(
        ymin = mean - std_err,
        ymax = mean + std_err),
      alpha = 0.5
      ) +
      geom_line(size = 1.5) +
      facet_wrap(~.metric, scales = "free", nrow = 2) +
      scale_x_log10() +
      theme(legend.position = "none")),
  lowest_rmse = target(
    lasso_grid %>%
      select_best("rmse")),
  final_lasso = target(
    finalize_workflow(
      wf %>% add_model(tune_spec), lowest_rmse)),
  vip_plot = target(
  final_lasso %>%
    fit(office_train) %>%
    pull_workflow_fit() %>%
    vip::vi(lambda = lowest_rmse$penalty) %>%
    mutate(
      Importance = abs(Importance),
      Variable = fct_reorder(Variable, Importance)) %>%
    ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
    geom_col() +
    scale_x_continuous(expand = c(0, 0)) +
    labs(y = NULL)),
  final_lasso_metric = target(
    last_fit(final_lasso, office_split) %>%
    collect_metrics()))

make(office_plan)
  
viz <- vis_drake_graph(office_plan)
htmlwidgets::saveWidget(viz,
  file = here::here("extra/office_plan.html"))

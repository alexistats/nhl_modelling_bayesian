import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pytensor.tensor as pt
import logging
import os
from data_prep_bayesian import process_player_data
from data_prep_schedules import format_team_schedule

## Define input and output directories
INPUT_DIR = os.path.join('inputs')
OUTPUT_DIR = os.path.join('outputs')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_team_mapping(mapping_file):
    team_mapping = pd.read_csv(mapping_file)
    full_name_to_id = dict(zip(team_mapping['team_name'], team_mapping['id']))
    abbrev_to_id = dict(zip(team_mapping['abbreviation'], team_mapping['id']))
    full_name_to_abbrev = dict(zip(team_mapping['team_name'], team_mapping['abbreviation']))
    return full_name_to_id, abbrev_to_id, full_name_to_abbrev

def load_and_prepare_data(player_name, player_data_file, schedule_file, team_mapping_file):
    # Load data
    player_df = pd.read_csv(os.path.join(INPUT_DIR, player_data_file))
    schedule = pd.read_csv(os.path.join(INPUT_DIR, schedule_file))
    full_name_to_id, abbrev_to_id, full_name_to_abbrev = load_team_mapping(os.path.join(INPUT_DIR, team_mapping_file))

    # Data preparation
    schedule['Home'] = schedule['Vs'].str.contains('vs').astype(int)

    # Prepare schedule data
    schedule['Home'] = schedule['Vs'].str.contains('vs').astype(int)
    schedule['Teams_2'] = schedule['Opponent'].map(full_name_to_abbrev)
    schedule['opp_int'] = schedule['Opponent'].map(full_name_to_id)

    # Prepare player data
    player_df['opp_int'] = player_df['opponent'].map(abbrev_to_id)
    # Create season2 field
    unique_seasons = sorted(player_df['season'].unique())
    season_mapping = {season: i+1 for i, season in enumerate(unique_seasons)}
    player_df['season2'] = player_df['season'].map(season_mapping)

    df_current_season = player_df[player_df['season2'] == player_df['season2'].max()]
    N_games = len(df_current_season)
    curr_assists = df_current_season['Assists'].sum()
    curr_goals = df_current_season['Goals'].sum()
    remaining_schedule = schedule.iloc[N_games:]

    return player_df, remaining_schedule, curr_assists, curr_goals


def build_and_sample_model(player_df, remaining_schedule, curr_assists, curr_goals, player_name):
    with pm.Model() as model:
        num_seasons = int(player_df['season2'].max())  # Convert to Python int
        # Priors
        mu_assists_t = pm.GaussianRandomWalk('mu_assists_t', sigma=0.25, shape=num_seasons)
        mu_goals_t = pm.GaussianRandomWalk('mu_goals_t', sigma=0.25, shape=num_seasons)

        sigma_assists_team = pm.HalfNormal('sigma_assists_team', sigma=0.08)
        sigma_goals_team = pm.HalfNormal('sigma_goals_team', sigma=0.08)

        rho_a = pm.TruncatedNormal('rho_a', mu=0, sigma=0.3, lower=0, upper=1)
        rho_g = pm.TruncatedNormal('rho_g', mu=0, sigma=0.3, lower=0, upper=1)

        mu_assists_team = pm.AR('mu_assists_team', rho=rho_a, sigma=sigma_assists_team,
                                shape=(31, num_seasons))
        mu_goals_team = pm.AR('mu_goals_team', rho=rho_g, sigma=sigma_goals_team,
                              shape=(31, num_seasons))

        b_home = pm.Normal('b_home', mu=0, sigma=0.5, shape=3)
        beta_goal = pm.Normal('beta_goal', mu=0, sigma=0.5)

        # Likelihood
        mn_a = (pt.subtensor.take(mu_assists_t, player_df['season2'].astype('int32') - 1) +
                pt.subtensor.take(mu_assists_team,
                                  (player_df['opp_int'].astype('int32') - 1,
                                   player_df['season2'].astype('int32') - 1)) +
                player_df['Home'].values * b_home[1] +
                player_df['Goals'].values * beta_goal)
        mn_g = (pt.subtensor.take(mu_goals_t, player_df['season2'].astype('int32') - 1) +
                pt.subtensor.take(mu_goals_team,
                                  (player_df['opp_int'].astype('int32') - 1,
                                   player_df['season2'].astype('int32') - 1)) +
                player_df['Home'].values * b_home[2])

        assists = pm.Poisson('assists', mu=pm.math.exp(mn_a.sum(axis=0)), observed=player_df['Assists'])
        goals = pm.Poisson('goals', mu=pm.math.exp(mn_g.sum(axis=0)), observed=player_df['Goals'])

        # Predictions
        current_season = int(player_df['season2'].max())
        pred_assists = pm.Poisson('pred_assists', mu=pt.exp(
            mu_assists_t[current_season - 1] + mu_assists_team[
                remaining_schedule['opp_int'].astype('int32') - 1, current_season - 1] +
            remaining_schedule['Home'].values * b_home[1]
        ), shape=len(remaining_schedule))

        pred_goals = pm.Poisson('pred_goals', mu=pt.exp(
            mu_goals_t[current_season - 1] + mu_goals_team[
                remaining_schedule['opp_int'].astype('int32') - 1, current_season - 1] +
            remaining_schedule['Home'].values * b_home[2]
        ), shape=len(remaining_schedule))

        pred_total_assists = pm.Deterministic('pred_total_assists', curr_assists + pred_assists.sum())
        pred_total_goals = pm.Deterministic('pred_total_goals', curr_goals + pred_goals.sum())
        pred_total_points = pm.Deterministic('pred_total_points', pred_total_assists + pred_total_goals)

    # Sampling
    with model:
        trace = pm.sample(100, tune=100, return_inferencedata=True, progressbar=True)
    az.to_netcdf(trace, os.path.join(OUTPUT_DIR, f"{player_name}_model_results.nc"))

    return trace


def analyze_results(trace, player_name):
    point_draws = az.extract(trace, var_names=['pred_total_points', 'pred_total_goals', 'pred_total_assists'],
                             combined=True)

    print(f"Results for {player_name}:")
    print(f"Expected total points: {point_draws['pred_total_points'].mean().item():.1f}")
    print(f"93% HDI for total points: {az.hdi(point_draws['pred_total_points'].values.flatten(), hdi_prob=0.93)}")
    print(f"93% HDI for total goals: {az.hdi(point_draws['pred_total_goals'].values.flatten(), hdi_prob=0.93)}")
    print(f"93% HDI for total assists: {az.hdi(point_draws['pred_total_assists'].values.flatten(), hdi_prob=0.93)}")

    plt.figure(figsize=(10, 6))
    plt.hist(point_draws['pred_total_points'].values.flatten(), bins=50, density=True, alpha=0.7)
    plt.axvline(point_draws['pred_total_points'].mean().item(), color='red', linestyle='dashed')
    plt.title(f"{player_name} Posterior Predicted Points")
    plt.xlabel("Total Points")
    plt.ylabel("Density")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{player_name}_posterior_predicted_points.png"))
    plt.show()

    plt.close()

    return point_draws


def main(player_name, player_data_file, schedule_file, team_mapping_file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting analysis for {player_name}")

    player_df, remaining_schedule, curr_assists, curr_goals = load_and_prepare_data(player_name, player_data_file,
                                                                                    schedule_file, team_mapping_file)
    logging.info("Data preparation complete, starting model building and sampling")

    trace = build_and_sample_model(player_df, remaining_schedule, curr_assists, curr_goals, player_name)
    logging.info("Model sampling complete, analyzing results")

    results = analyze_results(trace, player_name)
    logging.info("Analysis completed successfully")

    return results


if __name__ == '__main__':
    player_names = ['Sidney Crosby', 'Jack Hughes', 'Cale Makar', 'Kirill Kaprizov', 'JT Miller', 'Matthew Tkachuk']
    team_names = ['penguins', 'devils', 'avalanche', 'wild', 'canucks', 'panthers']
    for player_name, team_name in zip(player_names, team_names):
        schedule_df = format_team_schedule(team_name)
        player_df = process_player_data(player_name, team_name)
        player_data_file = f"{player_name}_df.csv"  # Change this to the appropriate file for each player
        schedule_file = f"{team_name}_schedule_2425_formatted.csv"  # This might need to change depending on the player's team
        team_mapping_file = "team_mapping.csv"
        main(player_name, player_data_file, schedule_file, team_mapping_file)
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pytensor
import pytensor.tensor as pt
import pytensor.tensor as at
import logging



def build_and_sample_model():
    # Model definition
    with pm.Model() as model:
        # Priors
        mu_assists_t = pm.GaussianRandomWalk('mu_assists_t', sigma=0.25, shape=7)
        mu_goals_t = pm.GaussianRandomWalk('mu_goals_t', sigma=0.25, shape=7)

        sigma_assists_team = pm.HalfNormal('sigma_assists_team', sigma=0.08)
        sigma_goals_team = pm.HalfNormal('sigma_goals_team', sigma=0.08)

        rho_a = pm.TruncatedNormal('rho_a', mu=0, sigma=0.3, lower=0, upper=1)
        rho_g = pm.TruncatedNormal('rho_g', mu=0, sigma=0.3, lower=0, upper=1)

        mu_assists_team = pm.AR('mu_assists_team', rho=rho_a, sigma=sigma_assists_team, shape=(31, 7))
        mu_goals_team = pm.AR('mu_goals_team', rho=rho_g, sigma=sigma_goals_team, shape=(31, 7))

        b_home = pm.Normal('b_home', mu=0, sigma=0.5, shape=3)
        beta_goal = pm.Normal('beta_goal', mu=0, sigma=0.5)

        # Likelihood
        import pytensor.tensor as pt

        mn_a = (pt.subtensor.take(mu_assists_t, mcdavid_df['season2'].astype('int32') - 1) +
                pt.subtensor.take(mu_assists_team,
                                  (mcdavid_df['opp_int'].astype('int32') - 1,
                                   mcdavid_df['season2'].astype('int32') - 1)) +
                mcdavid_df['Home'].values * b_home[1] +
                mcdavid_df['Goals'].values * beta_goal)
        mn_g = (pt.subtensor.take(mu_goals_t, mcdavid_df['season2'].astype('int32') - 1) +
                pt.subtensor.take(mu_goals_team,
                                  (mcdavid_df['opp_int'].astype('int32') - 1,
                                   mcdavid_df['season2'].astype('int32') - 1)) +
                mcdavid_df['Home'].values * b_home[2])

        # print("Shape of mn_a:", mn_a.eval().shape)
        # print("Shape of mn_g:", mn_g.eval().shape)
        # print("Shape of mcdavid_df['Assists']:", mcdavid_df['Assists'].shape)
        # print("Shape of mcdavid_df['Goals']:", mcdavid_df['Goals'].shape)

        assists = pm.Poisson('assists', mu=pm.math.exp(mn_a.sum(axis=0)), observed=mcdavid_df['Assists'])
        goals = pm.Poisson('goals', mu=pm.math.exp(mn_g.sum(axis=0)), observed=mcdavid_df['Goals'])

        # Predictions
        pred_assists = pm.Poisson('pred_assists', mu=at.exp(
            mu_assists_t[6] + mu_assists_team[oilers2['opp_int'].astype('int32') - 1, 6] +
            oilers2['Home'].values * b_home[1]
        ), shape=len(oilers2))

        pred_goals = pm.Poisson('pred_goals', mu=at.exp(
            mu_goals_t[6] + mu_goals_team[oilers2['opp_int'].astype('int32') - 1, 6] +
            oilers2['Home'].values * b_home[2]
        ), shape=len(oilers2))

        pred_total_assists = pm.Deterministic('pred_total_assists', curr_assists + pred_assists.sum())
        pred_total_goals = pm.Deterministic('pred_total_goals', curr_goals + pred_goals.sum())
        pred_total_points = pm.Deterministic('pred_total_points', pred_total_assists + pred_total_goals)

    # Sampling
    with model:
        ## trace = pm.sample(2000, tune=2000, return_inferencedata=True, progressbar=True)
        trace = pm.sample(100, tune=100, return_inferencedata=True, progressbar=True)

    az.to_netcdf(trace, "mcdavid_model_results.nc")

    return trace

def analyze_results(trace, mdavid_df, oilers2):
    # Results analysis
    point_draws = az.extract(trace, var_names=['pred_total_points', 'pred_total_goals', 'pred_total_assists'], combined=True)
    return point_draws


if __name__ == '__main__':
    from fastprogress import fastprogress

    fastprogress.printing = lambda: True
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # At the start of your script
    logging.info("Starting the script")


    # Load data
    mcdavid_df = pd.read_csv("McDavid_df.csv")
    oilers_schedule = pd.read_csv("oilers_schedule_2122.csv")
    logging.info("Data loaded successfully")
    # Data preparation
    oilers_schedule['Home'] = oilers_schedule['Vs'].str.contains('vs').astype(int)
    teams = oilers_schedule['Opponent'].unique()
    teams2 = ["VAN", "CGY", "ANA", "ARI", "VGK", "PHI", "SEA", "NSH", "NYR", "DET", "BOS", "BUF", "STL", "WPG", "CHI",
              "DAL", "PIT", "LAK", "MIN", "CAR", "TOR", "CBJ", "SJS", "NJD", "NYI", "OTT", "FLA", "MTL", "WSH", "TBL",
              "COL"]
    oilers_schedule['Teams_2'] = pd.Categorical(oilers_schedule['Opponent']).rename_categories(dict(zip(teams, teams2)))
    oilers_schedule['opp_int'] = pd.Categorical(oilers_schedule['Teams_2']).codes + 1

    df_s7 = mcdavid_df[mcdavid_df['season2'] == 7]
    N_games = len(df_s7)
    curr_assists = df_s7['Assists'].sum()
    curr_goals = df_s7['Goals'].sum()
    oilers2 = oilers_schedule.iloc[N_games:]
    logging.info("Data preparation complete, trace starting")
    trace = build_and_sample_model()
    logging.info("Trace complete, analyzing results")
    # When the model already exists,  load from storage
    #trace = az.from_netcdf("mcdavid_model_results.nc")
    results = analyze_results(trace, mcdavid_df, oilers2)
    print(f"Expected total points: {results['pred_total_points'].mean().item():.1f}")
    print(f"93% HDI for total points: {az.hdi(results['pred_total_points'].values.flatten(), hdi_prob=0.93)}")
    print(f"93% HDI for total goals: {az.hdi(results['pred_total_goals'].values.flatten(), hdi_prob=0.93)}")
    print(f"93% HDI for total assists: {az.hdi(results['pred_total_assists'].values.flatten(), hdi_prob=0.93)}")

    plt.figure(figsize=(10, 6))
    plt.hist(results['pred_total_points'].values.flatten(), bins=50, density=True, alpha=0.7)
    plt.axvline(results['pred_total_points'].mean().item(), color='red', linestyle='dashed')
    plt.title("Connor McDavid Posterior Predicted Points")
    plt.xlabel("Total Points")
    plt.ylabel("Density")
    plt.show()
    logging.info("Script completed successfully")


import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('kkupfl_scoring_2018_2023_preprocessed.csv')

# Filter data for Connor McDavid
mcdavid_data = df[df['Player Name'] == 'Connor McDavid']


def build_and_sample_model():
    with pm.Model() as model:
        # Player-specific intercept
        alpha = pm.Normal('alpha', mu=0, sigma=10)

        # Slopes for each feature
        beta_pos_d = pm.Normal('beta_Pos_D', mu=0, sigma=10)
        beta_pos_lw = pm.Normal('beta_Pos_LW', mu=0, sigma=10)
        beta_pos_rw = pm.Normal('beta_Pos_RW', mu=0, sigma=10)
        beta_games = pm.Normal('beta_GP', mu=0, sigma=10)
        beta_goals = pm.Normal('beta_G', mu=0, sigma=10)
        beta_assists = pm.Normal('beta_A', mu=0, sigma=10)
        beta_shots = pm.Normal('beta_SOG', mu=0, sigma=10)
        beta_blocks = pm.Normal('beta_BS', mu=0, sigma=10)
        beta_hits = pm.Normal('beta_Hits', mu=0, sigma=10)
        beta_shg = pm.Normal('beta_SHG', mu=0, sigma=10)
        beta_sha = pm.Normal('beta_SHA', mu=0, sigma=10)
        beta_ppg = pm.Normal('beta_PPG', mu=0, sigma=10)
        beta_ppa = pm.Normal('beta_PPA', mu=0, sigma=10)

        # Expected value of fantasy points
        mu = (alpha +
              beta_pos_d * mcdavid_data['Pos_D_norm'] +
              beta_pos_lw * mcdavid_data['Pos_LW_norm'] +
              beta_pos_rw * mcdavid_data['Pos_RW_norm'] +
              beta_games * mcdavid_data['GP_norm'] +
              beta_goals * mcdavid_data['G_norm'] +
              beta_assists * mcdavid_data['A_norm'] +
              beta_shots * mcdavid_data['SOG_norm'] +
              beta_blocks * mcdavid_data['BS_norm'] +
              beta_hits * mcdavid_data['Hits_norm'] +
              beta_shg * mcdavid_data['SHG_norm'] +
              beta_sha * mcdavid_data['SHA_norm'] +
              beta_ppg * mcdavid_data['PPG_norm'] +
              beta_ppa * mcdavid_data['PPA_norm']
              )

        # Model error
        sigma = pm.HalfNormal('sigma', sigma=10)

        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=mcdavid_data['fantasy_points'])

        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    return trace


def analyze_mcdavid(trace, data):
    # Extract posterior samples
    alpha_samples = trace.posterior['alpha'].values.flatten()

    # Calculate expected fantasy points for the last season data
    last_season_data = data.iloc[-1]
    expected_points = (
            alpha_samples.mean() +
            trace.posterior['beta_Pos_D'].mean().values * last_season_data['Pos_D_norm'] +
            trace.posterior['beta_Pos_LW'].mean().values * last_season_data['Pos_LW_norm'] +
            trace.posterior['beta_Pos_RW'].mean().values * last_season_data['Pos_RW_norm'] +
            trace.posterior['beta_GP'].mean().values * last_season_data['GP_norm'] +
            trace.posterior['beta_G'].mean().values * last_season_data['G_norm'] +
            trace.posterior['beta_A'].mean().values * last_season_data['A_norm'] +
            trace.posterior['beta_SOG'].mean().values * last_season_data['SOG_norm'] +
            trace.posterior['beta_BS'].mean().values * last_season_data['BS_norm'] +
            trace.posterior['beta_Hits'].mean().values * last_season_data['Hits_norm'] +
            trace.posterior['beta_SHG'].mean().values * last_season_data['SHG_norm'] +
            trace.posterior['beta_SHA'].mean().values * last_season_data['SHA_norm'] +
            trace.posterior['beta_PPG'].mean().values * last_season_data['PPG_norm'] +
            trace.posterior['beta_PPA'].mean().values * last_season_data['PPA_norm']
    )

    # Calculate credible interval
    credible_interval = pm.stats.hpd(alpha_samples)

    # Calculate variance
    variance = np.var(alpha_samples)

    return {
        'player': 'Connor McDavid',
        'expected_points': expected_points,
        'credible_interval': credible_interval,
        'variance': variance
    }


if __name__ == '__main__':
    # Build and sample the model
    trace = build_and_sample_model()

    # Analyze McDavid's performance
    result = analyze_mcdavid(trace, mcdavid_data)

    # Print results
    print(f"Analysis for {result['player']}:")
    print(f"Expected Points: {result['expected_points']:.2f}")
    print(f"95% Credible Interval: ({result['credible_interval'][0]:.2f}, {result['credible_interval'][1]:.2f})")
    print(f"Variance: {result['variance']:.2f}")

    # Plot the posterior distribution of expected points
    plt.figure(figsize=(10, 6))
    pm.plot_posterior(trace, var_names=['alpha'], rope=(-1, 1))
    plt.title(f"Posterior Distribution of Expected Points for Connor McDavid")
    plt.xlabel("Fantasy Points")
    plt.show()
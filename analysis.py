def analyze_player(player_name, trace, data):
    player_data = data[data['player'] == player_name]
    player_index = player_data['player'].astype('category').cat.codes.iloc[0]

    # Extract posterior samples for the player
    alpha_samples = trace.posterior['alpha'][:, :, player_index].values.flatten()

    # Calculate expected fantasy points for the player's last season data
    last_season_data = player_data.iloc[-1]
    expected_points = (
            alpha_samples.mean() +
            trace.posterior['beta_season'].mean().values * last_season_data['season_norm'] +
            trace.posterior['beta_games'].mean().values * last_season_data['games_played_norm'] +
            trace.posterior['beta_goals'].mean().values * last_season_data['goals_norm'] +
            trace.posterior['beta_assists'].mean().values * last_season_data['assists_norm'] +
            trace.posterior['beta_shots'].mean().values * last_season_data['shots_norm'] +
            trace.posterior['beta_blocks'].mean().values * last_season_data['blocks_norm'] +
            trace.posterior['beta_hits'].mean().values * last_season_data['hits_norm'] +
            trace.posterior['beta_shp'].mean().values * last_season_data['short_handed_points_norm']
    )

    # Calculate credible interval
    credible_interval = pm.stats.hpd(alpha_samples)

    # Calculate variance
    variance = np.var(alpha_samples)

    return {
        'player': player_name,
        'expected_points': expected_points,
        'credible_interval': credible_interval,
        'variance': variance
    }


# Analyze all players
results = [analyze_player(player, trace, df) for player in df['player'].unique()]

# Sort players by expected points
results.sort(key=lambda x: x['expected_points'], reverse=True)

# Print top 10 players
for i, result in enumerate(results[:10], 1):
    print(f"{i}. {result['player']}:")
    print(f"   Expected Points: {result['expected_points']:.2f}")
    print(f"   95% Credible Interval: ({result['credible_interval'][0]:.2f}, {result['credible_interval'][1]:.2f})")
    print(f"   Variance: {result['variance']:.2f}")
    print()
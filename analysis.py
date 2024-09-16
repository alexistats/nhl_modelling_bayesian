import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join('outputs')

def load_model(file_name):
    return az.from_netcdf(os.path.join(OUTPUT_DIR, file_name))

def calculate_probability(player1_data, player2_data):
    return (player1_data[:, None] > player2_data).mean()
def plot_distributions(player1_data, player2_data, player1_name, player2_name, stat_name):
    plt.figure(figsize=(10, 6))
    plt.hist(player1_data, bins=50, alpha=0.5, label=player1_name)
    plt.hist(player2_data, bins=50, alpha=0.5, label=player2_name)
    plt.xlabel(stat_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {stat_name}')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{stat_name.lower()}_distribution.png'))
    plt.close()

def calculate_lower_bound(data, percentile=0.05):
    return np.percentile(data, percentile)

# Load model results
hughes_model = load_model('Jack Hughes_model_results.nc')
makar_model = load_model('Cale Makar_model_results.nc')

# Extract relevant data
hughes_points = hughes_model.posterior['pred_total_points'].values.flatten()
hughes_goals = hughes_model.posterior['pred_total_goals'].values.flatten()
hughes_assists = hughes_model.posterior['pred_total_assists'].values.flatten()

makar_points = makar_model.posterior['pred_total_points'].values.flatten()
makar_goals = makar_model.posterior['pred_total_goals'].values.flatten()
makar_assists = makar_model.posterior['pred_total_assists'].values.flatten()

# Calculate probabilities
prob_points = calculate_probability(hughes_points, makar_points)
prob_goals = calculate_probability(hughes_goals, makar_goals)
prob_assists = calculate_probability(hughes_assists, makar_assists)

print(f"Probability Hughes > Makar (Points): {prob_points:.3f}")
print(f"Probability Hughes > Makar (Goals): {prob_goals:.3f}")
print(f"Probability Hughes > Makar (Assists): {prob_assists:.3f}")

# Plot distributions
plot_distributions(hughes_points, makar_points, 'Hughes', 'Makar', 'Points')
plot_distributions(hughes_goals, makar_goals, 'Hughes', 'Makar', 'Goals')
plot_distributions(hughes_assists, makar_assists, 'Hughes', 'Makar', 'Assists')

# Calculate 95% lower bounds
hughes_points_lb = calculate_lower_bound(hughes_points)
hughes_goals_lb = calculate_lower_bound(hughes_goals)
hughes_assists_lb = calculate_lower_bound(hughes_assists)

makar_points_lb = calculate_lower_bound(makar_points)
makar_goals_lb = calculate_lower_bound(makar_goals)
makar_assists_lb = calculate_lower_bound(makar_assists)

print(f"\nHughes 95% Lower Bounds:")
print(f"Points: {hughes_points_lb:.1f}")
print(f"Goals: {hughes_goals_lb:.1f}")
print(f"Assists: {hughes_assists_lb:.1f}")

print(f"\nMakar 95% Lower Bounds:")
print(f"Points: {makar_points_lb:.1f}")
print(f"Goals: {makar_goals_lb:.1f}")
print(f"Assists: {makar_assists_lb:.1f}")
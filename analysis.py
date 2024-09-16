import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join('outputs')
player_1_name = 'Kirill Kaprizov'
player_2_name = 'JT Miller'

def load_model(file_name):
    return az.from_netcdf(os.path.join(OUTPUT_DIR, file_name))
def calculate_and_print_hdi(data, player_name, stat_name):
    hdi = az.hdi(data, hdi_prob=0.95)
    print(f"{player_name} 95% HDI for {stat_name}: [{hdi[0]:.1f}, {hdi[1]:.1f}]")
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

def calculate_lower_bound(data, percentile=5):
    return np.percentile(data, percentile)

# Load model results
player_1_model = load_model(f'{player_1_name}_model_results.nc')
player_2_model = load_model(f'{player_2_name}_model_results.nc')

# Extract relevant data
player_1_points = player_1_model.posterior['pred_total_points'].values.flatten()
player_2_points = player_2_model.posterior['pred_total_points'].values.flatten()

player_1_goals = player_1_model.posterior['pred_total_goals'].values.flatten()
player_2_goals = player_2_model.posterior['pred_total_goals'].values.flatten()

player_1_assists = player_1_model.posterior['pred_total_assists'].values.flatten()
player_2_assists = player_2_model.posterior['pred_total_assists'].values.flatten()

# Calculate probabilities
prob_points = calculate_probability(player_1_points, player_2_points)
prob_goals = calculate_probability(player_1_goals, player_2_goals)
prob_assists = calculate_probability(player_1_assists, player_2_assists)

print(f"Probability {player_1_name} > {player_2_name} (Points): {prob_points:.3f}")
print(f"Probability {player_1_name} > {player_2_name} (Goals): {prob_goals:.3f}")
print(f"Probability {player_1_name} > {player_2_name} (Assists): {prob_assists:.3f}")

# Plot distributions
plot_distributions(player_1_points, player_2_points, player_1_name, player_2_name, 'Points')
plot_distributions(player_1_goals, player_2_goals, player_1_name, player_2_name, 'Goals')
plot_distributions(player_1_assists, player_2_assists, player_1_name, player_2_name, 'Assists')

# Calculate 95% lower bounds
player_1_points_lb = calculate_lower_bound(player_1_points)
player_1_goals_lb = calculate_lower_bound(player_1_goals)
player_1_assists_lb = calculate_lower_bound(player_1_assists)

player_2_points_lb = calculate_lower_bound(player_2_points)
player_2_goals_lb = calculate_lower_bound(player_2_goals)
player_2_assists_lb = calculate_lower_bound(player_2_assists)

player_1_points_ub = calculate_lower_bound(player_1_points, 95)
player_1_goals_ub = calculate_lower_bound(player_1_goals, 95)
player_1_assists_ub = calculate_lower_bound(player_1_assists, 95)

player_2_points_ub = calculate_lower_bound(player_2_points, 95)
player_2_goals_ub = calculate_lower_bound(player_2_goals, 95)
player_2_assists_ub = calculate_lower_bound(player_2_assists, 95)

print(f"\n95% Lower Bounds:")
print(f"{player_1_name} Points: {player_1_points_lb:.1f}")
print(f"{player_2_name} Points: {player_2_points_lb:.1f}")

print(f"{player_1_name} Goals: {player_1_goals_lb:.1f}")
print(f"{player_2_name} Goals: {player_2_goals_lb:.1f}")

print(f"{player_1_name} Assists: {player_1_assists_lb:.1f}")
print(f"{player_2_name} Assists: {player_2_assists_lb:.1f}")


print(f"\n 95% Upper Bounds:")
print(f"{player_1_name} Points: {player_1_points_ub:.1f}")
print(f"{player_2_name} Points: {player_2_points_ub:.1f}")

print(f"{player_1_name} Goals: {player_1_goals_ub:.1f}")
print(f"{player_2_name} Goals: {player_2_goals_ub:.1f}")

print(f"{player_1_name} Assists: {player_1_assists_ub:.1f}")
print(f"{player_2_name} Assists: {player_2_assists_ub:.1f}")


# Calculate and print 95% HDI
calculate_and_print_hdi(player_1_points, player_1_name, "Points")
calculate_and_print_hdi(player_2_points, player_2_name, "Points")

calculate_and_print_hdi(player_1_goals, player_1_name, "Goals")
calculate_and_print_hdi(player_2_goals, player_2_name, "Goals")

calculate_and_print_hdi(player_1_assists, player_1_name, "Assists")
calculate_and_print_hdi(player_2_assists, player_2_name, "Assists")


# Calculate variances
player_1_points_var = np.std(player_1_points)
player_1_goals_var = np.std(player_1_goals)
player_1_assists_var = np.std(player_1_assists)

player_2_points_var = np.std(player_2_points)
player_2_goals_var = np.std(player_2_goals)
player_2_assists_var = np.std(player_2_assists)

# Print variances
print("\nVariances:")
print(f"{player_1_name} Points Variance: {player_1_points_var:.2f}")
print(f"\n{player_2_name} Points Variance: {player_2_points_var:.2f}")

print(f"{player_1_name} Goals Variance: {player_1_goals_var:.2f}")
print(f"{player_2_name} Goals Variance: {player_2_goals_var:.2f}")

print(f"{player_1_name} Assists Variance: {player_1_assists_var:.2f}")
print(f"{player_2_name} Assists Variance: {player_2_assists_var:.2f}")


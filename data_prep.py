import pandas as pd
import numpy as np

# Assuming your data is in a CSV file
df = pd.read_csv('kkupfl_scoring_2018_2023_input.csv')

# Calculate fantasy points and fantasy points per game (fppg)
df['fantasy_points'] = (
    4.5 * df['G'] +
    3 * df['A'] +
    0.5 * df['SOG'] +
    0.5 * df['BS'] +
    0.25 * df['Hits'] +
    2 * (df['SHG'] + df['SHA'])
)

df['fppg'] = (
    4.5 * df['G'] +
    3 * df['A'] +
    0.5 * df['SOG'] +
    0.5 * df['BS'] +
    0.25 * df['Hits'] +
    2 * (df['SHG'] + df['SHA'])
)/df['GP']


# filter out goalies
df = df[df['Pos'] != 'G']

# Drop the Player Name = Sebastian Aho and Pos = 'D'
df = df[~((df['Player Name'] == 'Sebastian Aho') & (df['Pos'] == 'D'))]


# filter out players with less than 20 games played
df = df[df['GP'] >= 20]

## Filter out players that do not have 3 consecutive seasons of play
# Get the number of seasons played by each player
seasons_played = df.groupby('Player Name')['Year'].nunique()
# Filter out players with less than 3 seasons
df = df[df['Player Name'].isin(seasons_played[seasons_played >= 3].index)]

# drop unnecessary columns: W, GA, SV and SO
df = df.drop(['Team', 'W', 'GA', 'SV', 'SO'], axis=1)

# Drop rows with missing values
df = df.dropna()

# one hot encode the position column, don't keep all the positions
df = pd.get_dummies(df, columns=['Pos'], drop_first=True)

## Normalize the dummy variables
for col in ['Pos_D', 'Pos_LW', 'Pos_RW']:
    df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()

# Normalize continuous variables
for col in ['GP', 'G', 'A', 'SOG', 'PPG', 'PPA', 'SHG', 'SHA', 'Hits', 'BS']:
    df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()

## Drop the non normalized columns
df = df.drop(['GP', 'G', 'A', 'Pts', 'SOG', 'PPG', 'PPA', 'SHG', 'SHA', 'Hits', 'BS', 'Pos_D', 'Pos_LW', 'Pos_RW'], axis=1)

# Save the preprocessed data
df.to_csv('kkupfl_scoring_2018_2023_preprocessed.csv', index=False)


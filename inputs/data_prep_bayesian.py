import pandas as pd
import datetime

# Read the Jack Hughes data
df = pd.read_csv('jack_hughes_df.txt', delimiter=',')

# Function to convert age to birth date
def age_to_birthdate(age_str, game_date):
    years, days = map(int, age_str.split('-'))
    game_date = datetime.datetime.strptime(game_date, '%Y-%m-%d')
    birth_date = game_date - datetime.timedelta(days=days) - datetime.timedelta(days=years*365)
    return birth_date

# Get birth date from the first game
birth_date = age_to_birthdate(df['Age'].iloc[0], df['Date'].iloc[0])

# Function to calculate season
def get_season(date):
    year = int(date[:4])
    month = int(date[5:7])
    if month >= 9:
        return f"{year}{year+1-2000}"
    else:
        return f"{year-1}{year-2000}"

# Create new dataframe with desired columns
new_df = pd.DataFrame()

new_df['Date'] = df['Date']
new_df['Team'] = 'NJD'
new_df['Goals'] = df['G']
new_df['Assists'] = df['A']
new_df['Points'] = df['PTS']
new_df['Plusminus'] = df['+/-']
new_df['PIM'] = df['PIM']
new_df['PPG'] = df['PP']
new_df['PPP'] = df['PP']  # Assuming PP points are same as PP goals for now
new_df['SHG'] = df['SH']
new_df['SHP'] = df['SH']  # Assuming SH points are same as SH goals for now
new_df['GWG'] = df['GW']
new_df['OTG'] = 0  # This information is not in the original data
new_df['Shots'] = df['S']
new_df['TOI'] = df['TOI']
new_df['Shifts'] = df['SHFT']
new_df['Year'] = pd.to_datetime(df['Date']).dt.year
new_df['season'] = df['Date'].apply(get_season)
new_df['Home'] = (df['Tm'] == df['Opp']).astype(int)
new_df['toi2'] = df['TOI']
new_df['toi3'] = df['TOI']
new_df['toi_seconds'] = pd.to_timedelta(df['TOI']).dt.total_seconds().astype(int)
new_df['opponent'] = df['Opp'].str.replace('@', '')
new_df['opp_2'] = new_df['opponent']
new_df['season2'] = (pd.to_datetime(df['Date']) - birth_date).astype('<m8[Y]').astype(int) + 1

# Reorder columns to match McDavid's data
column_order = ['Date', 'Team', 'Goals', 'Assists', 'Points', 'Plusminus', 'PIM', 'PPG', 'PPP', 'SHG', 'SHP', 'GWG', 'OTG', 'Shots', 'TOI', 'Shifts', 'Year', 'season', 'Home', 'toi2', 'toi3', 'toi_seconds', 'opponent', 'opp_2', 'season2']
new_df = new_df[column_order]

# Save to CSV
new_df.to_csv('Jack Hughes_df.csv', index=False)

print("Data has been reformatted and saved to 'Jack Hughes_df.csv'")
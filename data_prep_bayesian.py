import pandas as pd
import datetime
from datetime import datetime, timedelta
import os

def process_player_data(player_name, team_name):
    INPUT_DIR = os.path.join('inputs')
    schedule_file = f'{team_name}_schedule_2425_formatted.csv'

    # Read the Jack Hughes data
    df = pd.read_csv(os.path.join(INPUT_DIR, f'{player_name}_df.txt'), delimiter=',')
    schedule = pd.read_csv(os.path.join(INPUT_DIR, schedule_file))

    # Replace 'PHX' with 'ARI' and 'ATL' with 'WPG' in the 'Opp' column
    df['Opp'] = df['Opp'].replace({'PHX': 'ARI', 'ATL': 'WPG'})



    # Function to convert age to birth date
    def age_to_birthdate(age_str, game_date):
        years, days = map(int, age_str.split('-'))
        game_date = datetime.strptime(game_date, '%Y-%m-%d')
        birth_date = game_date - timedelta(days=days) - timedelta(days=years*365)
        return birth_date

    def load_team_mapping(mapping_file):
        team_mapping = pd.read_csv(mapping_file)
        full_name_to_id = dict(zip(team_mapping['team_name'], team_mapping['id']))
        abbrev_to_id = dict(zip(team_mapping['abbreviation'], team_mapping['id']))
        full_name_to_abbrev = dict(zip(team_mapping['team_name'], team_mapping['abbreviation']))
        return full_name_to_id, abbrev_to_id, full_name_to_abbrev


    # Calculate season2
    def calculate_season(date):
        game_date = pd.to_datetime(date)
        age = (game_date - birth_date).days / 365.25  # More precise calculation
        return int(age) + 1

    def time_to_seconds(time_str):
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Unexpected time format: {time_str}")

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
    new_df['Team'] = df['Unnamed: 5'].fillna('vs') + ' ' + df['Opp']
    new_df['Goals'] = df['G.1']
    new_df['Assists'] = df['A']
    new_df['Points'] = df['PTS']
    new_df['Plusminus'] = df['+/-']
    new_df['PIM'] = df['PIM']
    new_df['PPG'] = df['PP']
    new_df['PPP'] = df['PP']  # Assuming PP points are same as PP goals for now
    new_df['SHG'] = df['SH']
    new_df['SHP'] = df['SH'] # Assuming SH points are same as SH goals for now
    new_df['GWG'] = df['GW']
    new_df['OTG'] = 0  # This information is not in the original data
    new_df['Shots'] = df['S']
    new_df['TOI'] = df['TOI']
    new_df['Shifts'] = df['SHFT']
    new_df['Year'] = pd.to_datetime(df['Date']).dt.year
    new_df['season'] = df['Date'].apply(get_season)
    new_df['Home'] = new_df['Team'].str.contains('vs').astype(int)
    new_df['toi2'] = df['TOI']
    new_df['toi3'] = df['TOI']
    new_df['toi_seconds'] = df['TOI'].apply(time_to_seconds)
    new_df['opponent'] = df['Opp'].str.replace('@', '')
    new_df['opp_2'] = new_df['opponent']
    #new_df['season2'] = df['Date'].apply(calculate_season)

    # Reorder columns to match McDavid's data
    column_order = ['Date', 'Team', 'Goals', 'Assists', 'Points', 'Plusminus', 'PIM', 'PPG', 'PPP', 'SHG', 'SHP', 'GWG', 'OTG', 'Shots', 'TOI', 'Shifts', 'Year', 'season', 'Home', 'toi2', 'toi3', 'toi_seconds', 'opponent', 'opp_2']
    new_df = new_df[column_order]

    # Check if 2024-2025 season exists
    team_mapping_file = "team_mapping.csv"
    full_name_to_id, abbrev_to_id, full_name_to_abbrev = load_team_mapping(os.path.join(INPUT_DIR, team_mapping_file))
    first_game = schedule.iloc[0]

    if '202425' not in new_df['season'].values:
        # Create a dummy row for 2024-2025 season
        dummy_row = pd.DataFrame({
            'Date': [first_game['Date']],  # Example date, adjust as needed
            'Team': [f"{first_game['Vs']} {full_name_to_abbrev.get(first_game['Opponent'], first_game['Opponent'])}"],
            'Goals': [0],
            'Assists': [0],
            'Points': [0],
            'Plusminus': [0],
            'PIM': [0],
            'PPG': [0],
            'PPP': [0],
            'SHG': [0],
            'SHP': [0],
            'GWG': [0],
            'OTG': [0],
            'Shots': [0],
            'TOI': ['00:00'],
            'Shifts': [0],
            'Year': ['2024'],
            'season': ['202425'],
            'Home': [1 if first_game['Vs'] == 'vs' else 0],
            'toi2': ['00:00'],
            'toi3': ['00:00'],
            'toi_seconds': [0],
            'opponent': [full_name_to_abbrev.get(first_game['Opponent'], first_game['Opponent'])],
            'opp_2': [full_name_to_abbrev.get(first_game['Opponent'], first_game['Opponent'])],
        })

        # Append the dummy row to new_df
        new_df = pd.concat([new_df, dummy_row], ignore_index=True)


    # Save to CSV
    new_df.to_csv(os.path.join(INPUT_DIR, f'{player_name}_df.csv'), index=False)

    print(f"Data has been reformatted and saved to '{player_name}_df.csv'")
    return new_df
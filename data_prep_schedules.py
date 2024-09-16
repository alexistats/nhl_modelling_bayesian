import pandas as pd
from datetime import datetime
import os

def format_team_schedule(team_name):
    INPUT_DIR = os.path.join('inputs')
    # Read the Devils schedule
    devils_schedule = pd.read_csv(os.path.join(INPUT_DIR, f"{team_name}_schedule_2425.txt"))
    team_mapping = pd.read_csv(os.path.join(INPUT_DIR, "team_mapping.csv"))

    # Create a dictionary for team name mapping
    team_name_mapping = dict(zip(team_mapping['team_name_2'], team_mapping['team_name']))

    # Create a new DataFrame with the desired structure
    formatted_schedule = pd.DataFrame(columns=["", "Date", "Vs", "Opponent"])

    # Fill the new DataFrame
    for index, row in devils_schedule.iterrows():
        date = datetime.strptime(row['Date'], '%Y-%m-%d')
        formatted_date = date.strftime('%a, %b %d')

        vs = "@" if pd.notna(row.iloc[3]) and row.iloc[3] == '@' else "vs"

        # Replace 'Utah Hockey Club' with 'Arizona Coyotes'
        opponent = 'Arizona Coyotes' if row['Opponent'] == 'Utah Hockey Club' else row['Opponent']
        # Map the opponent name to the team_name from the mapping table
        opponent = team_name_mapping.get(opponent, opponent)

        new_row = pd.DataFrame({
            "": [str(index + 1)],
            "Date": [formatted_date],
            "Vs": [vs],
            "Opponent": opponent
        })

        formatted_schedule = pd.concat([formatted_schedule, new_row], ignore_index=True)

    # Save the formatted schedule
    formatted_schedule.to_csv(os.path.join(INPUT_DIR, f"{team_name}_schedule_2425_formatted.csv"), index=False)

    print(formatted_schedule.head())
    return formatted_schedule
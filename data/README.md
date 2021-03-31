## Data Repository

Data cleaning
1. Get the raw match data csv from https://stathead.com/basketball in batches of 100 data point. Save them in `./match`
2. **merge** the raw match csv data points into `match_data.csv`
3. Follow the teams from `match_data.csv`, get the respective player data and merge the player data into match data, save into `match_player_data.csv`
4. add additional player related feature to `match_player_data.csv` and save back into `match_player_data.csv`
5. From data in `match_player_data.csv`, get the past performance average, save into `math_player_data_past_ave.csv`
6. Remove duplicate matches (team and opponent) and save into `final_match_player_data.csv`
6. Feature cleaning the `final_match_player_data.csv` and save the final cleaned, and save into `match_features.csv`
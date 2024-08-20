# from flask import Flask, request, jsonify, render_template, url_for
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# import joblib

# app = Flask(__name__)

# # Load and preprocess data
# daily_match_data = pd.read_csv('Daily_matchdata.csv')
# filtered_fantasy_data = pd.read_csv('filtered_fantasy_data_with_players.csv')

# # Merge and clean data
# merged_data = pd.merge(daily_match_data, filtered_fantasy_data, on=['Player', 'Team'], how='left')
# merged_data.rename(columns={'Role_y': 'Role'}, inplace=True)
# merged_data.drop(columns=['Role_x'], inplace=True, errors='ignore')

# # Handle missing values
# imputer = SimpleImputer(strategy='mean')
# merged_data['Average_Fantasy_Points'] = imputer.fit_transform(merged_data[['Average_Fantasy_Points']].values.reshape(-1, 1))
# merged_data.dropna(subset=['Average_Fantasy_Points'], inplace=True)

# # Normalize roles and remove duplicates
# merged_data['Role'] = merged_data['Role'].replace({'ALL': 'AR', 'BALL': 'BOWL'})
# merged_data.drop_duplicates(subset=['Player', 'Team', 'Role'], inplace=True)

# # Train the model
# def train_model(data):
#     X = data[['Average_Fantasy_Points']]
#     y = data['Average_Fantasy_Points']

#     imputer = SimpleImputer(strategy='mean')
#     X_imputed = imputer.fit_transform(X.values.reshape(-1, 1))

#     X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

#     model = RandomForestRegressor(n_estimators=10, random_state=42)
#     model.fit(X_train, y_train)
#     return model

# model = train_model(merged_data)
# joblib.dump(model, 'random_forest_model.pkl')
# model = joblib.load('random_forest_model.pkl')

# def predict_team_performance(model, team):
#     if team:
#         return sum(model.predict(pd.DataFrame(team['Average_Fantasy_Points']).values.reshape(-1, 1)))
#     return 0

# def select_fantasy_team(merged_data, team1, team2, configurations):
#     selected_teams_data = merged_data[(merged_data['Team'] == team1) | (merged_data['Team'] == team2)]
#     selected_teams_data_sorted = selected_teams_data.sort_values(by='Average_Fantasy_Points', ascending=False)

#     successful_teams = []

#     for config in configurations:
#         team = {'WK': [], 'BAT': [], 'AR': [], 'BOWL': [], 'Total': [], 'Average_Fantasy_Points': []}
#         selected_players = set()

#         valid = True
#         for role, count in zip(['WK', 'BAT', 'AR', 'BOWL'], config):
#             available_players = selected_teams_data_sorted[
#                 (selected_teams_data_sorted['Role'] == role) &
#                 (~selected_teams_data_sorted['Player'].isin(selected_players))
#             ][:count]

#             if len(available_players) < count:
#                 valid = False
#                 break

#             team[role] = available_players['Player'].tolist()
#             team['Average_Fantasy_Points'].extend(available_players['Average_Fantasy_Points'].tolist())
#             selected_players.update(available_players['Player'])

#         if valid:
#             successful_teams.append({'configuration': config, 'team': team})

#     return successful_teams

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/results')
# def results():
#     return render_template('results.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     team1 = request.form.get('team1')
#     team2 = request.form.get('team2')

#     configurations = [
#         [1, 3, 2, 5], [1, 3, 3, 4], [1, 4, 1, 5],
#         [1, 4, 2, 4], [1, 4, 3, 3], [1, 5, 1, 4], [1, 5, 2, 3]
#     ]

#     teams = select_fantasy_team(merged_data, team1, team2, configurations)

#     best_score = 0
#     best_team = None
#     best_configuration = None

#     for team_info in teams:
#         score = predict_team_performance(model, team_info['team'])
#         if score > best_score:
#             best_score = score
#             best_team = team_info['team']
#             best_configuration = team_info['configuration']

#     return jsonify({
#         'best_score': best_score,
#         'best_configuration': best_configuration,
#         'best_team': best_team
#     })

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# import joblib

# app = Flask(__name__)

# # Load and preprocess data
# daily_match_data = pd.read_csv('Daily_matchdata.csv')
# filtered_fantasy_data = pd.read_csv('filtered_fantasy_data_with_players.csv')

# # Merge and clean data
# merged_data = pd.merge(daily_match_data, filtered_fantasy_data, on=['Player', 'Team'], how='left')
# merged_data.rename(columns={'Role_y': 'Role'}, inplace=True)
# merged_data.drop(columns=['Role_x'], inplace=True, errors='ignore')

# # Handle missing values
# imputer = SimpleImputer(strategy='mean')
# merged_data['Average_Fantasy_Points'] = imputer.fit_transform(merged_data[['Average_Fantasy_Points']].values.reshape(-1, 1))
# merged_data.dropna(subset=['Average_Fantasy_Points'], inplace=True)

# # Normalize roles and remove duplicates
# merged_data['Role'] = merged_data['Role'].replace({'ALL': 'AR', 'BALL': 'BOWL'})
# merged_data.drop_duplicates(subset=['Player', 'Team', 'Role'], inplace=True)

# # Train the model
# def train_model(data):
#     X = data[['Average_Fantasy_Points']]
#     y = data['Average_Fantasy_Points']

#     imputer = SimpleImputer(strategy='mean')
#     X_imputed = imputer.fit_transform(X.values.reshape(-1, 1))

#     X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

#     model = RandomForestRegressor(n_estimators=10, random_state=42)
#     model.fit(X_train, y_train)
#     return model

# model = train_model(merged_data)
# joblib.dump(model, 'random_forest_model.pkl')
# model = joblib.load('random_forest_model.pkl')

# def predict_team_performance(model, team):
#     if team:
#         return sum(model.predict(pd.DataFrame(team['Average_Fantasy_Points']).values.reshape(-1, 1)))
#     return 0

# def select_fantasy_team(merged_data, team1, team2, configurations):
#     selected_teams_data = merged_data[(merged_data['Team'] == team1) | (merged_data['Team'] == team2)]
#     selected_teams_data_sorted = selected_teams_data.sort_values(by='Average_Fantasy_Points', ascending=False)

#     successful_teams = []

#     for config in configurations:
#         team = {'WK': [], 'BAT': [], 'AR': [], 'BOWL': [], 'Total': [], 'Average_Fantasy_Points': []}
#         selected_players = set()

#         valid = True
#         for role, count in zip(['WK', 'BAT', 'AR', 'BOWL'], config):
#             available_players = selected_teams_data_sorted[
#                 (selected_teams_data_sorted['Role'] == role) &
#                 (~selected_teams_data_sorted['Player'].isin(selected_players))
#             ][:count]

#             if len(available_players) < count:
#                 valid = False
#                 break

#             team[role] = available_players['Player'].tolist()
#             team['Average_Fantasy_Points'].extend(available_players['Average_Fantasy_Points'].tolist())
#             selected_players.update(available_players['Player'])

#         if valid:
#             successful_teams.append({'configuration': config, 'team': team})

#     return successful_teams

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     team1 = request.form.get('team1')
#     team2 = request.form.get('team2')

#     configurations = [
#         [1, 3, 2, 5], [1, 3, 3, 4], [1, 4, 1, 5],
#         [1, 4, 2, 4], [1, 4, 3, 3], [1, 5, 1, 4], [1, 5, 2, 3]
#     ]

#     teams = select_fantasy_team(merged_data, team1, team2, configurations)

#     best_score = 0
#     best_team = None
#     best_configuration = None

#     for team_info in teams:
#         score = predict_team_performance(model, team_info['team'])
#         if score > best_score:
#             best_score = score
#             best_team = team_info['team']
#             best_configuration = team_info['configuration']

#     # Determine captain and vice-captain
#     if best_team:
#         players_points = list(zip(best_team['Average_Fantasy_Points'], sum(best_team.values(), [])))
#         players_points.sort(reverse=True)
#         captain = players_points[0][1]
#         vice_captain = players_points[1][1]
#     else:
#         captain = None
#         vice_captain = None

#     return jsonify({
#         'best_score': best_score,
#         'best_configuration': best_configuration,
#         'best_team': best_team,
#         'captain': captain,
#         'vice_captain': vice_captain
#     })

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import random

app = Flask(__name__)

# Load and preprocess data
daily_match_data = pd.read_csv('Daily_matchdata.csv')
filtered_fantasy_data = pd.read_csv('filtered_fantasy_data_with_players.csv')

# Merge and clean data
merged_data = pd.merge(daily_match_data, filtered_fantasy_data, on=['Player', 'Team'], how='left')
merged_data.rename(columns={'Role_y': 'Role'}, inplace=True)
merged_data.drop(columns=['Role_x'], inplace=True, errors='ignore')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
merged_data['Average_Fantasy_Points'] = imputer.fit_transform(merged_data[['Average_Fantasy_Points']].values.reshape(-1, 1))
merged_data.dropna(subset=['Average_Fantasy_Points'], inplace=True)

# Normalize roles and remove duplicates
merged_data['Role'] = merged_data['Role'].replace({'ALL': 'AR', 'BALL': 'BOWL'})
merged_data.drop_duplicates(subset=['Player', 'Team', 'Role'], inplace=True)

# Train the model
def train_model(data):
    X = data[['Average_Fantasy_Points']]
    y = data['Average_Fantasy_Points']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(merged_data)
joblib.dump(model, 'random_forest_model.pkl')
model = joblib.load('random_forest_model.pkl')

def predict_team_performance(model, team):
    if team:
        return sum(model.predict(pd.DataFrame(team['Average_Fantasy_Points']).values.reshape(-1, 1)))
    return 0

def select_fantasy_team(merged_data, team1, team2, configurations):
    selected_teams_data = merged_data[(merged_data['Team'] == team1) | (merged_data['Team'] == team2)]
    selected_teams_data_sorted = selected_teams_data.sort_values(by='Average_Fantasy_Points', ascending=False)

    successful_teams = []

    for config in configurations:
        team = {'WK': [], 'BAT': [], 'AR': [], 'BOWL': [], 'Total': [], 'Average_Fantasy_Points': []}
        selected_players = set()

        valid = True
        for role, count in zip(['WK', 'BAT', 'AR', 'BOWL'], config):
            available_players = selected_teams_data_sorted[
                (selected_teams_data_sorted['Role'] == role) &
                (~selected_teams_data_sorted['Player'].isin(selected_players))
            ][:count]

            if len(available_players) < count:
                valid = False
                break

            team[role] = available_players['Player'].tolist()
            team['Average_Fantasy_Points'].extend(available_players['Average_Fantasy_Points'].tolist())
            selected_players.update(available_players['Player'])

        if valid:
            successful_teams.append({'configuration': config, 'team': team})

    return successful_teams

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')

    configurations = [
        [1, 3, 2, 5], [1, 3, 3, 4], [1, 4, 1, 5],
        [1, 4, 2, 4], [1, 4, 3, 3], [1, 5, 1, 4], [1, 5, 2, 3]
    ]

    teams = select_fantasy_team(merged_data, team1, team2, configurations)

    best_score = 0
    best_team = None
    best_configuration = None

    for team_info in teams:
        score = predict_team_performance(model, team_info['team'])
        if score > best_score:
            best_score = score
            best_team = team_info['team']
            best_configuration = team_info['configuration']

    # Determine captain and vice-captain
    if best_team:
        players_points = list(zip(best_team['Average_Fantasy_Points'], sum(best_team.values(), [])))
        players_points.sort(reverse=True)
        captain_candidates = [player for points, player in players_points if points == players_points[0][0]]
        captain = random.choice(captain_candidates)
        vice_captain_candidates = [player for points, player in players_points if points == players_points[len(captain_candidates)][0]]
        vice_captain = random.choice(vice_captain_candidates)
    else:
        captain = None
        vice_captain = None

    return jsonify({
        'best_score': best_score,
        'best_configuration': best_configuration,
        'best_team': best_team,
        'captain': captain,
        'vice_captain': vice_captain
    })

if __name__ == '__main__':
    app.run(debug=True)

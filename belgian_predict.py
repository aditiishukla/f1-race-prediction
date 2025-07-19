import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#Collecting data
belgian_gp_rounds = {
    2021: 12,
    2022: 14,
    2023: 13,
    2024: 13
}
all_data = []

for year, rd in belgian_gp_rounds.items():
    try:
        print(f"\nLoading Belgian GP {year} (Round {rd})...")
        session = fastf1.get_session(year, rd, "R")
        session.load()

        #Getting lap data
        laps = session.laps[["Driver", "Team", "LapTime"]].copy()
        laps.dropna(subset=["LapTime"], inplace=True)
        laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

        #Average lap time per driver
        avg_lap_times = laps.groupby(["Driver", "Team"])["LapTime (s)"].mean().reset_index()

        #Finishing position from previous official race results
        results = session.results[["Abbreviation", "Position"]]
        avg_lap_times = avg_lap_times.merge(results, left_on="Driver", right_on="Abbreviation")
        avg_lap_times["Year"] = year

        final_data = avg_lap_times[["Year", "Driver", "Team", "LapTime (s)", "Position"]]
        all_data.append(final_data)

    except Exception as e:
        print(f"Failed to fetch data for {year}: {e}")

#Combining all data in one dataset
combined_df = pd.concat(all_data, ignore_index=True)

#Preparing data for model training
combined_df["DriverCode"] = combined_df["Driver"].astype("category").cat.codes
combined_df["TeamCode"] = combined_df["Team"].astype("category").cat.codes

X = combined_df[["LapTime (s)", "DriverCode", "TeamCode"]]
y = combined_df["Position"]

#Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

#Getting latest Belgian GP (2024) avg lap times
latest_gp = combined_df[combined_df["Year"] == 2024].copy()

latest_gp["DriverCode"] = latest_gp["Driver"].astype("category").cat.codes
latest_gp["TeamCode"] = latest_gp["Team"].astype("category").cat.codes

X_future = latest_gp[["LapTime (s)", "DriverCode", "TeamCode"]]

#Predicting the positions
latest_gp["PredictedPosition"] = model.predict(X_future)
latest_gp = latest_gp.sort_values(by="PredictedPosition")

#Displaying the result
print("\n🏁 Predicted 2025 Belgian GP Finishing Order 🏁\n")
print(latest_gp[["Driver", "Team", "PredictedPosition"]])


#Evaluating our model performance 
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel MAE on validation data: {mae:.2f} position points")

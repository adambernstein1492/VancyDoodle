import pandas as pd
import datetime

def parse_patient_file(file_content):
    data = {
        "birthdate": None, "height": None, "weight": None, "creatinine": None,
        "model": None, "doses": [],
        "concentrations": []  # Matches the engine list variable
    }

    current_section = "demographics"

    for line in file_content.decode("utf-8").splitlines():
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        if line == "[Doses]":
            current_section = "doses"
            continue
        elif line == "[Levels]":
            current_section = "levels"
            continue

        if current_section == "demographics":
            key, value = line.split(":")
            key = key.strip().lower()

            # Keep model and birthdate as strings; convert the rest to floats
            if key in ["model", "birthdate"]:
                data[key] = value.strip()
            else:
                data[key] = float(value.strip())

        elif current_section == "doses":
            time_str, amt, inf_time = line.split(",")
            # Requires datetime.datetime.strptime since 'import datetime' is used
            dt = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M")
            data["doses"].append({
                "datetime": dt,
                "dose": float(amt),
                "infusion_time": float(inf_time)
            })

        elif current_section == "levels":
            time_str, val = line.split(",")
            # Requires datetime.datetime.strptime since 'import datetime' is used
            dt = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M")
            data["concentrations"].append({
                "datetime": dt,
                "conc": float(val)
            })

    return data

def make_train_of_doses(priordoses, newdose, interval, infusion_time, first_dose_datetime, num_doses=10):
    for i in range(int(num_doses)):
        updated_datetime = first_dose_datetime + datetime.timedelta(hours=interval * i)
        dose = pd.DataFrame({'Dose': [newdose], 'InfusionTime': [infusion_time], 'DateTime': [updated_datetime]})

        if priordoses.empty:
            priordoses = dose.copy()
        else:
            priordoses = pd.concat([priordoses, dose], ignore_index=True)

    return priordoses

def format_input_data(levels_df, doses_df):
    doses = doses_df.copy()
    levels = levels_df.copy()

    if not doses.empty:
        doses['DateTime'] = pd.to_datetime(doses['DateTime'])
        doses['Event'] = 'Dose'

    if not levels.empty:
        levels['DateTime'] = pd.to_datetime(levels['DateTime'])
        levels['Event'] = 'Level'

    # Filter out empty dataframes before concatenating
    dfs_to_concat = [df for df in [doses, levels] if not df.empty]
    clinical_data = pd.concat(dfs_to_concat, ignore_index=True) if dfs_to_concat else pd.DataFrame()

    if clinical_data.empty:
        return clinical_data

    if not doses.empty:
        first_dose_time = doses['DateTime'].min()
    else:
        first_dose_time = clinical_data['DateTime'].min()

    time_diff = clinical_data['DateTime'] - first_dose_time
    clinical_data['Time_hr'] = time_diff.dt.total_seconds() / 3600.0

    # 4. Sort & Clean
    clinical_data = clinical_data.sort_values(by='Time_hr').reset_index(drop=True)
    clinical_data = clinical_data.drop_duplicates(keep='first').reset_index(drop=True)

    # 5. Column ordering
    cols = ['Time_hr', 'Event', 'Dose', 'InfusionTime', 'Level', 'DateTime']
    cols = [c for c in cols if c in clinical_data.columns]

    return clinical_data[cols]


def generate_patient_file_content(birthdate, weight, height, creatinine, model, doses_df, levels_df):
    lines = []

    # Demographics
    lines.append(f"Birthdate: {birthdate.strftime('%Y-%m-%d')}")
    lines.append(f"Height: {height}")
    lines.append(f"Weight: {weight}")
    lines.append(f"Creatinine: {creatinine}")
    lines.append(f"Model: {model}")
    lines.append("")

    # Doses
    lines.append("[Doses]")
    lines.append("# Format: YYYY-MM-DD HH:MM, dose_in_mg, infusion_time_in_hours")
    if not doses_df.empty:
        for _, row in doses_df.iterrows():
            # Ensure it is a pandas timestamp/datetime before formatting
            dt = pd.to_datetime(row['DateTime'])
            lines.append(f"{dt.strftime('%Y-%m-%d %H:%M')}, {row['Dose']}, {row['InfusionTime']}")
    lines.append("")

    # Levels
    lines.append("[Levels]")
    lines.append("# Format: YYYY-MM-DD HH:MM, level_in_mg_L")
    if not levels_df.empty:
        for _, row in levels_df.iterrows():
            dt = pd.to_datetime(row['DateTime'])
            lines.append(f"{dt.strftime('%Y-%m-%d %H:%M')}, {row['Level']}")

    return "\n".join(lines)
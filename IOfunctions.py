import pandas as pd
import datetime

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
import argparse
import pandas as pd
from engine import VancomycinBayesEngine
from IOfunctions import parse_patient_file, format_input_data
from Visualization import get_concentration_plot
import datetime


def main():
    parser = argparse.ArgumentParser(description="Run Van-ke Doodle via CLI")
    parser.add_argument("file", help="Path to the patient data .txt file")
    args = parser.parse_args()

    # 1. Read and parse the local file
    with open(args.file, "rb") as f:
        patient_data = parse_patient_file(f.read())

    print(f"Loaded profile for Model: {patient_data.get('model', 'Unknown')}")

    # 2. Convert parsed dictionaries to pandas DataFrames
    # Rename keys to match the capitalized columns required by format_input_data
    doses_df = pd.DataFrame(patient_data["doses"]).rename(columns={
        "datetime": "DateTime",
        "dose": "Dose",
        "infusion_time": "InfusionTime"
    })

    levels_df = pd.DataFrame(patient_data["concentrations"]).rename(columns={
        "datetime": "DateTime",
        "conc": "Level"
    })

    # Calculate age in days
    birthdate = datetime.datetime.strptime(patient_data["birthdate"], "%Y-%m-%d").date()
    today = datetime.date.today()
    age_total_days = (today - birthdate).days

    # 3. Initialize the Bayesian Engine
    engine = VancomycinBayesEngine(
        weight_kg=patient_data["weight"],
        height_cm=patient_data["height"],
        age_total_days=age_total_days,
        creatinine=patient_data["creatinine"],
        model=patient_data["model"]
    )

    # 4. Format time-relative clinical data and run the MAP estimation
    clinical_data = format_input_data(levels_df, doses_df)
    engine.fit_patient(clinical_data)

    # 5. Calculate Confidence Intervals
    prior_low, prior_high = engine.calculate_ci_boundaries(
        engine.population_mean,
        clinical_data
    )

    fit_low, fit_high = engine.calculate_ci_boundaries(
        engine.map_params,
        clinical_data
    )

    # 6. Generate the Plotly figure
    fig = get_concentration_plot(
        engine=engine,
        clinical_data=clinical_data,
        show_prior=True,
        show_fit=True,
        show_labs=True,
        show_ci=True,
        prior_ci_bounds=(prior_low, prior_high),
        fit_ci_bounds=(fit_low, fit_high)
    )

    # 7. Open the interactive chart natively in the default web browser
    output_filename = "pk_plot.html"
    fig.write_html(output_filename)
    print(f"Plot successfully saved to {output_filename}")

    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath(output_filename))


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
from fpdf import FPDF
from Visualization import get_concentration_plot


class PKReportPDF(FPDF):
    def header(self):
        # Add the Van-ke Doodle logo if it exists in the directory
        if os.path.exists('logo1.png'):
            self.image('logo1.png', 10, 8, 30)

        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Van-ke Doodle: Pharmacokinetic Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, 'Not intended for clinical use.', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def round_to_25(val):
    """Rounds a dose to the nearest 25 mg."""
    return max(25.0, np.round(val / 25.0) * 25.0)


def calculate_ideal_regimens(CL, target_auc=500.0):
    """Calculates q6 and q8 doses targeting a specific AUC based on a given Clearance."""
    daily_dose = target_auc * CL

    q6_dose = round_to_25(daily_dose / 4.0)
    q8_dose = round_to_25(daily_dose / 3.0)

    q6_auc = (q6_dose * 4.0) / CL
    q8_auc = (q8_dose * 3.0) / CL

    return q6_dose, q6_auc, q8_dose, q8_auc


def generate_pdf_report(engine, clinical_data, filename="VanKe_Doodle_Report.pdf"):
    pdf = PKReportPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 11)

    # --- PATIENT DEMOGRAPHICS ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Patient Profile & Model', ln=1, border='B')
    pdf.set_font('Arial', '', 11)

    model_name = engine.model_name if hasattr(engine, 'model_name') else "Selected Population Model"
    pdf.cell(0, 6,
             f"Weight: {engine.weight} kg   |   Height: {engine.height} cm   |   Creatinine: {engine.creatinine} mg/dL",
             ln=1)
    pdf.cell(0, 6, f"PK Model: {model_name}", ln=1)
    pdf.ln(5)

    # --- CURRENT REGIMEN EVALUATION ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Current Regimen Evaluation (Post-Fit)', ln=1, border='B')
    pdf.set_font('Arial', '', 11)

    doses = clinical_data[clinical_data['Event'] == 'Dose']
    if len(doses) >= 2:
        last_dose = doses.iloc[-1]['Dose']
        t_inf = doses.iloc[-1]['InfusionTime']
        interval = round(doses.iloc[-1]['Time_hr'] - doses.iloc[-2]['Time_hr'])

        # Calculate AUC and probabilities using the engine
        auc_ss = engine.get_steady_state_auc(clinical_data)
        _, _, metrics = engine.evaluate_regimen(last_dose, interval, t_inf)

        pdf.cell(0, 6, f"Current Regimen: {last_dose:.0f} mg q{interval:.0f}hr", ln=1)
        pdf.cell(0, 6, f"Estimated Steady-State AUC: {auc_ss:.1f} mg*h/L", ln=1)
        pdf.cell(0, 6, f"Probability of Subtherapeutic AUC (<400): {metrics['prob_subtherapeutic']:.1f}%", ln=1)
        pdf.cell(0, 6, f"Probability of Supratherapeutic AUC (>600): {metrics['prob_supratherapeutic']:.1f}%", ln=1)
    else:
        pdf.cell(0, 6, "Insufficient dosing history to evaluate steady-state interval.", ln=1)
    pdf.ln(5)

    # --- IDEAL DOSING: PRIOR VS POSTERIOR ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Target Dosing Recommendations (Target AUC = 500)', ln=1, border='B')
    pdf.set_font('Arial', '', 10)

    # Prior Calculations
    prior_CL = engine.population_mean[2]
    prior_q6_d, prior_q6_a, prior_q8_d, prior_q8_a = calculate_ideal_regimens(prior_CL)

    # Posterior (MAP) Calculations
    map_CL = engine.map_params[2] if engine.calibrated else prior_CL
    map_q6_d, map_q6_a, map_q8_d, map_q8_a = calculate_ideal_regimens(map_CL)

    # Table Header
    col_w = [40, 45, 45, 60]
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(col_w[0], 8, "Basis", border=1, fill=True, align='C')
    pdf.cell(col_w[1], 8, "Interval", border=1, fill=True, align='C')
    pdf.cell(col_w[2], 8, "Recommended Dose", border=1, fill=True, align='C')
    pdf.cell(col_w[3], 8, "Predicted AUC", border=1, fill=True, align='C')
    pdf.ln()

    def add_row(basis, interval, dose, auc):
        pdf.cell(col_w[0], 8, basis, border=1, align='C')
        pdf.cell(col_w[1], 8, interval, border=1, align='C')
        pdf.cell(col_w[2], 8, f"{dose:.0f} mg", border=1, align='C')
        pdf.cell(col_w[3], 8, f"{auc:.1f}", border=1, align='C')
        pdf.ln()

    add_row("Prior (Population)", "q6hr", prior_q6_d, prior_q6_a)
    add_row("Prior (Population)", "q8hr", prior_q8_d, prior_q8_a)
    add_row("Posterior (MAP Fit)", "q6hr", map_q6_d, map_q6_a)
    add_row("Posterior (MAP Fit)", "q8hr", map_q8_d, map_q8_a)
    pdf.ln(10)

    # --- PLOTLY FIGURE INJECTION ---
    # Generate the plot and save temporarily via Kaleido
    try:
        prior_low, prior_high = engine.calculate_ci_boundaries(engine.population_mean, clinical_data)
        fit_low, fit_high = engine.calculate_ci_boundaries(engine.map_params, clinical_data)

        fig = get_concentration_plot(
            engine=engine, clinical_data=clinical_data,
            show_prior=True, show_fit=True, show_labs=True, show_ci=True,
            prior_ci_bounds=(prior_low, prior_high), fit_ci_bounds=(fit_low, fit_high)
        )

        # Adjust layout for static PDF rendering (white background)
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
        fig.update_xaxes(gridcolor='lightgray', zerolinecolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray', zerolinecolor='lightgray')

        temp_img_path = "temp_pk_plot.png"
        fig.write_image(temp_img_path, width=800, height=400, scale=2)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Concentration-Time Profile', ln=1, border='B')
        pdf.image(temp_img_path, x=10, w=190)

        os.remove(temp_img_path)
    except Exception as e:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 6, f"(Could not render figure in PDF. Ensure 'kaleido' is installed. Error: {e})", ln=1)

    pdf.output(filename)
    return filename
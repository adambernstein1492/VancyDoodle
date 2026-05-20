import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_pk_table(engine):
    """
    Constructs a DataFrame comparing Population Priors to MAP estimates,
    ensuring the column structure remains intact even before a fit.
    """
    param_names = ['Central Volume (Vc)', 'Peripheral Volume (Vp)', 'Clearance (CL)', 'Intercompartmental Q']

    table_data = {
        'Parameter': param_names,
        'Prior (Pop)': np.round(engine.population_mean, 2)
    }

    # Insert MAP estimates if calibrated, otherwise populate with consistent placeholders
    if engine.calibrated and engine.map_params is not None:
        table_data['Updated (MAP)'] = np.round(engine.map_params, 2)
    else:
        table_data['Updated (MAP)'] = ["—", "—", "—", "—"]

    return pd.DataFrame(table_data)

def get_cv_table(engine):
    """
    Constructs a DataFrame comparing Population Prior CV% to
    Individual Post-Fit CV% to display narrowing uncertainty.
    """
    param_names = ['Central Volume (Vc)', 'Peripheral Volume (Vp)', 'Clearance (CL)', 'Intercompartmental Q']
    prior_cvs, fit_cvs = engine.get_coefficients_of_variation()

    # Format values as clean strings with '%' or show '-' if no IIV exists
    prior_strings = [f"{round(val, 1)}%" if mask else "—" for val, mask in zip(prior_cvs, engine.has_iiv)]

    table_data = {
        'Parameter': param_names,
        'Prior CV': prior_strings
    }

    if engine.calibrated and fit_cvs is not None:
        fit_strings = [f"{round(val, 1)}%" if mask else "—" for val, mask in zip(fit_cvs, engine.has_iiv)]
        table_data['Post-Fit CV'] = fit_strings
    else:
        table_data['Post-Fit CV'] = ["—" if mask else "—" for mask in engine.has_iiv]

    return pd.DataFrame(table_data)

def get_concentration_plot(engine, clinical_data, show_prior=True, show_fit=True, show_labs=True, show_ci=False, prior_ci_bounds=(None, None), fit_ci_bounds=(None, None)):
    """
    Builds the Plotly figure using fast pre-computed CI boundaries.
    """
    fig = go.Figure()

    # Goal Trough and Peak Layout elements
    fig.add_hrect(y0=engine.trough_min, y1=engine.trough_max, fillcolor="gray", opacity=0.15, layer="below", line_width=0)
    fig.add_hline(y=engine.peak, line_dash="dash", line_color="red", layer="below")

    # Fast point simulations for the main lines
    times, prior_conc = engine.simulate_profile(engine.population_mean, clinical_data)
    if engine.calibrated:
        _, fit_conc = engine.simulate_profile(engine.map_params, clinical_data)

    # Unpack pre-computed CI tuples
    prior_low, prior_high = prior_ci_bounds
    fit_low, fit_high = fit_ci_bounds

    # 3. Prior PK & Ribbon
    if show_prior:
        if show_ci and prior_low is not None and prior_high is not None:
            fig.add_trace(go.Scatter(x=times, y=prior_high, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=times, y=prior_low, mode='lines', line=dict(width=0), fill='tonexty', fillcolor="rgba(149, 165, 166, 0.20)", name='95% Prior CI', legendgroup="prior_ci", hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=times, y=prior_conc, mode='lines', line=dict(color='#95a5a6', width=2), name='Population Prior', legendgroup="prior_ci"))

    # 4. Fit PK & Ribbon
    if show_fit and engine.calibrated:
        if show_ci and fit_low is not None and fit_high is not None:
            fig.add_trace(go.Scatter(x=times, y=fit_high, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=times, y=fit_low, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.20)', name='95% Fit MAP CI', legendgroup="fit_ci", hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=times, y=fit_conc, mode='lines', line=dict(color='#2ecc71', width=2.5), name='Fitted MAP', legendgroup="fit_ci"))

    # 5. Lab Markers
    if show_labs and not clinical_data.empty:
        labs = clinical_data[clinical_data['Event'] == 'Level']
        fig.add_trace(go.Scatter(x=labs['Time_hr'], y=labs['Level'], mode='markers', marker=dict(color='red', size=10), name='Measured Levels'))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time (Hours)", yaxis_title="Vancomycin Concentration (mg/L)",
        hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def get_auc_histogram(auc_samples, target_min=400, target_max=600):
    """
    Builds a clean, theme-aligned histogram with subtle color choices.
    """
    fig = go.Figure()

    # Using a professional, muted slate-blue that matches the sidebar aesthetic
    fig.add_trace(go.Histogram(
        x=auc_samples,
        nbinsx=60,
        marker_color='#3498db',
        marker_line_color='#2c3e50',  # Subtle border for definition
        marker_line_width=0.5,
        opacity=0.75,
        name='Predicted AUC'
    ))

    # Using softer, transparent fills for the clinical zones
    upper_bound = max(1000.0, max(auc_samples) + 100.0)

    # Subtherapeutic Zone (Soft Amber)
    fig.add_vrect(
        x0=0, x1=target_min,
        fillcolor="#f1c40f", opacity=0.1, layer="below", line_width=0,
    )
    # Therapeutic Target Zone (Soft Emerald)
    fig.add_vrect(
        x0=target_min, x1=target_max,
        fillcolor="#2ecc71", opacity=0.1, layer="below", line_width=0,
    )
    # Supratherapeutic Zone (Soft Crimson)
    fig.add_vrect(
        x0=target_max, x1=upper_bound,
        fillcolor="#e74c3c", opacity=0.1, layer="below", line_width=0,
    )

    fig.update_layout(
        xaxis_title="Predicted Steady-State AUC24 (mg·h/L)",
        yaxis_title="Patient Density",
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x",
        xaxis=dict(range=[200, 1000], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )

    return fig


def render_predictive_evaluation(engine):
    """
    Renders the Streamlit UI components and metrics for the Predictive Regimen Evaluation.
    """
    st.write("---")
    st.subheader("Predictive Regimen Evaluation")
    st.markdown("Test a maintenance dosing regimen to predict steady-state performance and toxicity risks.")

    # Input Form
    with st.form(key="regimen_eval_form"):
        rc1, rc2, rc3, rc4 = st.columns(4)
        eval_dose = rc1.number_input("Test Dose (mg)", min_value=0.0, value=750.0, step=50.0)
        eval_int = rc2.number_input("Test Interval (hrs)", min_value=1.0, value=12.0, step=1.0)
        eval_tinf = rc3.number_input("Infusion Time (hrs)", min_value=0.5, value=1.0, step=0.25)

        # Submit Button
        rc4.write("")
        rc4.write("")
        submit_eval = rc4.form_submit_button("Simulate Regimen", width='stretch')

    # Execution Block
    if submit_eval:
        with st.spinner("Calculating deterministic steady-state limits..."):
            auc_samples, peak_samples = engine.evaluate_regimen(eval_dose, eval_int, eval_tinf)

            # Calculate Statistical Percentages
            p_sub = np.mean(auc_samples < engine.target_auc_min) * 100
            p_target = np.mean((auc_samples >= engine.target_auc_min) & (auc_samples <= engine.target_auc_max)) * 100
            p_supra = np.mean(auc_samples > engine.target_auc_max) * 100
            p_peak = np.mean(peak_samples > engine.peak) * 100

            st.write("")

            # Display Top-Level Metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Subtherapeutic Risk", f"{p_sub:.1f}%", "< 400 AUC", delta_color="off")
            mc2.metric("Target Probability", f"{p_target:.1f}%", "400 - 600 AUC", delta_color="normal")
            mc3.metric("Supratherapeutic Risk", f"{p_supra:.1f}%", "> 600 AUC", delta_color="off")
            mc4.metric("Toxic Peak Risk", f"{p_peak:.1f}%", "> 50 mg/L", delta_color="inverse")

            st.write("")

            # Display Histogram (calling the function from within this same file)
            fig_hist = get_auc_histogram(auc_samples, engine.target_auc_min, engine.target_auc_max)
            st.plotly_chart(fig_hist, width='stretch')


def get_regimen_grid(grid_results):
    """
    Converts the engine results dictionary into a 5-column grid.
    """
    df = pd.DataFrame.from_dict(grid_results, orient='index',
                                columns=['Dose -2', 'Dose -1', 'Optimal Dose', 'Dose +1', 'Dose +2'])

    # Stylize the middle column (Optimal) to be highlighted
    def highlight_optimal(col):
        return ['background-color: rgba(46, 204, 113, 0.2)' if col.name == 'Optimal Dose' else '' for _ in col]

    return df.style.apply(highlight_optimal)
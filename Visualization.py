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


def get_concentration_plot(engine, clinical_data, show_prior=True, show_fit=True, show_labs=True, show_ci=False,
                           prior_ci_bounds=(None, None), fit_ci_bounds=(None, None)):
    """
    Builds the Plotly figure using fast pre-computed CI boundaries.
    """
    fig = go.Figure()

    # Goal Trough and Peak Layout elements
    fig.add_hrect(y0=engine.trough_min, y1=engine.trough_max, fillcolor="gray", opacity=0.15, layer="below",
                  line_width=0)
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
            fig.add_trace(
                go.Scatter(x=times, y=prior_high, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=times, y=prior_low, mode='lines', line=dict(width=0), fill='tonexty',
                                     fillcolor="rgba(149, 165, 166, 0.20)", name='95% Prior CI', legendgroup="prior_ci",
                                     hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=times, y=prior_conc, mode='lines', line=dict(color='#95a5a6', width=2),
                                 name='Population Prior', legendgroup="prior_ci"))

    # 4. Fit PK & Ribbon
    if show_fit and engine.calibrated:
        if show_ci and fit_low is not None and fit_high is not None:
            fig.add_trace(
                go.Scatter(x=times, y=fit_high, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=times, y=fit_low, mode='lines', line=dict(width=0), fill='tonexty',
                                     fillcolor='rgba(46, 204, 113, 0.20)', name='95% Fit MAP CI', legendgroup="fit_ci",
                                     hoverinfo='skip'))
        fig.add_trace(
            go.Scatter(x=times, y=fit_conc, mode='lines', line=dict(color='#2ecc71', width=2.5), name='Fitted MAP',
                       legendgroup="fit_ci"))

    # 5. Lab Markers
    if show_labs and not clinical_data.empty:
        labs = clinical_data[clinical_data['Event'] == 'Level']
        fig.add_trace(go.Scatter(x=labs['Time_hr'], y=labs['Level'], mode='markers', marker=dict(color='red', size=10),
                                 name='Measured Levels'))

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

    # Calculate the mean of the simulated AUCs
    mean_auc = np.mean(auc_samples)

    # Define fixed bins. A size of 20 ensures bin edges hit exactly at 400 and 600.
    fig.add_trace(go.Histogram(
        x=auc_samples,
        xbins=dict(
            start=0,
            end=2000,
            size=10
        ),
        marker_color='#95a5a6',  # A soft, mid-tone slate gray
        marker_line_color='#7f8c8d',  # A slightly darker gray for the gentle outline
        marker_line_width=1,
        opacity=0.7,
        name='Predicted AUC'
    ))

    # Using softer, transparent fills for the clinical zones
    upper_bound = max(1000.0, max(auc_samples) + 100.0)

    # Subtherapeutic Zone (Soft Amber)
    fig.add_vrect(
        x0=0, x1=target_min,
        fillcolor="#f1c40f", opacity=0.2, layer="below", line_width=0,
    )
    # Therapeutic Target Zone (Soft Emerald)
    fig.add_vrect(
        x0=target_min, x1=target_max,
        fillcolor="#2ecc71", opacity=0.2, layer="below", line_width=0,
    )
    # Supratherapeutic Zone (Soft Crimson)
    fig.add_vrect(
        x0=target_max, x1=upper_bound,
        fillcolor="#e74c3c", opacity=0.2, layer="below", line_width=0,
    )

    # --- NEW: Vertical Demarcation Lines ---

    # Threshold Lines (400 and 600) - Dotted and subtle
    fig.add_vline(x=target_min, line_dash="dot", line_color="rgba(250, 250, 250, 0.8)", line_width=2)
    fig.add_vline(x=target_max, line_dash="dot", line_color="rgba(250, 250, 250, 0.8)", line_width=2)

    # Mean AUC Line - Dashed, prominent blue, with an annotation
    fig.add_vline(
        x=mean_auc,
        line_dash="dash",
        line_color="#3498db",
        line_width=2.5,
        annotation_text=f"Mean: {mean_auc:.0f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#3498db")
    )

    fig.update_layout(
        title="Predicted Probability Distrubition of the SS AUC24",
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
    Converts the engine results dictionary into a 4-column metrics grid.
    """
    df = pd.DataFrame.from_dict(grid_results, orient='index')

    # We stylize the Optimal Dose and % PTA to stand out, and apply warning colors to the risks
    def highlight_optimal(col):
        if col.name == 'Optimal Dose':
            return ['background-color: rgba(46, 204, 113, 0.2)'] * len(col)
        elif col.name == '% PTA':
            return ['color: #2ecc71; font-weight: bold'] * len(col)
        elif 'Risk' in col.name:
            return ['color: #e74c3c'] * len(col)
        else:
            return [''] * len(col)

    return df.style.apply(highlight_optimal)


def get_1cmt_comparison_plot(engine, clinical_data, pk_estimates):
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    doses = clinical_data[clinical_data['Event'] == 'Dose'].copy()
    levels = clinical_data[clinical_data['Event'] == 'Level'].copy()

    last_dose = doses.iloc[-1]
    dose_amt = last_dose['Dose']
    t_inf = last_dose['InfusionTime']

    if len(doses) > 1:
        intervals = np.diff(doses['Time_hr'])
        interval = np.mean(intervals[intervals > 0]) if any(intervals > 0) else 24.0
    else:
        interval = 24.0

    t_half = pk_estimates['Half-life (hr)']
    ke = np.log(2) / t_half
    V = pk_estimates['Volume of Distribution (L/kg)'] * engine.weight
    Cmin = pk_estimates['Estimated Cmin (mg/L)']
    Cmax = pk_estimates['Estimated Cmax (mg/L)']
    Rate = dose_amt / t_inf

    t = np.linspace(0, interval, 200)
    c_1cmt = np.zeros_like(t)

    for i, time in enumerate(t):
        if time <= t_inf:
            c_1cmt[i] = Cmin * np.exp(-ke * time) + (Rate / (V * ke)) * (1 - np.exp(-ke * time))
        else:
            c_1cmt[i] = Cmax * np.exp(-ke * (time - t_inf))

    # Generate Steady State 2-cmt (using MAP params via _compute_superposition)
    mock_doses = []
    for i in range(12):
        mock_doses.append({
            'Event': 'Dose',
            'Time_hr': i * interval,
            'Dose': dose_amt,
            'InfusionTime': t_inf
        })
    mock_df = pd.DataFrame(mock_doses)

    t_ss = np.linspace(10 * interval, 11 * interval, 200)
    c_2cmt_ss = engine._compute_superposition(engine.map_params, mock_df, t_ss)

    fig = go.Figure()

    # 1-Cmt Area & Line
    fig.add_trace(go.Scatter(
        x=t, y=c_1cmt, fill='tozeroy', fillcolor='rgba(43, 106, 179, 0.2)',
        mode='lines', line=dict(color='#4b8bdf', width=3, dash='dash'),
        name='1-Cmt Fit (AUC Integration)'
    ))

    # 2-Cmt Line
    fig.add_trace(go.Scatter(
        x=t, y=c_2cmt_ss, mode='lines',
        line=dict(color='#e74c3c', width=3),
        name='2-Cmt MAP Fit (Steady State)'
    ))

    # Shifted Lab Values
    rel_times = []
    lab_vals = []
    for _, level in levels.iterrows():
        lvl_time = level['Time_hr']
        past_doses = doses[doses['Time_hr'] <= lvl_time]
        if len(past_doses) > 0:
            ld = past_doses.iloc[-1]
            t_since = lvl_time - ld['Time_hr']
            if level['Level'] > 0:
                rel_times.append(t_since)
                lab_vals.append(level['Level'])

    fig.add_trace(go.Scatter(
        x=rel_times, y=lab_vals, mode='markers',
        marker=dict(color='#ecf0f1', size=12, symbol='x', line=dict(width=2, color='rgba(255,255,255,0.8)')),
        name='Measured Levels (Shifted)'
    ))

    # Styling for professional dashboard
    fig.update_layout(
        title="Single Interval Kinetics: 1-Compartment vs 2-Compartment MAP",
        xaxis_title="Time Since Last Dose (hrs)",
        yaxis_title="Vancomycin Concentration (mg/L)",
        template="plotly_dark",
        paper_bgcolor="#1e2129",
        plot_bgcolor="#1e2129",
        font=dict(color="#ecf0f1"),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(t=60, b=50, l=60, r=40)
    )

    # Add off-white outlines to the grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#34495e', zerolinecolor='#bdc3c7')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#34495e', zerolinecolor='#bdc3c7')

    return fig


def get_predicted_regimen_plot(engine, clinical_data, dose_amt, interval, t_inf):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # Use MAP if calibrated, else Prior
    params = engine.map_params if (engine.calibrated and engine.map_params is not None) else engine.population_mean

    # Base dataframe
    combined_df = clinical_data.copy() if not clinical_data.empty else pd.DataFrame(
        columns=['Event', 'Time_hr', 'Dose', 'InfusionTime'])

    # Identify last dose time
    doses_hist = combined_df[combined_df['Event'] == 'Dose']
    last_dose_time = doses_hist['Time_hr'].max() if not doses_hist.empty else 0.0

    # Append recommended doses
    future_doses = []
    num_future_doses = int(np.ceil(72.0 / interval)) + 1
    for i in range(1, num_future_doses + 1):
        future_doses.append({
            'Event': 'Dose',
            'Time_hr': last_dose_time + (i * interval),
            'Dose': dose_amt,
            'InfusionTime': t_inf
        })

    future_df = pd.DataFrame(future_doses)

    # Filter out empty dataframes to avoid Pandas concatenation FutureWarnings
    dfs_to_concat = [df for df in [combined_df, future_df] if not df.empty]
    proj_df = pd.concat(dfs_to_concat, ignore_index=True) if dfs_to_concat else pd.DataFrame(
        columns=['Event', 'Time_hr', 'Dose', 'InfusionTime'])

    # Generate time grid
    start_time = combined_df['Time_hr'].min() if not combined_df.empty else 0.0
    end_time = last_dose_time + 72.0
    times = np.linspace(start_time, end_time, 1500)

    # Main Line
    conc = engine._compute_superposition(params, proj_df, times)

    # CI Simulation (if calibrated)
    if engine.calibrated and engine.posterior_covariance is not None:
        # Sample parameters
        n_samples = 100
        samples = np.random.multivariate_normal(np.log(engine.map_params), engine.posterior_covariance, n_samples)
        sample_params = np.exp(samples)

        # Collect concs
        all_concs = np.zeros((n_samples, len(times)))
        for i in range(n_samples):
            all_concs[i, :] = engine._compute_superposition(sample_params[i], proj_df, times)

        lower_ci = np.percentile(all_concs, 2.5, axis=0)
        upper_ci = np.percentile(all_concs, 97.5, axis=0)
    else:
        lower_ci, upper_ci = None, None

    fig = go.Figure()

    # Shade Forecasted Area (The projection part: last_dose_time to end_time)
    fig.add_vrect(x0=last_dose_time, x1=end_time, fillcolor="rgba(200, 200, 200, 0.2)", layer="below", line_width=0)

    # Goal Trough band & Peak Line
    fig.add_hrect(y0=engine.trough_min, y1=engine.trough_max, fillcolor="gray", opacity=0.15, layer="below",
                  line_width=0)
    fig.add_hline(y=engine.peak, line_dash="dash", line_color="red", layer="below")

    # Plot CI
    if lower_ci is not None and upper_ci is not None:
        fig.add_trace(
            go.Scatter(x=times, y=upper_ci, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=times, y=lower_ci, mode='lines', line=dict(width=0), fill='tonexty',
                                 fillcolor='rgba(46, 204, 113, 0.3)', name='95% Forecast CI'))

    # Concentration Profile
    fig.add_trace(go.Scatter(
        x=times, y=conc, mode='lines',
        line=dict(color='#2ecc71', width=3),
        name='Concentration Profile'
    ))

    # Mark Doses
    fig.add_trace(go.Scatter(
        x=proj_df[proj_df['Event'] == 'Dose']['Time_hr'],
        y=np.zeros(len(proj_df[proj_df['Event'] == 'Dose'])),
        mode='markers', marker=dict(symbol='triangle-up', size=8, color='white'),
        name='Dose Administrations'
    ))

    fig.update_layout(
        title=f"72hr Projection: {dose_amt:.0f} mg q{interval:.0f}hr",
        xaxis_title="Time Since First Dose (Hours)",
        yaxis_title="Vancomycin Concentration (mg/L)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#34495e', zerolinecolor='#bdc3c7')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#34495e', zerolinecolor='#bdc3c7')

    return fig
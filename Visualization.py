import pandas as pd
import numpy as np
import plotly.graph_objects as go


def get_pk_table(engine):
    """
    Constructs a DataFrame comparing Population Priors to MAP estimates.
    """
    param_names = ['Central Volume (Vc)', 'Peripheral Volume (Vp)', 'Clearance (CL)', 'Intercompartmental Q']
    units = ['L', 'L', 'L/hr', 'L/hr']

    table_data = {
        'Parameter': param_names,
        'Prior (Pop)': np.round(engine.population_mean, 2)
    }

    # Insert MAP estimates if the model has been fit
    if engine.calibrated:
        table_data['Updated (MAP)'] = np.round(engine.map_params, 2)

    table_data['Units'] = units

    return pd.DataFrame(table_data)


def get_concentration_plot(engine, clinical_data, show_prior=True, show_fit=True, show_labs=True):
    """
    Builds the interactive Plotly figure for vancomycin concentrations.
    """
    fig = go.Figure()

    # 1. Shaded Trough Goal (Layered below the curves)
    fig.add_hrect(
        y0=engine.trough_min, y1=engine.trough_max,
        fillcolor="gray", opacity=0.2, layer="below", line_width=0,
        annotation_text="Goal Trough", annotation_position="top right",
        annotation_font_color="gray"
    )

    # 2. Peak Threshold Line (Red, Dashed)
    fig.add_hline(
        y=engine.peak, line_dash="dash", line_color="red", layer="below",
        annotation_text="Peak Threshold", annotation_position="top right",
        annotation_font_color="red"
    )

    # --- GENERATE SIMULATION ARRAYS ---
    times, prior_conc = engine.simulate_profile(engine.population_mean, clinical_data)
    times, fit_conc = engine.simulate_profile(engine.map_params, clinical_data)
    # ---------------------------------------------------------------------------------

    # 3. Prior PK (White Line)
    if show_prior:
        fig.add_trace(go.Scatter(
            x=times, y=prior_conc,
            mode='lines', line=dict(color='#95a5a6', width=2),
            name='Population Prior'
        ))

    # 4. Fit PK (Green Line)
    if show_fit:
        fig.add_trace(go.Scatter(
            x=times, y=fit_conc,
            mode='lines', line=dict(color='#2ecc71', width=2.5),
            name='Fitted MAP'
        ))

    # 5. Lab Values (Red Dots)
    if show_labs and not clinical_data.empty:
        labs = clinical_data[clinical_data['Event'] == 'Level']
        fig.add_trace(go.Scatter(
            x=labs['Time_hr'], y=labs['Level'],
            mode='markers', marker=dict(color='red', size=10, symbol='circle'),
            name='Measured Levels'
        ))

    # 6. Formatting for Streamlit Integration
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time (Hours)",
        yaxis_title="Vancomycin Concentration (mg/L)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig
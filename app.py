import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, date
from VancyDoodle import VancomycinBayesEngine

n_sims = 50000

# --- 1. INITIALIZATION ---
if 'results' not in st.session_state:
    st.session_state.results = None

# Hard-coded Baseline Data (Pediatric Dosing ~15mg/kg)
if 'dose_data' not in st.session_state:
    st.session_state.dose_data = [
        {"Date": date.today(), "Time": time(8, 0), "Dose (mg)": 450.0, "Infusion (hr)": 1.0},
        {"Date": date.today(), "Time": time(16, 0), "Dose (mg)": 450.0, "Infusion (hr)": 1.0}
    ]

if 'level_data' not in st.session_state:
    st.session_state.level_data = [
        {"Date": date.today(), "Time": time(10, 30), "Level (mg/L)": 37},  # Low trough
        {"Date": date.today(), "Time": time(15, 45), "Level (mg/L)": 16}   # Low steady-state trough
    ]

st.set_page_config(page_title="Vancy-Doodle", layout="wide")
st.title("Vancy-Doodle")
st.markdown("### Pediatric Vancomycin Precision Dosing")

# --- 2. SIDEBAR (Demographics & Engine Persistence) ---
# --- 1. SIDEBAR DATA MANAGEMENT ---
with st.sidebar:
    st.sidebar.title("VancyDoodle")
    st.sidebar.caption("Precision Vancomycin Dosing v1.0 for Christian and his friends")

    # A. Demographics (Top Section)
    st.subheader("Patient Stats")
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", value=28.0, step=0.1)
        age = st.number_input("Age (years)", value=8, step=1)
    with col2:
        height = st.number_input("Height (cm)", value=128.0, step=0.1)
        scr = st.number_input("Creatinine (mg/dL)", value=0.4, step=0.1)

    st.divider()

    # B. Dosing History
    st.subheader("💊 Administered Doses")
    if 'dose_history' not in st.session_state:
        st.session_state.dose_history = pd.DataFrame([
            {"Date": date.today(), "Time": time(8, 0), "Dose (mg)": 450.0, "Infusion (hr)": 1.0}
        ])

    edited_doses = st.data_editor(
        st.session_state.dose_history,
        num_rows="dynamic",
        width='stretch',
        column_config={
            "Date": st.column_config.DateColumn("Date", format="MM/DD/YYYY", required=True),
            "Time": st.column_config.TimeColumn("Time", format="HH:mm", required=True),
            "Dose (mg)": st.column_config.NumberColumn("mg", min_value=0),
            "Infusion (hr)": st.column_config.NumberColumn("Dur", min_value=0.5, step=0.5)
        },
        key="dose_editor"
    )
    st.session_state.dose_history = edited_doses

    st.divider()

    # C. Laboratory History
    st.subheader("🧪 Measured Levels")
    if 'level_history' not in st.session_state:
        st.session_state.level_history = pd.DataFrame([
            {"Date": date.today(), "Time": time(15, 30), "Level (mg/L)": 6.5}
        ])

    edited_levels = st.data_editor(
        st.session_state.level_history,
        num_rows="dynamic",
        width='stretch',
        column_config={
            "Date": st.column_config.DateColumn("Date", format="MM/DD/YYYY", required=True),
            "Time": st.column_config.TimeColumn("Time", format="HH:mm", required=True),
            "Level (mg/L)": st.column_config.NumberColumn("mg/L", min_value=0.0)
        },
        key="level_editor"
    )
    st.session_state.level_history = edited_levels

# --- 2. ENGINE INITIALIZATION (Main Page) ---
# We initialize the engine using the sidebar variables
engine = VancomycinBayesEngine(
    weight_kg=weight,
    height_cm=height,
    age_years=age,
    creatinine=scr
)
st.session_state.pk_engine = engine

# --- 3. TABS SETUP ---
tab1, tab2, tab3 = st.tabs(["📊 Empiric Dosing", "🧬 Bayesian Refinement", "🧪 Sawchuk-Moiser (Manual PK)"])

with tab1:
    # --- EXECUTION TRIGGER ---
    if st.button("Calculate Optimal Regimen", type="primary", width="stretch"):
        with st.spinner("Running population simulations..."):
            st.session_state.results = engine.suggest_optimal_regimen(n_sims=n_sims, use_prior=True)

    # --- DISPLAY RESULTS ---
    if st.session_state.results:
        res = st.session_state.results
        best = res['best_overall']
        ld_dose = res['loading_dose_mg']
        m_dose, m_int = best['maint_mg'], best['interval_hrs']

        # Day 1 AUC Calculation for display
        t_grid_24 = np.linspace(0, 24, (24 * 60) + 1)
        day1_regimen = [{"dose": ld_dose, "start": 0.0, "t_inf": 1.5}]
        for t_start in range(m_int, 24, m_int):
            day1_regimen.append({"dose": m_dose, "start": float(t_start), "t_inf": 1.0})

        # Use posterior center if calibrated, else population prior
        center = engine.map_log_params if engine.calibrated else engine.pop_log_means
        cc_day1, _ = engine._solve_trajectory(center, t_grid_24, day1_regimen)
        predicted_day1_auc = np.mean(cc_day1[0]) * 24

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Loading Dose", f"{ld_dose} mg", f"Day 1 AUC: {predicted_day1_auc:.1f}")
        with c2:
            st.metric("Maintenance Dose", f"{m_dose} mg q{m_int}h", "Optimized SS")
        with c3:
            st.metric("PTA", f"{best['pta']:.1f}%", "Target 400-600")

        st.divider()

        # --- TESTED REGIMENS SUMMARY TABLE ---
        st.subheader("Tested Regimens Summary")
        summary_df = pd.DataFrame(res['by_interval'].values())
        summary_df = summary_df.rename(columns={
            "interval_hrs": "Interval (hrs)",
            "maint_mg": "Dose (mg)",
            "pta": "PTA (%)",
            "risk_toxic_peak": "Risk Peak >50 (%)",
            "predicted_auc": "Mean AUC"
        })
        summary_df = summary_df.sort_values(by="PTA (%)", ascending=False)

        st.dataframe(
            summary_df,
            hide_index=True,
            width="stretch",
            column_config={
                "PTA (%)": st.column_config.ProgressColumn(
                    "Probability of Target Attainment",
                    format="%.1f%%", min_value=0, max_value=100,
                ),
                "Interval (hrs)": st.column_config.NumberColumn("Dosing Interval", format="%d hrs"),
                "Dose (mg)": st.column_config.NumberColumn("Maintenance Dose", format="%d mg")
            }
        )
        st.divider()

        col_plot1, col_plot2 = st.columns([1.8, 1.2])
        with col_plot1:
            st.subheader("Concentration Projection")
            stats = engine.get_projection_stats(m_dose, m_int, ld_dose)
            st.pyplot(engine.plot_prior_projections(stats), width='stretch')

        with col_plot2:
            st.subheader("Steady-State AUC Distribution")
            samples = np.random.multivariate_normal(center, engine.full_covariance, n_sims)
            cc_ss = engine._solve_steady_state_analytical(samples, m_int, m_dose)
            auc_samples = np.mean(cc_ss, axis=1) * 24
            st.pyplot(engine.plot_risk_histogram(auc_samples), width='stretch')

with tab2:

    if "pk_engine" in st.session_state:
        engine = st.session_state.pk_engine

        st.header("🧮 Bayesian Clinical Refinement")

        if st.button("Run Bayesian Fit & Optimize"):
            with st.spinner("Calculating Bayesian shift and finding optimal doses..."):
                # 1. Gather History from Sidebar
                raw_doses = []
                for _, row in st.session_state.dose_history.dropna().iterrows():
                    dt_str = f"{row['Date'].strftime('%m/%d/%Y')} {row['Time'].strftime('%H:%M')}"
                    raw_doses.append({"mg": row['Dose (mg)'], "dt": dt_str, "t_inf": row['Infusion (hr)']})

                raw_labs = []
                for _, row in st.session_state.level_history.dropna().iterrows():
                    dt_str = f"{row['Date'].strftime('%m/%d/%Y')} {row['Time'].strftime('%H:%M')}"
                    raw_labs.append({"val": row['Level (mg/L)'], "dt": dt_str})

                # 2. Fit and Optimize
                f_doses, f_labs, _ = engine.format_gui_inputs(raw_doses, raw_labs)
                engine.fit_patient(f_doses, f_labs)

                if engine.calibrated:
                    st.success("✅ Bayesian Refinement Complete")

                    # --- RESTORED: Parameter Shift Table ---
                    st.subheader("📈 Pharmacokinetic Parameter Shift")
                    st.markdown("Comparison of population average vs. this patient's calculated values.")
                    df_params = engine.get_parameter_comparison()
                    st.table(df_params)
                    # ---------------------------------------

                    # 3. GET THE TWO RECOMMENDATIONS
                    prior_results = engine.suggest_optimal_regimen(n_sims=n_sims, use_prior=True)
                    best_empiric = prior_results['best_overall']

                    post_results = engine.suggest_optimal_regimen(n_sims=n_sims, use_prior=False)
                    best_individual = post_results['best_overall']

                    # 4. Display Individualized Recommendation
                    st.divider()
                    st.subheader("⭐ Recommended Individualized Regimen")
                    col_rec1, col_rec2, col_rec3 = st.columns(3)
                    col_rec1.metric("Dose", f"{best_individual['maint_mg']} mg")
                    col_rec2.metric("Interval", f"q{best_individual['interval_hrs']}h")
                    col_rec3.metric("Projected PTA", f"{best_individual['pta']:.1f}%")

                    # 5. Side-by-Side Comparison
                    st.divider()
                    st.subheader("🎯 Risk Profile: Empiric vs. Individualized")

                    samples = np.random.multivariate_normal(engine.map_log_params, engine.full_covariance, n_sims)

                    cc_ss_emp = engine._solve_steady_state_analytical(samples, best_empiric['interval_hrs'],
                                                                     best_empiric['maint_mg'])
                    auc_emp = np.mean(cc_ss_emp, axis=1) * 24

                    cc_ss_ind = engine._solve_steady_state_analytical(samples, best_individual['interval_hrs'],
                                                                     best_individual['maint_mg'])
                    auc_ind = np.mean(cc_ss_ind, axis=1) * 24

                    col_hist1, col_hist2 = st.columns(2)
                    with col_hist1:
                        st.write(
                            f"**Standard Empiric ({best_empiric['maint_mg']}mg q{best_empiric['interval_hrs']}h)**")
                        fig_emp = engine.plot_risk_histogram(auc_emp)
                        st.pyplot(fig_emp)

                    with col_hist2:
                        st.write(
                            f"**Individualized ({best_individual['maint_mg']}mg q{best_individual['interval_hrs']}h)**")
                        fig_ind = engine.plot_risk_histogram(auc_ind)
                        st.pyplot(fig_ind)

                    st.write("---")
                    st.subheader("📈 Predicted Clinical Trajectory")

                    # 1. Re-format UI data for the engine
                    plot_doses = []
                    for d in st.session_state.dose_data:
                        dt_str = datetime.combine(d['Date'], d['Time']).strftime('%m/%d/%Y %H:%M')
                        plot_doses.append({'dt': dt_str, 'mg': d['Dose (mg)'], 't_inf': d['Infusion (hr)']})

                    plot_labs = []
                    for l in st.session_state.level_data:
                        dt_str = datetime.combine(l['Date'], l['Time']).strftime('%m/%d/%Y %H:%M')
                        plot_labs.append({'dt': dt_str, 'val': l['Level (mg/L)']})

                    # 2. Get formatted inputs and run the plot
                    try:
                        formatted_doses, formatted_labs, t0 = engine.format_gui_inputs(plot_doses, plot_labs)

                        with st.spinner("Calculating confidence intervals..."):
                            fig_clinical = engine.plot_clinical_trajectory(
                                doses=formatted_doses,
                                labs=formatted_labs,
                                recommendation=st.session_state.results['best_overall']
                            )
                            st.pyplot(fig_clinical, width='stretch')
                    except Exception as e:
                        st.error(f"Error generating trajectory: {e}")

with tab3:
    st.header("🧪 Sawchuk-Moiser (Manual PK)")
    st.info(
        "Input the timestamps for the dose and the two resulting levels to calculate patient-specific 1-compartment kinetics.")

    # --- INPUT SECTION ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dose Event")
        sm_dose = st.number_input("Dose Amount (mg)", value=1000, step=250)
        sm_tinf = st.number_input("Infusion Time (hr)", value=1.5, step=0.5)
        sm_tau = st.number_input("Dosing Interval (hr)", value=12.0, step=1.0)

        d_start = st.date_input("Date Dose Started", value=date.today(), key="sm_d1")
        # step=60 allows selecting any minute
        t_start = st.time_input("Time Dose Started", value=time(8, 0), key="sm_t1", step=60)
        dt_start = datetime.combine(d_start, t_start)

    with col2:
        st.subheader("Measured Levels")
        c_p = st.number_input("Peak Level (mg/L)", value=30.0)
        d_p = st.date_input("Date Peak Drawn", value=date.today(), key="sm_d2")
        # step=60 allows selecting any minute
        t_p = st.time_input("Time Peak Drawn", value=time(10, 30), key="sm_t2", step=60)
        dt_peak = datetime.combine(d_p, t_p)

        st.write("---")

        c_t = st.number_input("Trough Level (mg/L)", value=10.0)
        d_t = st.date_input("Date Trough Drawn", value=date.today(), key="sm_d3")
        # step=60 allows selecting any minute
        t_t = st.time_input("Time Trough Drawn", value=time(19, 30), key="sm_t3", step=60)
        dt_trough = datetime.combine(d_t, t_t)

    # --- CALCULATION ---
    if st.button("Calculate Sawchuk-Moiser Parameters"):
        if dt_peak <= dt_start or dt_trough <= dt_peak:
            st.error("Timeline Error: Ensure the peak is after the dose start, and the trough is after the peak.")
        else:
            res = engine.calculate_sawchuk_moiser(
                sm_dose, sm_tinf, c_p, c_t,
                dt_start, dt_peak, dt_trough, sm_tau
            )

            # Display Plot
            st.pyplot(res['fig'])

            # Display Relative Times for User Reference
            st.success(
                f"Calculated relative timing: Peak was drawn at {res['t_peak_rel']:.2f}h and Trough at {res['t_trough_rel']:.2f}h post-dose start.")

            # Display Results Table
            st.subheader("📊 Kinetic Parameters")
            metrics = {
                "Parameter": ["Elimination Rate (k)", "Clearance (CL)", "Vol. of Distribution (Vd)", "Vd per kg",
                              "Predicted AUC"],
                "Value": [
                    f"{res['k']:.4f} hr⁻¹",
                    f"{res['cl']:.2f} L/hr",
                    f"{res['vd']:.2f} L",
                    f"{res['vd_l_kg']:.2f} L/kg",
                    f"{res['auc']:.1f} mg·h/L"
                ]
            }
            st.table(pd.DataFrame(metrics))
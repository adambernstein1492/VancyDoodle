import streamlit as st
import pandas as pd
import IOfunctions
import Visualization
from engine import VancomycinBayesEngine

########################################## TITLE PAGE ##############################################
st.set_page_config(page_title='Pediatric Vancomycin Model-Informed Precision Dosing Tool',
                   page_icon='logo1.png',
                   layout="wide")

st.title('Pediatric Vancomycin Model-Informed Precision Dosing Tool', width='stretch')
st.sidebar.image('logo1.png')
st.subheader('An interactive vancomycin dosing tool meant for demonstration purposes, not intended for clinical use.')
st.divider(width='stretch')
########################################## TITLE PAGE ##############################################


######################################## INITIALIAZATION ###########################################
# Initialize Data
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = patient_data = {'Age': [], 'Weight': [], 'Height': [], 'Creatinine': []}
    st.session_state['patient_data_updated'] = False

if 'doses' not in st.session_state:
    st.session_state['doses'] = pd.DataFrame({'Dose': [], 'InfusionTime': [], 'DateTime': []})

if 'levels' not in st.session_state:
    st.session_state['levels'] = pd.DataFrame({'Level': [], 'DateTime': []})

if 'clinical_data' not in st.session_state:
    st.session_state['clinical_data'] = pd.DataFrame()

if 'prior_ci' not in st.session_state:
    st.session_state['prior_ci'] = (None, None)

if 'fit_ci' not in st.session_state:
    st.session_state['fit_ci'] = (None, None)

if 'dosing_regimen' not in st.session_state:
    st.session_state['dosing_regimen'] = False

if 'specific_doses' not in st.session_state:
    st.session_state['specific_doses'] = False

if 'data_fit' not in st.session_state:
    st.session_state['data_fit'] = False

if 'model_initialized' not in st.session_state:
    st.session_state['model_initialized'] = False

if 'bayes_engine' not in st.session_state:
    st.session_state['bayes_engine'] = None

if 'doses_added' not in st.session_state:
    st.session_state['doses_addes'] = False

if 'levels_added' not in st.session_state:
    st.session_state['levels_added'] = False
######################################## INITIALIAZATION ###########################################


########################################## INPUT DATA ##############################################
# Input Patient Data
st.sidebar.header('Patient Data')
with st.sidebar.popover('Input Patient Data'):
    age = st.number_input(label='Age', value=10, key='age')
    weight = st.number_input(label='Weight (kg)', value=30.0, key='weight')
    height = st.number_input(label='Height (cm)', value=140, key='height')
    creatinine = st.number_input(label='Creatinine', value=0.62, key='creatinine')

    if st.button(label='Update Patient Info'):
        st.session_state['patient_data_updated'] = True

st.session_state['patient_data'] = {'Age': age, 'Weight': weight, 'Height': height, 'Creatinine': creatinine}
st.sidebar.table(st.session_state['patient_data'], border='horizontal')

if st.session_state['patient_data_updated']:
    if st.sidebar.button(label='Initialize the Selected Model',
                         disabled=(not st.session_state['patient_data_updated'])):
        st.session_state['bayes_engine'] = VancomycinBayesEngine(
            st.session_state['patient_data']['Weight'],
            st.session_state['patient_data']['Height'],
            st.session_state['patient_data']['Age'],
            st.session_state['patient_data']['Creatinine']
        )
        st.session_state['model_initialized'] = True
        st.session_state['data_fit'] = False

        # Reset CI states on initialization
        st.session_state['prior_ci'] = (None, None)
        st.session_state['fit_ci'] = (None, None)

st.sidebar.divider(width='stretch')

# Input Dosing Information including dose datetime, dose value, and infusion time
# This offers two methods to input the data
st.sidebar.header('Vancomycin Dosing Information')


def lock_regimen_menu():
    st.session_state['specific_doses'] = True


def lock_single_dose_menu():
    st.session_state['dosing_regimen'] = True


col1, col2, col3 = st.sidebar.columns([0.4, 0.1, 0.5], vertical_alignment='center')

with col1:
    with st.popover('Add a dose', width='stretch', disabled=st.session_state['dosing_regimen']):
        dose = st.number_input(label='Dose (mg)', value=20 * st.session_state['patient_data']['Weight'], key='dose1')
        InfusionTime = st.number_input(label='Infusion Time (hr)', value=1.0, key='InfustionTime1')
        date_time = st.datetime_input(label='Date and Time Dose Administered', format='MM/DD/YYYY', step=60)

        new_dose = pd.DataFrame({'Dose': [dose], 'InfusionTime': [InfusionTime], 'DateTime': [date_time]})

        # Append Dose to list of doses
        if st.button(label='Add Dose', key='add_single_dose', on_click=lock_regimen_menu):
            if st.session_state['doses'].empty:
                st.session_state['doses'] = new_dose.copy()
            else:
                st.session_state['doses'] = pd.concat([st.session_state['doses'], new_dose], ignore_index=True)

with col2:
    st.header('OR', text_alignment='center')

with col3:
    with st.popover('Specify Dosing Regimen', width='stretch', disabled=st.session_state['specific_doses']):
        col11, col22 = st.columns(2)
        with col11:
            dose = st.number_input(label='Dose (mg)', value=20 * st.session_state['patient_data']['Weight'],
                                   key='dose2')
            InfusionTime = st.number_input(label='Infusion Time (hr)', value=1.0, key='InfusionTime2')
            interval = st.number_input(label='Dosing Interval (hr)', value=8.0)

        with col22:
            date_time = st.datetime_input(label='Date and Time First Dose Administered', format='MM/DD/YYYY', step=60)
            num_doses = st.number_input(label='Number of Doses', value=10, key='num_doses')

        # Create an appropriately formatted list of doses
        if st.button(label='Add Dose', key='add_multiple_doses', on_click=lock_single_dose_menu):
            st.session_state['doses'] = IOfunctions.make_train_of_doses(
                st.session_state['doses'], dose, interval, InfusionTime, date_time, num_doses
            )

st.session_state['doses'] = st.session_state['doses'].reset_index(drop=True)
st.session_state['doses'] = st.sidebar.data_editor(st.session_state['doses'], num_rows='dynamic', hide_index=True)
st.sidebar.divider(width='stretch')

# Input measured blood levels of vancomycin for Bayesian fitting
st.sidebar.header('Vancomycin Blood Levels')
with st.sidebar.popover('Add a level'):
    level = st.number_input(label='Vancomycin Level (mg/L)')
    date_time = st.datetime_input(label='Date and Time Level Drawn', step=60)

    new_level = pd.DataFrame({'Level': level, 'DateTime': [date_time]})

    # Append Dose to list of doses
    if st.button(label='Add Level'):
        if st.session_state['levels'].empty:
            st.session_state['levels'] = new_level.copy()
        else:
            st.session_state['levels'] = pd.concat([st.session_state['levels'], new_level], ignore_index=True)

# Display editable table of input vancomycin levels
st.session_state['levels'] = st.sidebar.data_editor(st.session_state['levels'], num_rows='dynamic', hide_index=True)
########################################## INPUT DATA ##############################################

############
# Once inputs are in, we can start running the actual fits
############

has_doses = not st.session_state['doses'].empty
has_levels = not st.session_state['levels'].empty

if st.session_state['model_initialized'] and has_doses and has_levels:
    if st.sidebar.button('Fit Data'):
        # Format input data
        engine = st.session_state['bayes_engine']

        # Ensure clinical data is fully up to date right before fitting
        st.session_state['clinical_data'] = IOfunctions.format_input_data(
            st.session_state['levels'],
            st.session_state['doses']
        )

        # Run MAP parameter estimation
        engine.fit_patient(st.session_state['clinical_data'])
        st.session_state['data_fit'] = True

        # --- CALCULATE BOTH CIs SIMULTANEOUSLY HERE ---
        # 1. Prior CI (using population parameters)
        prior_low, prior_high = engine.calculate_ci_boundaries(
            engine.population_mean,
            st.session_state['clinical_data']
        )
        st.session_state['prior_ci'] = (prior_low, prior_high)

        # 2. Post-Fit MAP CI (using individualized parameters & posterior covariance)
        fit_low, fit_high = engine.calculate_ci_boundaries(
            engine.map_params,
            st.session_state['clinical_data']
        )
        st.session_state['fit_ci'] = (fit_low, fit_high)

        st.sidebar.success("Optimization and uncertainty profiling complete!")

# Initialize the engine - all that needs to be done prior to this running is the patient demographics must be udpated
col1, col2, col3 = st.columns([0.3, 0.6, 0.1], vertical_alignment='center')

# --- COLUMN 1: The Tables ---
if st.session_state['model_initialized']:
    col1.subheader('Pharmacokinetic Parameters')
    engine = st.session_state['bayes_engine']

    # 1. Main PK Parameter Table
    pk_df = Visualization.get_pk_table(engine)
    col1.dataframe(pk_df, hide_index=True, width='stretch')

    col1.write("")  # Spacer

    # 2. Coefficient of Variation Table (Rendered directly underneath)
    col1.subheader('Parameter Uncertainty (CV%)')
    cv_df = Visualization.get_cv_table(engine)
    col1.dataframe(cv_df, hide_index=True, width='stretch')

# --- COLUMNS 2 & 3: The Plot and Checkboxes ---
if st.session_state['data_fit']:
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        show_prior = st.checkbox('Prior PK', value=True)
        show_fit = st.checkbox('Fit PK', value=True)
        show_labs = st.checkbox('Lab Values', value=True)
        show_ci = st.checkbox('Show 95% CIs', value=False)

        st.write("---")
        st.caption("Predicted Steady-State AUC:")

        engine = st.session_state['bayes_engine']
        auc_ss = engine.get_steady_state_auc(st.session_state['clinical_data'])

        if auc_ss:
            # Determine color
            if 400 <= auc_ss <= 600:
                color = "green"
            elif auc_ss < 400:
                color = "orange"
            else:
                color = "red"

            st.markdown(f"### <span style='color:{color}'>{auc_ss:.1f}</span>", unsafe_allow_html=True)
        else:
            st.text("—")

    with col2:
        st.subheader('Predicted Vancomycin Concentrations')

        # Safely extract the calculated boundaries from session state
        prior_bounds = st.session_state.get('prior_ci', (None, None))
        fit_bounds = st.session_state.get('fit_ci', (None, None))

        fig = Visualization.get_concentration_plot(
            engine=st.session_state['bayes_engine'],
            clinical_data=st.session_state['clinical_data'],
            show_prior=show_prior,
            show_fit=show_fit,
            show_labs=show_labs,
            show_ci=show_ci,
            prior_ci_bounds=prior_bounds,
            fit_ci_bounds=fit_bounds
        )

        st.plotly_chart(fig, width='stretch')

# --- 1-COMPARTMENT ESTIMATION ---
if st.session_state.get('model_initialized', False) and st.session_state.get('data_fit', False):
    st.write("---")
    st.subheader("1-Compartment PK Estimation (Exponential Decay)")
    st.markdown(
        "Estimate basic PK parameters assuming simple exponential decay using all entered doses and levels. Extrapolated times are relative to the most recent dose prior to the measured level.")

    engine = st.session_state['bayes_engine']
    pk_estimates = engine.estimate_1compartment_pk(st.session_state['clinical_data'])

    if "error" in pk_estimates:
        st.warning(pk_estimates["error"])
    else:
        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
        ec1.metric("Estimated AUC24", f"{pk_estimates['Estimated AUC24']:.1f}")
        ec2.metric("Half-life", f"{pk_estimates['Half-life (hr)']:.1f} hr")
        ec3.metric("Clearance", f"{pk_estimates['Clearance (L/hr)']:.2f} L/hr")
        ec4.metric("Volume of Distribution", f"{pk_estimates['Volume of Distribution (L/kg)']:.2f} L/kg")
        ec5.metric("Estimated Cmax", f"{pk_estimates['Estimated Cmax (mg/L)']:.1f} mg/L")
        ec6.metric("Estimated Cmin", f"{pk_estimates['Estimated Cmin (mg/L)']:.1f} mg/L")

        st.write("")
        st.markdown("**Trajectory Comparison (Phase Plot)**")
        fig_1cmt = Visualization.get_1cmt_comparison_plot(engine, st.session_state['clinical_data'], pk_estimates)
        st.plotly_chart(fig_1cmt, width='stretch')

# --- REGIMEN OPTIMIZER ---
if st.session_state.get('model_initialized', False):
    st.write("---")
    st.subheader("Model-based Dose Recommendations")
    st.markdown(
        "Generates optimal doses across standard intervals and calculates their expected target attainment and toxicity risks.")

    engine = st.session_state['bayes_engine']

    # 1. Ensure the increment is in memory so we can read it before the widget renders
    if 'dose_increment' not in st.session_state:
        st.session_state['dose_increment'] = 25.0

    current_increment = st.session_state['dose_increment']

    # 2. Execution & Smart Caching
    state_id = f"{current_increment}_{engine.calibrated}"
    if engine.calibrated and engine.map_params is not None:
        state_id += f"_{engine.map_params[0]}"

    # Only simulate if the dose increment changed or the model was fitted with new data
    if st.session_state.get('regimen_state_id') != state_id or 'regimen_grid' not in st.session_state:
        with st.spinner("Simulating optimal regimens..."):
            st.session_state['regimen_grid'] = engine.suggest_regimens(dose_step=current_increment)
            st.session_state['regimen_state_id'] = state_id

    # 3. Render the Grid with Selection Enabled (UN-INDENTED!)
    grid_results = st.session_state['regimen_grid']
    if not grid_results:
        st.warning("No regimens met the minimum threshold for target attainment.")
    else:
        styled_df = Visualization.get_regimen_grid(grid_results)

        # Adding a fixed key prevents the grid from unmounting during a rerun!
        grid_event = st.dataframe(
            styled_df,
            width='content',
            hide_index=False,
            on_select="rerun",
            selection_mode="single-row",
            key="interactive_regimen_grid"
        )

        # --- Process Row Clicks ---
        current_selection = grid_event.selection.rows[0] if grid_event.selection.rows else None
        last_selection = st.session_state.get('last_grid_selection', None)

        # If the user clicked a new row, update the inputs for the predictive section
        if current_selection != last_selection:
            st.session_state['last_grid_selection'] = current_selection

            if current_selection is not None:
                # Extract the interval (e.g., 'q12hr' -> 12.0)
                interval_str = styled_df.data.index[current_selection]
                interval_val = float(interval_str.replace('q', '').replace('hr', ''))

                # Extract the dose (e.g., '575 mg' -> 575.0)
                dose_str = styled_df.data.iloc[current_selection]['Optimal Dose']
                dose_val = float(dose_str.replace(' mg', ''))

                # Force update the input widgets below
                st.session_state['eval_dose_input'] = dose_val
                st.session_state['eval_int_input'] = interval_val

    dose_increment = st.number_input("Dose Increment (mg)", min_value=5.0, max_value=250.0, step=5.0, width=150,
                                     key='dose_increment')
    st.info("The optimal dose for each interval is chosen to maximize Probability of Target Attainment (PTA).")

# --- PREDICTIVE REGIMEN EVALUATION ---
if st.session_state.get('model_initialized', False):
    st.write("---")
    st.subheader("Predictive Regimen Evaluation")
    st.markdown("Test a maintenance dosing regimen to predict steady-state performance and toxicity risks.")

    # 1. Initialize default values in session state if they don't exist yet
    if 'eval_dose_input' not in st.session_state:
        default_dose = 750.0
        if 'regimen_grid' in st.session_state and 'q8hr' in st.session_state['regimen_grid']:
            opt_dose_str = st.session_state['regimen_grid']['q8hr']['Optimal Dose']
            default_dose = float(opt_dose_str.replace(' mg', ''))

        st.session_state['eval_dose_input'] = default_dose
        st.session_state['eval_int_input'] = 8.0

    # 2. UI: Collect Inputs (Bound directly to session state keys!)
    rc1, rc2, rc3 = st.columns(3)

    # We remove the 'value=' argument because Streamlit manages the value via the 'key'
    eval_dose = rc1.number_input("Test Dose (mg)", min_value=0.0, step=25.0, key='eval_dose_input')
    eval_int = rc2.number_input("Test Interval (hrs)", min_value=1.0, step=1.0, key='eval_int_input')
    eval_tinf = rc3.number_input("Infusion Time (hrs)", min_value=0.5, value=1.0, step=0.25)

    # 2. Execution Block: Calculate automatically on input change
    engine = st.session_state['bayes_engine']

    with st.spinner("Calculating deterministic steady-state limits..."):
        # We correctly unpack all THREE returned values from engine.py
        auc_samples, peak_samples, metrics = engine.evaluate_regimen(eval_dose, eval_int, eval_tinf)

    # 3. UI: Render Metrics and Chart
    st.write("")
    mc1, mc2, mc3, mc4 = st.columns(4)

    # We use the metrics dictionary generated by the engine
    mc1.metric("Subtherapeutic Risk", f"{metrics['p_sub']:.1f}%", "< 400 AUC", delta_color="off")
    mc2.metric("Target Probability", f"{metrics['p_target']:.1f}%", "400 - 600 AUC", delta_color="normal")
    mc3.metric("Supratherapeutic Risk", f"{metrics['p_supra']:.1f}%", "> 600 AUC", delta_color="inverse")
    mc4.metric("Toxic Peak Risk", f"{metrics['p_peak']:.1f}%", "> 50 mg/L", delta_color="inverse")

    st.write("")

    # Render Plotly Histogram and Profile side-by-side or stacked
    fig_hist = Visualization.get_auc_histogram(auc_samples, engine.target_auc_min, engine.target_auc_max)
    fig_reg_prof = Visualization.get_predicted_regimen_plot(engine, st.session_state['clinical_data'], eval_dose,
                                                            eval_int, eval_tinf)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.plotly_chart(fig_hist, width='stretch')
    with col_p2:
        st.plotly_chart(fig_reg_prof, width='stretch')
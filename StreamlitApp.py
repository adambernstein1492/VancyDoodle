import streamlit as st
import pandas as pd
import IOfunctions
import Visualization
from engine import VancomycinBayesEngine

########################################## TITLE PAGE ##############################################
st.set_page_config(page_title='Vanc you very much: Vancomycin Model-Informed Precision Dosing Tool',
                   page_icon='logo.png',
                   layout="wide")
st.title('Vanc you very much: Vancomycin Model-Informed Precision Dosing Tool', width='stretch')
st.sidebar.image('logo.png')
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
    weight = st.number_input(label='Weight (kg)', value=30, key='weight')
    height = st.number_input(label='Height (cm)', value=140, key='height')
    creatinine = st.number_input(label='Creatinine', value=0.62, key='creatinine')

    if st.button(label='Update Patient Info'):
        st.session_state['patient_data_updated'] = True

st.session_state['patient_data'] = {'Age': age, 'Weight': weight, 'Height': height, 'Creatinine': creatinine}
st.sidebar.table(st.session_state['patient_data'], border='horizontal')

if st.session_state['patient_data_updated']:
    if st.sidebar.button(label='Initialize the Selected Model', disabled=(not st.session_state['patient_data_updated'])):
        st.session_state['bayes_engine'] = VancomycinBayesEngine(
            st.session_state['patient_data']['Weight'],
            st.session_state['patient_data']['Height'],
            st.session_state['patient_data']['Age'],
            st.session_state['patient_data']['Creatinine'],
            model='Smit2021'
        )

        st.session_state['model_initialized'] = True

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
        dose = st.number_input(label='Dose (mg)', value=20*st.session_state['patient_data']['Weight'], key='dose1')
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
            dose = st.number_input(label='Dose (mg)', value=20*st.session_state['patient_data']['Weight'], key='dose2')
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
    date_time = st.datetime_input(label='Date and Time Level Drawn')

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
        st.session_state['clinical_data'] = IOfunctions.format_input_data(
            st.session_state['levels'],
            st.session_state['doses']
        )

        # Run the fitting engine
        st.session_state['bayes_engine'].fit_patient(st.session_state['clinical_data'])

        # Flag that the fit is complete
        st.session_state['data_fit'] = True

# Initialize the engine - all that needs to be done prior to this running is the patient demographics must be udpated
col1, col2, col3 = st.columns([0.2, 0.7, 0.1], vertical_alignment='center')

# --- COLUMN 1: The Table ---
if st.session_state['model_initialized']:
    col1.subheader('Pharmacokinetic Parameters')
    engine = st.session_state['bayes_engine']

    # Call the new function to get the clean DataFrame
    pk_df = Visualization.get_pk_table(engine)
    col1.dataframe(pk_df, hide_index=True, width='stretch')

# --- COLUMNS 2 & 3: The Plot and Checkboxes ---
if st.session_state['data_fit']:
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        show_prior = st.checkbox('Prior PK', value=True)
        show_fit = st.checkbox('Fit PK', value=True)
        show_labs = st.checkbox('Lab Values', value=True)

    with col2:
        st.subheader('Predicted Vancomycin Concentrations')

        # Call the new function, passing the checkbox states directly into it
        fig = Visualization.get_concentration_plot(
            engine=st.session_state['bayes_engine'],
            clinical_data=st.session_state['clinical_data'],
            show_prior=show_prior,
            show_fit=show_fit,
            show_labs=show_labs
        )

        st.plotly_chart(fig, width='stretch')


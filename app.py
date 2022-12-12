
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from productml import ProductML

def relative_difference(real, predicted):
    diff = (predicted - real)/max(predicted, real)
    return diff*100

def get_all_data(start, end, turbine_id_list, progress_bar):

    increment = 1/(len(turbine_id_list))
    percent_complete = 0

    data = []
    process_data = []
    for asset_id in turbine_id_list:

        print(asset_id)

        try:
            df = ml.get_asset_data(asset_id, start, end, include_predictions=True, prediction_scenario='power_curve')
            data.append(df)
        except:
            pass

        try:
            df = ml.get_asset_processes(asset_id, num_records=100)
            process_data.append(df)
        except:
            pass

        percent_complete += increment
        if percent_complete > 1.0:
            percent_complete = 1.0
        progress_bar.progress(percent_complete)
    
    if len(data) > 0:
        data = pd.concat(data, ignore_index=True)
    else:
        data = pd.DataFrame(columns=['asset_id', 'read_at', 'power_average', 'expected_power', 'prediction_scenario'])

    progress_bar.progress(1.0)

    return data, process_data

run_state_colors = {'completed': '#CFFFB0', 'failed': '#E03616', 'pending': '#D1D1D1', 'running': '#156AC5'}

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("ML Platorm Demo")

ml = ProductML('http://ml-backend.smartwatt.net')
farms = ml.get_assets(7, 'aggregator')

col1, col2, col3 = st.columns(3)

with col1:

    farm_name = st.selectbox('Choose a wind farm', ['all']+farms['name'].tolist())
    if farm_name == 'all':
        farm_id_list = farms['id'].tolist()
    else:
        farm_id_list = [farms.loc[farms['name'] == farm_name]['id'].item()]

with col2:
    start_date = st.date_input("Start date")

with col3:
    end_date = st.date_input("End date")

col1, col2, col3 = st.columns(3)

with col1:

    if farm_name != 'all':
        turbines = ml.get_assets(7, 'child', farm_id_list[0])
        turbine_name = st.selectbox('Choose a wind turbine', ['all']+turbines['name'].tolist())
        if turbine_name == 'all':
            turbine_id_list = turbines['id'].tolist()
        else:
            turbine_id_list = [turbines.loc[turbines['name'] == turbine_name]['id'].item()]
    else:
        turbine_name = st.selectbox('Choose a wind turbine', ['all'])
        turbines = []
        for farm_id in farm_id_list:
            turbines.append(ml.get_assets(7, 'child', farm_id))
        turbines = pd.concat(turbines, ignore_index=True)
        turbine_id_list = turbines['id'].tolist()


with col2:
    start_time = st.time_input("Start time")

with col3:
    end_time = st.time_input("End time")

submit = st.button("Go!", key=1)

if submit:

    tab1, tab2, tab3, tab4 = st.tabs(["Overall", "Demo Monitoring", "Platform Monitoring", "Model Performance"])

    start = start_date.strftime('%Y-%m-%d') + ' ' + start_time.strftime('%H:%M:%S')
    end = end_date.strftime('%Y-%m-%d') + ' ' + end_time.strftime('%H:%M:%S')

    progress_bar = st.progress(0)

    asset_data, process_data = get_all_data(start, end, turbine_id_list, progress_bar)

    if len(asset_data) == 0:

        st.write("No data to show!")
    
    else:

        data_gb_time = asset_data.groupby('read_at').mean(numeric_only=True).reset_index().drop(columns=['asset_id'])
        data_gb_time = data_gb_time.rename(columns={'power_average': 'Real Power', 'expected_power': 'ML Platform Prediction', 'read_at': 'Time'})
        data_gb_time['Prediction Error (%)'] = data_gb_time.apply(lambda row: relative_difference(row['Real Power'], row['ML Platform Prediction']) if (np.nan not in (row['Real Power'], row['ML Platform Prediction'])) and (row['Real Power'] != 0 or row['ML Platform Prediction'] != 0) else np.nan, axis=1)

        asset_data['Prediction Error (%)'] = asset_data.apply(lambda row: round(relative_difference(row['power_average'], row['expected_power']), 1) if (np.nan not in (row['power_average'], row['expected_power'])) and (row['power_average'] != 0 or row['expected_power'] != 0) else np.nan, axis=1)
        data_gb_asset = asset_data[['asset_id', 'Prediction Error (%)', 'power_average']].groupby('asset_id').mean(numeric_only=True).reset_index().sort_values(by='Prediction Error (%)', ascending=False)
        data_gb_asset = data_gb_asset.rename(columns={'asset_id': 'Asset', 'power_average': 'Average Real Power', 'Prediction Error (%)': 'Average Prediction Error (%)'})
        data_gb_asset = data_gb_asset.round({'Average Real Power': 1, 'Average Prediction Error (%)': 1})
        data_gb_asset['Asset'] = data_gb_asset['Asset'].apply(lambda x: f'WT-{x}')

        with tab1:

            st.metric(label="Average Error (%)", value=data_gb_asset['Average Prediction Error (%)'].mean())
            
            st.markdown(f'<h4>Assets Overview</h4>', unsafe_allow_html=True)
            st.table(data_gb_asset)

        with tab4:

            st.subheader("Predicted Power")

            if len(asset_data) > 0:
            
                st.line_chart(data=data_gb_time, x='Time', y=['ML Platform Prediction', 'Real Power'])

                st.subheader("Relative error (%)")
                st.line_chart(data=data_gb_time, x='Time', y='Prediction Error (%)')

                st.subheader("Wind Speed")
                st.line_chart(data=data_gb_time.rename(columns={'wind_speed': 'Wind Speed'}), x='Time', y='Wind Speed')

                st.subheader("Wind Direction")
                st.line_chart(data=data_gb_time.rename(columns={'wind_direction': 'Wind Direction'}), x='Time', y='Wind Direction')

                st.subheader("Exterior Temperature")
                st.line_chart(data=data_gb_time.rename(columns={'exterior_temperature': 'Exterior Temperature'}), x='Time', y='Exterior Temperature')

            else:

                st.write("No data")

        with tab2:
            st.write("WIP")

        with tab3:

            st.markdown(f'<h4>Platform Monitoring</h4>', unsafe_allow_html=True)
                
            if len(process_data) == 0:
                st.write("No data")
            else:
                process_data = pd.concat(process_data, ignore_index=True)
    
                first = process_data['created_at'].min().strftime('%Y-%m-%d')
                final = process_data['created_at'].max().strftime('%Y-%m-%d')

                st.write(f'Obtained platfrom processes between {first} and {final}')
                st.write('')
                st.markdown(f'<h5>Number of runs per state</h5>', unsafe_allow_html=True)

                cols_list = st.columns(len(process_data['data_flow_type'].unique()))

                for i, type_flow in enumerate(process_data['data_flow_type'].unique()):
                    with cols_list[i]:
                        df = process_data.loc[process_data['data_flow_type'] == type_flow].copy().groupby('state').count().reset_index().sort_values(by='state').rename(columns={'run_id': 'Number of runs', 'data_flow_type': 'Type of process', 'state': 'State of process'})
                        
                        if type_flow == 'etl':
                            type_flow = 'ETL'
                        else:
                            type_flow = type_flow.capitalize()
                        
                        colors = []
                        for state in df['State of process'].tolist():
                            colors.append(run_state_colors[state])
                        fig = px.bar(data_frame=df, x='State of process', y='Number of runs', color='State of process', color_discrete_sequence=colors, title=type_flow)
                        fig.update_layout(showlegend=False)
                        fig.update_layout({
                            'plot_bgcolor': 'rgba(0,0,0,0)',
                            'paper_bgcolor': 'rgba(0,0,0,0)'
                        })
                        st.plotly_chart(fig, use_container_width=True)

                st.write('')
                st.markdown(f'<h5>Processes duration</h5>', unsafe_allow_html=True)

                cols_list = st.columns(len(process_data['data_flow_type'].unique()))

                for i, type_flow in enumerate(process_data['data_flow_type'].unique()):
                    with cols_list[i]:

                        df = process_data.loc[(process_data['state'] == 'completed')&(process_data['data_flow_type'] == type_flow)].copy()

                        if type_flow == 'etl':
                            type_flow = 'ETL'
                        else:
                            type_flow = type_flow.capitalize()

                        fig = px.histogram(data_frame=df, x='duration', title=type_flow, color_discrete_sequence=['#156AC5'])
                        fig.update_layout(showlegend=False)
                        fig.update_layout({
                            'plot_bgcolor': 'rgba(0,0,0,0)',
                            'paper_bgcolor': 'rgba(0,0,0,0)'
                        })
                        st.plotly_chart(fig, use_container_width=True)

                st.write('')
                st.markdown(f'<h5>Failed processes per asset</h5>', unsafe_allow_html=True)

                data_failed = process_data.loc[process_data['state'] == 'failed'].copy()
                if len(data_failed) > 0:
                    data_failed = data_failed.groupby('asset_id').count().reset_index().rename(columns={'asset_id': 'Asset', 'run_id': 'Failed runs'})
                st.bar_chart(data=data_failed, x='Asset', y='Failed runs')

                st.write('')
                st.markdown(f'<h5>Processes duration per asset</h5>', unsafe_allow_html=True)

                data_failed = process_data.loc[process_data['state'] == 'failed'].copy()
                if len(data_failed) > 0:
                    data_failed = data_failed.groupby('asset_id').mean().reset_index().rename(columns={'asset_id': 'Asset', 'duration': 'Average duration'})
                st.bar_chart(data=data_failed, x='Asset', y='Average duration')


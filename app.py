import streamlit as st
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import textwrap

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from io import BytesIO
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from io import BytesIO

from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt

import os
import requests
import json

#this is for multiple worker retrieval
import asyncio
import aiohttp



# Set page title
st.set_page_config(page_title="Library_Statistics_Panel", layout="wide")

# # Generate list of years: current year down to 5 years ago
# current_year = datetime.now().year
# years = [str(y) for y in range(current_year, current_year - 5, -1)]

######################################
#this dynamically adds next fiscal year 

# Get today's date
today = date.today()

# Determine the current fiscal year start
if today.month >= 7:  # July or later → new fiscal year started
    current_fy_start = today.year
else:  # Jan-Jun → still in previous fiscal year
    current_fy_start = today.year - 1

# Number of past fiscal years to include
num_past_years = 5

# Build list of fiscal year starts (latest first)
years = [current_fy_start - i for i in range(num_past_years)]

# Convert to fiscal year strings: "2026/27", "2025/26", etc.
fiscal_years = [f"{y}/{str(y+1)[-2:]}" for y in years]
#############################################################

# Set dropdown options for curator name
# curators_list = pd.read_excel("curators_list.xlsx")


#Now get the image as a base-64 text
#import base64

# Read and encode the image file as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("wulogo.png")


###############################################

st.markdown(f"""
    <style>
        .custom-header {{
            background-color: #A70618;
            padding: 10px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .title-block {{
            text-align: center;
            flex-grow: 1;
        }}
        .title-block h1 {{
            color: white;
            font-size: 36px;
            margin: 0;
            font-weight: bold;
        }}
        .title-block .subtitle {{
            color: white;
            font-size: 16px;
            font-weight: normal;
            margin-top: 4px;
            font-style: italic;
        }}
        .logo {{
            height: 60px;
        }}
        .stButton>button {{
            background-color: #A70618;
            color: white;
            font-weight: bold;
            border: None;
            padding: 0.5em 1.2em;
            border-radius: 5px;
        }}

        div.stDownloadButton > button {{
        background-color: #A70618 ;
        color: white;
        margin-right: 10px;
    }}

    
    </style>
    
    <div class="custom-header">
        <div class="title-block">
            <h1>Library Statistics Panel </h1>
            <div class="subtitle">Insights from library statistics</div>
        </div>
         <img class="logo" src="data:image/png;base64,{img_base64}" alt="Logo">
    </div>
    
""", unsafe_allow_html=True)

########################################

# Sidebar on the left with inputs

with st.sidebar:
    st.subheader("Role")
    selected_tab = st.selectbox("Please select a role", ["Please select","Department Admin", "Instructor"])
    # note:  it used to be "Please select", "Department Admin", "Curator"

    if selected_tab == "Department Admin":
        st.subheader("Department Admin Options")
        year_selection_da = st.selectbox("Select a fiscal year", fiscal_years, key="da_year") 
        submit_da = st.button("Submit", key="submit_da")

    elif selected_tab == "Instructor":
        st.subheader("Instructor Options")
        year_selection_curator = st.selectbox("Select a fiscal year", fiscal_years, key="curator_year")
        name_curator = st.selectbox("Select an instructor", curators_list["Curator"], key="name_curator")
        submit_curator = st.button("Submit", key="submit_curator")

#########################################################################################################################

# --- Main content area ---
if selected_tab == "Department Admin":
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["At A Glance","Instruction Levels","Departments","Demographics","Classrooms"])

    if submit_da:
        with st.spinner("Fetching and processing data..."):
            ##############################
            # --- Step 0: Fetch data from API ---
            CLIENT_ID = st.secrets["CLIENT_ID"]
            CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

            # Step 1: Get access token
            token_url = "https://wustl.libinsight.com/v1.0/oauth/token"
            token_payload = {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "client_credentials"
            }
            token_response = requests.post(token_url, data=token_payload)
            token_response.raise_for_status()
            access_token = token_response.json().get("access_token")

            # Step 2: Get data with pagination
            data_url = "https://wustl.libinsight.com/v1.0/custom-dataset/34838/data-grid"
            headers = {"Authorization": f"Bearer {access_token}"}

            # Convert selected fiscal year back to from/to dates safely
            start_year, end_year_suffix = year_selection_da.split("/")
            start_year = int(start_year)
            end_year_suffix = int(end_year_suffix)
            
            # Compute full 4-digit end year dynamically. Works across centuries
            # If suffix < start_year % 100, it means we crossed a century
            if end_year_suffix < start_year % 100:
                # Add 100 to get correct century
                end_year_full = (start_year // 100 + 1) * 100 + end_year_suffix
            else:
                end_year_full = (start_year // 100) * 100 + end_year_suffix
            
            from_date = f"{start_year}-07-01"
            to_date = f"{end_year_full}-06-30"
            
            params_base = {
                "from": from_date,
                "to": to_date
            }
            
            all_records = []
            page = 1
            
            while True:
                params = {**params_base, "page": page}
                data_response = requests.get(data_url, headers=headers, params=params)
                data_response.raise_for_status()
            
                data = data_response.json()
                records = data["payload"]["records"]
            
                if not records:
                    break  # no more data
            
                all_records.extend(records)
                page += 1
            
            # Step 3: Parse JSON
            df = pd.json_normalize(all_records)
            
            # Clean list-type columns
            for col in df.columns:
                df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            
            has_data = not df.empty


            ###############################################################
            #Note: this used to be steps 2 and 3 but it gave only 100 records
            #by default withot the pagination.
        
            # # Step 2: Get data
            # data_url = "https://wustl.libinsight.com/v1.0/custom-dataset/34838/data-grid"
            # headers = {"Authorization": f"Bearer {access_token}"}
            
            # # Convert selected fiscal year back to from/to dates safely
            # start_year, end_year_suffix = year_selection_da.split("/")
            # start_year = int(start_year)
            # end_year_suffix = int(end_year_suffix)
            
            # # Compute full 4-digit end year dynamically. Works across centuries
            # # If suffix < start_year % 100, it means we crossed a century
            # if end_year_suffix < start_year % 100:
            #     # Add 100 to get correct century
            #     end_year_full = (start_year // 100 + 1) * 100 + end_year_suffix
            # else:
            #     end_year_full = (start_year // 100) * 100 + end_year_suffix
            
            # from_date = f"{start_year}-07-01"
            # to_date = f"{end_year_full}-06-30"
            
            # params = {
            #     "from": from_date,
            #     "to": to_date
            # }
            # data_response = requests.get(data_url, headers=headers, params=params)
            # data_response.raise_for_status()

            # # Step 3: Parse JSON
            # if data_response.headers.get("Content-Type") == "application/json":
            #     data = data_response.json()
            #     df = pd.json_normalize(data["payload"]["records"])

            #     # Clean list-type columns
            #     for col in df.columns:
            #         df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

            #     has_data = not df.empty
            # else:
            #     st.error("The response is not JSON. Check endpoint or permissions.")
            #     df = pd.DataFrame()
            #     has_data = False
            ###############################################################

        ##############################
        # --- Tabs ---
        with tab1:
            if not has_data:
                st.write("No information for this query")
            else:
                # Example summary metrics
                classes_taught = len(df)
                departments_unique = df["Department"].nunique()
                df["Number of Participants"] = pd.to_numeric(df["Number of Participants"], errors="coerce").fillna(0).astype(int)
                total_participants = df["Number of Participants"].sum()

                # Metric card helper
                def metric_card(label, value, color="#2E86C1"):
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 12px; padding: 15px; 
                                    text-align: center; box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
                                    background-color: #fff; margin-bottom: 15px;">
                            <h4 style="margin-bottom: 5px; font-size: 16px; color: #555;">{label}</h4>
                            <h2 style="margin:0; color:{color};">{value}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Display main metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    metric_card("Classes", classes_taught)
                with col2:
                    metric_card("Departments", departments_unique)
                with col3:
                    metric_card("Participants", total_participants)

                        

        with tab2:
            if not has_data:
                st.write("No information for this query")
            else:
                # Levels/focus processing
                df_focus = df.dropna(subset=["Primary Focus/Type of Instruction"]).copy()
                df_focus["focus_list"] = df_focus["Primary Focus/Type of Instruction"].str.split(",")
                df_focus = df_focus.explode("focus_list")
                df_focus["focus_list"] = df_focus["focus_list"].str.strip()
                focus_counts = df_focus["focus_list"].value_counts()

                # Metric card helper
                def metric_card(label, value, color="#2E86C1"):
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 12px; padding: 15px; 
                                    text-align: center; box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
                                    background-color: #fff; margin-bottom: 15px;">
                            <h4 style="margin-bottom: 5px; font-size: 16px; color: #555;">{label}</h4>
                            <h2 style="margin:0; color:{color};">{value}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                # Display focus metrics
                cols = st.columns(3)
                for i, (focus, count) in enumerate(focus_counts.items()):
                    col = cols[i % 3]
                    with col:
                        metric_card(focus, count)


        
        with tab3:
            if not has_data:
                st.write("No information for this query")
            else:
                # Note: if sections get blocked off in the future, use plotly default settings as you did for instructors.
                # Department bar chart
                dept_counts = df["Department"].value_counts().reset_index()
                dept_counts.columns = ["department", "count"]
                dept_counts["department_wrapped"] = dept_counts["department"].apply(
                    lambda x: "<br>".join(textwrap.wrap(x, width=30))
                )
                fig = px.bar(
                    dept_counts,
                    x="count",
                    y="department_wrapped",
                    orientation="h",
                    text="count",
                    title="Count of Classes by Department"
                )
                fig.update_traces(marker_color="#215732", textposition="outside")
                fig.update_layout(
                    xaxis_title="Count",
                    yaxis_title="Department",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=50 * len(dept_counts)
                )
                st.plotly_chart(fig, use_container_width=True)

                

        with tab4:
            if not has_data:
                st.write("No information for this query")
            else:
                df_exploded = (
                    df["Primary patron demographics"]
                    .str.split(",")
                    .explode()
                    .str.strip()
                    .value_counts(normalize=True)
                    .mul(100)
                    .reset_index()
                )
                df_exploded.columns = ["Demographic", "Percentage"]
                fig = px.bar(
                    df_exploded,
                    x="Percentage",
                    y="Demographic",
                    orientation="h",
                    text="Percentage",
                    title="Proportion of Classes by Participant Demographic"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis_title="", xaxis_title="Proportion (%)", bargap=0.3)
                st.plotly_chart(fig, use_container_width=True)



        with tab5:
            if not has_data:
                st.write("No information for this query")
            else:
                # Note: if sections get blocked off in the future, use plotly default settings as you did for instructors.
                # Locations processing
                #st.write(f"{df.columns.tolist()}") #this shows you all the columns
                #locations_unique = df["Location."].unique()
                #st.write(f"{locations_unique}")

                # Locations bar chart
                location_counts = df["Location."].value_counts().reset_index()
                location_counts.columns = ["location", "count"]
                location_counts["location_wrapped"] = location_counts["location"].apply(
                    lambda x: "<br>".join(textwrap.wrap(x, width=30))
                )
                fig = px.bar(
                    location_counts,
                    x="count",
                    y="location_wrapped",
                    orientation="h",
                    text="count",
                    title="Count of Classes by Classroom"
                )
                fig.update_traces(marker_color="#F1B434", textposition="outside")
                fig.update_layout(
                    xaxis_title="Count",
                    yaxis_title="Classroom",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=50 * len(location_counts)
                )
                st.plotly_chart(fig, use_container_width=True)



############################################################################################################################








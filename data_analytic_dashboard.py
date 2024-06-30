import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from subprocess import call
from streamlit import session_state as ss
import time
import subprocess
import folium
from folium.plugins import HeatMap
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from pandas_profiling import ProfileReport
import ydata_profiling as pp
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer


st.set_page_config(page_title="Loan Detection!!!",layout='wide')

st.text('This project is about exploring and visualizing the dataset for the application: 📊 Charts:')

upload_counter = 0
def upload_and_visualize_data():
    global upload_counter
    # Generate a unique key using a counter
    unique_key = f"upload_visualize_{upload_counter}"

    uploaded_file = st.file_uploader(":file_folder: Upload CSV file for Visualization", type=["csv"], key=unique_key,accept_multiple_files=False,
                                 help="Only CSV files are allowed")
    if uploaded_file is not None:
        # Display balloons
        st.balloons()
        
        # Display progress bar
        progress_bar = st.progress(0)
        
        # Update progress bar while the file is being uploaded
        for i in range(1, 6):  # Reduced to 5 iterations
            time.sleep(0.5)  # Reduced sleep time
            progress_bar.progress(i * 20)  # Increased step size to maintain progress
        
        # Display spinner while processing the uploaded file
        with st.spinner('Wait for it...'):
            time.sleep(5) # Simulating processing time
        
        # Read CSV file
        uploaded_df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset")
        st.write(uploaded_df)

        # Increment the counter for the next upload
        upload_counter += 1
        
        # Save uploaded data to session state
        ss.uploaded_df = uploaded_df
        return uploaded_df



def interactive(df):
    columns_to_include = [
    'Client_Income', 'Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own',
    'Child_Count', 'Credit_Amount', 'Loan_Annuity', 'Population_Region_Relative', 'Age_Days'
    ]

    df_subset = df[columns_to_include]
    df_subset['color_column'] = df['Client_Income']  # Add the color column to the subset

    # Streamlit app
    st.title('Parallel Coordinates Plot using Plotly and Streamlit')

    # Create parallel coordinates plot
    fig = px.parallel_coordinates(df_subset, color='color_column',
                                labels={col: col for col in df_subset.columns},
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                color_continuous_midpoint=df['Client_Income'].mean())

    # Update layout to adjust size
    fig.update_layout(title='Parallel Coordinates Plot using Plotly', 
                    height=800, width=1200)  # Adjust these values as needed

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)




def visualize_columns():
    st.subheader('Visualizing dataset columns')
    st.markdown("""
    <style>
        .st-bz {
            background-color: green;
            color: white;
            border-color: black;
            border-width: 2px;
            border-style: solid;
        }
    </style>
    """, unsafe_allow_html=True)

    uploaded_df = upload_and_visualize_data()  # Call the upload function and get the DataFrame
    if uploaded_df is not None:
        # No need to re-read the CSV file, as it's already done in the upload function
        st.write('Visualizing numeric columns')
        numeric_data = uploaded_df.select_dtypes(include=['number','float64'])
        st.write(numeric_data)
        st.write('Visualizing categorical columns')
        categorical_data = uploaded_df.select_dtypes(include=['object'])
        st.write(categorical_data)
    



def detect_outliers_iqr(df):
    outliers = []
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            unique_values = df[column].nunique()
            if unique_values > 2:  # Exclude columns with only two unique values
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                if not column_outliers.empty:  # Only add if there are outliers
                    outliers.append((column, column_outliers))
    return outliers




def data_quality_report(df):
    st.subheader("Data Quality Report")
    st.write("Identifying missing values:")
    st.write(df.isnull().sum())

    st.write("Irregular cardinality:")
    st.write(df.nunique())

    st.write("Outliers:")
    outliers = detect_outliers_iqr(df)
    for column, outliers_df in outliers:
        st.write(f"Column: {column}")
        st.write(outliers_df)



#PREPROCESS THE DATA 

def preprocess(df):

    if df['Population_Region_Relative'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Population_Region_Relative'] = df['Population_Region_Relative'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Population_Region_Relative'] = pd.to_numeric(df['Population_Region_Relative'], errors='coerce')
    if df['Age_Days'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Age_Days'] = df['Age_Days'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Age_Days'] = pd.to_numeric(df['Age_Days'], errors='coerce')
    if df['Employed_Days'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Employed_Days'] = df['Employed_Days'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Employed_Days'] = pd.to_numeric(df['Employed_Days'], errors='coerce')
    # Clean and convert 'Employed_Days' column to numeric
    if df['Registration_Days'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Registration_Days'] = df['Registration_Days'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Registration_Days'] = pd.to_numeric(df['Registration_Days'], errors='coerce')
    if df['Client_Income'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Client_Income'] = df['Client_Income'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Client_Income'] = pd.to_numeric(df['Client_Income'], errors='coerce')
    if df['Credit_Amount'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Credit_Amount'] = df['Credit_Amount'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Credit_Amount'] = pd.to_numeric(df['Credit_Amount'], errors='coerce')
    
    if df['Loan_Annuity'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['Loan_Annuity'] = df['Loan_Annuity'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['Loan_Annuity'] = pd.to_numeric(df['Loan_Annuity'], errors='coerce')
    
    if df['ID_Days'].dtype == 'object':
        # Replace non-numeric characters with empty string
        df['ID_Days'] = df['ID_Days'].str.replace('[^0-9.]', '', regex=True)
        # Convert the column to numeric
        df['ID_Days'] = pd.to_numeric(df['ID_Days'], errors='coerce')

    if df['Score_Source_3'].dtype == 'object':
        df['Score_Source_3'] = df['Score_Source_3'].str.replace('[^0-9.]', '', regex=True)
        df['Score_Source_3'] = pd.to_numeric(df['Score_Source_3'], errors='coerce')

    return df



def descriptive_feature_identification(df):
    st.subheader("Descriptive Feature Identification")
    st.write("Correlation Matrix:")
    
    # Preprocess the data
    df = preprocess(df)
    numeric_data=df.select_dtypes(include=['number','float64'])

    # Calculate the correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()
    
    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    
    # Display the plot
    st.pyplot(fig)



def create_map(df):
    
    # Sample a subset of data to make the heatmap manageable
    sample_df = df

    map_center = [sample_df['Latitude_x'].mean(), sample_df['Longitude_x'].mean()]
    map = folium.Map(location=map_center, zoom_start=5)

    # Add a heatmap layer
    heatmap = HeatMap(list(zip(sample_df['Latitude_x'], sample_df['Longitude_x'])), min_opacity=0.2, radius=15, blur=15)
    map.add_child(heatmap)

    # Save the map as an HTML file
    
    # Display the map
    map.save("heatmap.html")

# Display the map
    st.components.v1.html(open("heatmap.html", "r").read(), width=700, height=500)


    
def create_visualizations(df):
    st.header("Plotting of various plots in the dataset")
    st.subheader("Data Visualizations")
    df = preprocess(df)
    
    # Add a selectbox to choose the type of plot
    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Pair Plot","Scatter Plot","3D Plot","Box Plot","Violin Plot","Line Plot","Pie Chart","Heatmap","Count Plot","Joint Plot","Area Plot","Sub category plot"])
    if plot_type == "Histogram":
                x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['number','float64']).columns)
                y_axis_column = st.selectbox("Select column for Y-axis", df.columns)
                num_bins = st.slider("Select number of bins", min_value=10, max_value=100, value=30)
                fig = px.histogram(df, x=x_axis_column, y=y_axis_column, nbins=num_bins, histfunc='sum', height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Pair Plot":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.snow()
        st.toast('Need Internet connection')
        st.header("Plotting of various plots in the dataset")
        sns.pairplot(df.iloc[:,:8])
        st.pyplot()

    elif plot_type=="Scatter Plot":
        x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['object']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis", df.select_dtypes(include=['number','float64']).columns)
        fig = px.scatter(df, x=x_axis_column, hover_data=[x_axis_column], y=y_axis_column, height=300) 
        st.plotly_chart(fig, use_container_width=True)

    
    elif plot_type == "3D Plot":
        x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['object']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis", df.select_dtypes(include=['object']).columns)
        z_axis_column = st.selectbox("Select column for Z-axis", df.select_dtypes(include=['number','float64']).columns)
        fig = px.scatter_3d(df, x=x_axis_column, y=y_axis_column, z=z_axis_column, color=z_axis_column)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Violin Plot":
        x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['object']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis", df.select_dtypes(include=['number','float64']).columns)
        fig = px.violin(df, x=x_axis_column, y=y_axis_column)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Line Plot":
        x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['object']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis", df.select_dtypes(include=['number','float64']).columns)
        fig = px.line(df, x=x_axis_column, y=y_axis_column)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Pie Chart":
        category_column = st.selectbox("Select column for category", df.select_dtypes(include=['object']).columns)
        fig = px.pie(df, names=category_column)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Box Plot":
        x_axis_column = st.selectbox("Select column for X-axis (categorical)", df.select_dtypes(include=['object']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis (numerical)", df.select_dtypes(include=['number','float64']).columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=x_axis_column, y=y_axis_column, data=df, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Heatmap":
        numeric_df = df.select_dtypes(include=['number', 'float64'])
        fig = px.imshow(numeric_df.corr(), labels=dict(x="Features", y="Features", color="Correlation"),
                        x=numeric_df.corr().columns, y=numeric_df.corr().columns)
        st.plotly_chart(fig)

    elif plot_type == "Count Plot":
        category_column = st.selectbox("Select column for category", df.select_dtypes(include=['object']).columns)
        fig = px.histogram(df, x=category_column)
        st.plotly_chart(fig)

    elif plot_type == "Joint Plot":
        x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['number','float64']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis", df.select_dtypes(include=['number','float64']).columns)
        fig = px.scatter(df, x=x_axis_column, y=y_axis_column)
        st.plotly_chart(fig)

    elif plot_type == "Area Plot":
        x_axis_column = st.selectbox("Select column for X-axis", df.select_dtypes(include=['object']).columns)
        y_axis_column = st.selectbox("Select column for Y-axis", df.select_dtypes(include=['number','float64']).columns)
        fig = px.area(df, x=x_axis_column, y=y_axis_column)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type=="Sub category plot":
        st.subheader(":point_right: Data Summary")
    
        with st.expander("Summary Table"):
            df_sample = df[0:5][["Client_Income", "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own", "Child_Count", "Credit_Amount"]]
            fig = ff.create_table(df_sample, colorscale="Cividis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Fill NaN values in 'Child_Count' with 0
        df["Child_Count"] = df["Child_Count"].fillna(0)
        
        # Create a scatter plot
        data1 = px.scatter(df, x="Client_Income", y="Credit_Amount", size="Child_Count")
        data1.update_layout(
            title="Relationship between Client Income and Credit Amount using Scatter Plot",
            title_font=dict(size=20),
            xaxis=dict(title="Client Income", titlefont=dict(size=19)),
            yaxis=dict(title="Credit Amount", titlefont=dict(size=19))
        )
        st.plotly_chart(data1, use_container_width=True)
        
        # Display filtered data with a gradient background
        filtered_df = df.iloc[:500, 1:20:2]
        st.write(filtered_df.style.background_gradient(cmap="Oranges"))
        
        # Provide option to download the dataset as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")




def exploratory_data_analysis(df):
    st.subheader("Exploratory Data Analysis")
    st.write("Basic Statistics:")
    st.write(df.describe())

    st.write("DataFrame Info:")
    st.write(df.info())
    
    st.write("DataFrame Shape:")
    st.write(df.shape)

    st.write("DataFrame Columns:")
    st.write(df.columns)
    
    df=preprocess(df)

    # Display data types
    st.write("Data Types:")
    st.write(df.dtypes)

    # Visualize distributions of numeric columns
    st.write("Distributions of Numeric Columns:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], ax=ax)
        st.pyplot(fig)

    # Visualize relationships between numeric columns
    st.write("Relationships between Numeric Columns:")
    sns.pairplot(df[numeric_cols])
    st.pyplot()

    # Visualize categorical columns
    st.write("Visualizing Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    # Add code to conduct exploratory data analysis here




def evaluate_visualization_design(df):
    st.subheader("Evaluate Visualization Design")
    st.title('Interactive Plot with Dropdowns')

    # Dropdown for Client_Income_Type filtering
    income_type = st.selectbox(
        'Select Client Income Type',
        ['All', 'Commercial', 'Service', 'Retired']
    )

    # Dropdown for plot type
    plot_type = st.selectbox(
        'Select Plot Type',
        ['Scatter Plot', 'Box Plot', 'Violin Plot']
    )

    # Filter data based on the selected income type
    if income_type != 'All':
        filtered_df = df[df['Client_Income_Type'] == income_type]
    else:
        filtered_df = df

    # Create plot based on the selected plot type
    if plot_type == 'Scatter Plot':
        fig = px.scatter(filtered_df, x='Client_Income', y='Credit_Amount', color='Client_Income_Type')
    elif plot_type == 'Box Plot':
        fig = px.box(filtered_df, x='Client_Income_Type', y='Client_Income')
    elif plot_type == 'Violin Plot':
        fig = px.violin(filtered_df, x='Client_Income_Type', y='Client_Income')

    # Display plot
    st.plotly_chart(fig)
        # Add code to evaluate visualization design here



def evaluate_color_palettes(df):
    st.subheader("Evaluate Color Palettes")
    
    # Color palettes to evaluate
    palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    x_col='Client_Income'
    y_col='Credit_Amount'
    hue='Default'
    
    for palette in palettes:
        st.write(f"### Color Palette: {palette.capitalize()}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax,hue=hue)
        st.pyplot(fig)



def text_annotate(df):
    st.subheader("Text and Document Visualization")
    
    # Select an object column to visualize
    object_columns = df.select_dtypes(include=['object']).columns
    selected_column = st.selectbox("Select an object column to visualize", object_columns)
    
    # Generate word cloud
    if selected_column:
        text = ' '.join(df[selected_column].astype(str))  # Convert all values to strings
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        st.pyplot(fig)


def apply_data_transformations(df):

    st.subheader("Apply Data Transformations")
    df = preprocess(df)
    df=pd.get_dummies(df)
    
    # 2. Fill missing values with median
    df.fillna(df.median(), inplace=True)
    
    # 3. Normalize numerical features
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
    
    # 4. Apply log transformation to skewed numerical features
    skewed_cols = df[numerical_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_cols = skewed_cols[skewed_cols > 0.75]  # Select columns with skewness greater than 0.75
    df[skewed_cols.index] = np.log1p(df[skewed_cols.index])

    st.write(df)
    # Add code to apply data transformations here


import streamlit.components.v1 as components

def st_profile_report(profile_report):
    try:
        # Convert the profile report to HTML and render it in Streamlit
        profile_html = profile_report.to_html()
        components.html(profile_html, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Error displaying profile report: {e}")

def main():
    st.title("Data Exploration and Visualization")
    st.info('Data exploration and visualization by Ithikash ')
    with st.sidebar: 
        st.image("/Users/ithikash/Downloads/data-exploration-diagram_prev_ui.png")
        #st.title("AutoNickML")
        st.sidebar.markdown("This dashboard allows you to analyse trending 📈 datas using Python and Streamlit.")
        st.sidebar.markdown("To get started <ol><li>Enter the <i>choice</i> you wish to analyse</li> <li>Hit <i>Drag files</i>.</li> <li>Get analyzing</li></ol>",unsafe_allow_html=True)

        pages = ["Visualize Dataset", "Data Analysis","Profiling", "Plotting of data","Modelling" ,"Data transformation", "Exploratory data analysis", "Visualization based on perception", "Map visualization", "Interactive visualization"]

        # Initialize session state for page index
        if 'page_index' not in st.session_state:
            st.session_state.page_index = 0

        # Navigation with radio button
        selected_page = st.radio("Select a page", pages, index=st.session_state.page_index)
        st.session_state.page_index = pages.index(selected_page)
        
        st.info("This project application helps you build and explore your data.")

    # Display the selected page
    page = pages[st.session_state.page_index]

    if page == "Visualize Dataset":
        visualize_columns()
        if "uploaded_df" in st.session_state:
            st.subheader("Uploaded Dataset")
            st.write(st.session_state.uploaded_df)
        else:
            st.warning("Please upload a dataset first.")
    elif page == "Interactive visualization":
        st.subheader("Interactive visualization")
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            uploaded_df = preprocess(uploaded_df)
            interactive(uploaded_df)
        else:
            st.warning("Please upload a dataset first.")
    elif page == "Data Analysis":
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            uploaded_df = preprocess(uploaded_df)
            data_quality_report(uploaded_df)
            descriptive_feature_identification(uploaded_df)
        else:
            st.warning("Please upload a dataset first.")

    elif page == "Profiling":

        if "uploaded_df" in st.session_state:
            st.subheader("Automated Exploratory Data Analysis")
            uploaded_df = st.session_state.uploaded_df
            uploaded_df = preprocess(uploaded_df)
            df = uploaded_df.iloc[:, :10]

            profile_report = ProfileReport(df)

            st_profile_report(profile_report)

        else:
            st.warning("Please upload a dataset first.")


    elif page == "Modelling":
        if "uploaded_df" in st.session_state:
            st.subheader("Automated Exploratory Data Analysis")
            uploaded_df = st.session_state.uploaded_df
            uploaded_df = preprocess(uploaded_df)
            df=uploaded_df
    
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            df = df.dropna(subset=[chosen_target])

            if st.button('Run Modelling'): 
             
                imputer = SimpleImputer(strategy='most_frequent')
                df['Default'] = imputer.fit_transform(df[['Default']])
                df['Default'] = df['Default'].astype(int)
                X = df.drop(columns=[chosen_target,'Longitude_y','Latitude_y'], axis=1)  # Replace 'Default' with your actual target column
                y = df['Default']  # Target variable already handled
                numerical=X.select_dtypes(include=['float64','number'])
                X=numerical

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a random forest classifier
                #model = RandomForestClassifier()
                #model.fit(X_train, y_train)

                # Save the trained model
                with open('saved_model.pkl', 'rb') as file:
                   loaded_model = pickle.load(file)


                # Make predictions on the test set
                y_pred = loaded_model.predict(X_test)

                y_pred = y_pred.astype(int)
                y_test = y_test.astype(int)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")


    elif page == "Plotting of data":
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            create_visualizations(uploaded_df)
        else:
            st.warning("Please upload a dataset first.")
    elif page == "Data transformation":
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            apply_data_transformations(uploaded_df)
        else:
            st.warning("Please upload a dataset first.")
    elif page == "Exploratory data analysis":
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            exploratory_data_analysis(uploaded_df)
        else:
            st.warning("Please upload a dataset first.")
    elif page == "Visualization based on perception":
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            uploaded_df = preprocess(uploaded_df)
    
            # Create tabs for this section
            tab1, tab2, tab3 = st.tabs([
                "Evaluate Visualization Design", "Evaluate Color Palettes", "Text and Document Visualization"
            ])

            with tab1:
                evaluate_visualization_design(uploaded_df)
            with tab2:
                evaluate_color_palettes(uploaded_df)
            with tab3:
                text_annotate(uploaded_df)
         
        else:
            st.warning("Please upload a dataset first.")
    elif page == "Map visualization":
        if "uploaded_df" in st.session_state:
            uploaded_df = st.session_state.uploaded_df
            create_map(uploaded_df)
        else:
            st.warning("Please upload a dataset first.")

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("⬅️"):
            if st.session_state.page_index > 0:
                st.session_state.page_index -= 1
                st.experimental_rerun()
    with col3:
        if st.button("➡️"):
            if st.session_state.page_index < len(pages) - 1:
                st.session_state.page_index += 1
                st.experimental_rerun()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f"### {st.session_state.page_index + 1}/{len(pages)}")


if __name__ == "__main__":
    main()

#st.select_slider-used for binning
#st.text_input
#st.number_input
#st.button('name',on_click=function_name)
#st.success('')
#st.error('')
#ml model creation so use this image :https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png




    # Split columns
    #left_col, right_col = st.columns(2)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

EMPTY_SPACE = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
# Constants
window_size = 4
series_length = 15
matrix_size = series_length - window_size + 1

# Initialize session state
if 'time_series' not in st.session_state:
    st.session_state.mask = np.array(
        [[1 if abs(i - j) <= 0 else 0 for j in range(matrix_size)] for i in range(matrix_size)])
    # st.session_state.time_series = np.array([1, 2, 3, 1, 2, 3, 1, 3, 4, 1, 2, 3, 1, 2, 3]) + np.random.rand(series_length)
    st.session_state.time_series = np.array(
        [1.63436492, 2.18271487, 3.70811004, 1.48877057, 2.97334547, 3.01957275, 1.71299617, 3.61932415, 4.70583954,
         1.32881378, 2.36883822, 3.33809397, 1.75938735, 2.37791786, 3.13643583])
    st.session_state.current_index = 0
    st.session_state.current_compared_index = 0
    st.session_state.distances_matrix = np.full((matrix_size, matrix_size), np.inf)
    st.session_state.matrix_profile = np.full(matrix_size, np.inf)
    st.session_state.matrix_profile_index = np.full(matrix_size, None)

# Sample time series data
time_series = st.session_state.time_series


# Function to compute distances and update the visualization
def compute_matrix_profile(isRow):
    end_index = st.session_state.current_index + window_size

    if end_index > len(time_series):
        # end - show with colors
        # col2.dataframe(pd.DataFrame(st.session_state.distances_matrix).style.background_gradient(cmap='viridis'))
        st.write("Matrix Profile computation is complete.")
        return
    # Extract the reference subsequence
    reference_subsequence = time_series[st.session_state.current_index:end_index]

    # Compute distances for the current window against all other subsequences
    distances = []
    to_compute = range(0, matrix_size) if isRow else [st.session_state.current_compared_index]
    for i in to_compute:
        subsequence = time_series[i:i + window_size]
        distance = np.linalg.norm(reference_subsequence - subsequence)  # You can use other distance measures here
        distances.append(distance)
        st.session_state.distances_matrix[st.session_state.current_index, i] = distance

    st.session_state.distances_matrix = st.session_state.distances_matrix
    previous_matrix_profile = st.session_state.matrix_profile
    masked_distance_matrix = np.ma.masked_array(st.session_state.distances_matrix, st.session_state.mask)
    min_along_axis_1 = masked_distance_matrix.min(axis=1)
    min_along_axis_0 = masked_distance_matrix.min(axis=0)
    st.session_state.matrix_profile = np.min([min_along_axis_1, min_along_axis_0], axis=0)

    # make it symetric before computing the arg min?
    previous_index = st.session_state.matrix_profile_index
    arg_min_along_axis_1 = masked_distance_matrix.argmin(axis=1)
    arg_min_along_axis_0 = masked_distance_matrix.argmin(axis=0)
    for i in range(len(st.session_state.matrix_profile_index)):
        min1 = st.session_state.distances_matrix[i, arg_min_along_axis_0[i]]
        min2 = st.session_state.distances_matrix[arg_min_along_axis_1[i], i]
        if min1 < min2:
            st.session_state.matrix_profile_index[i] = arg_min_along_axis_1[i]
        else:
            st.session_state.matrix_profile_index[i] = arg_min_along_axis_0[i]

    # Update the time series plot
    fig_time_series = px.line(x=range(len(time_series)), y=time_series, labels={'x': 'Time', 'y': 'Value'})
    fig_time_series.add_shape(
        type="rect",
        x0=st.session_state.current_index,
        y0=min(time_series),
        x1=end_index - 1,
        y1=max(time_series),
        line=dict(color="green", width=3),
        fillcolor="green",
        opacity=0.3,
    )
    fig_time_series.add_shape(
        type="rect",
        x0=st.session_state.current_compared_index,
        y0=min(time_series),
        x1=st.session_state.current_compared_index + window_size - 1,
        y1=max(time_series),
        line=dict(color="red", width=5),
        # fillcolor="red",
        opacity=0.3,
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_time_series)

    col2.header("Distances Matrix:")

    def style_specific_cell(x):
        color = 'background-color: orange'
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        if isRow:
            df1.iloc[st.session_state.current_index, st.session_state.current_compared_index:] = color
        else:
            df1.iloc[st.session_state.current_index, st.session_state.current_compared_index] = color
        return df1

    col2.write(pd.DataFrame(st.session_state.distances_matrix)
               .replace(np.inf, EMPTY_SPACE)
               .style
               .format(precision=3)
               .apply(style_specific_cell, axis=None)
               .to_html(), unsafe_allow_html=True)
    col2.header("Matrix Profile: ")
    changes = st.session_state.matrix_profile != previous_matrix_profile
    matrix_profile_df = pd.DataFrame(
        {"ed": st.session_state.matrix_profile, "nn": st.session_state.matrix_profile_index,
         "Changed": changes}).replace(np.inf, EMPTY_SPACE)

    def color_changed(c):
        if c["Changed"]:
            return ["background-color: orange"] * len(c)
        return [""] * len(c)

    col2.write(matrix_profile_df.transpose().style.apply(color_changed, axis=0).format(precision=3).hide(["Changed"],
                                                                                                         axis=0).to_html(),
               unsafe_allow_html=True)

    if isRow or st.session_state.current_compared_index == matrix_size - 1:
        st.session_state.current_compared_index = 0
        st.session_state.current_index += 1
    else:
        st.session_state.current_compared_index += 1

    if end_index >= len(time_series) and st.session_state.current_compared_index == 0:
        # end - show with colors
        # Update the matrix profile plot
        matrix_profile_plot = px.line(x=range(len(st.session_state.matrix_profile)), y=st.session_state.matrix_profile,
                                      labels={'x': 'Time', 'y': 'Matrix Profile'})
        col2.plotly_chart(matrix_profile_plot)
        col2.write("Matrix Profile computation is complete.")


# Streamlit app
st.title('Matrix Profile Computation Visualization')

compute__step_button = st.button("Compute Next Step")
compute_row_button = st.button("Compute Next Row")
if compute__step_button:
    compute_matrix_profile(False)
elif compute_row_button:
    compute_matrix_profile(True)

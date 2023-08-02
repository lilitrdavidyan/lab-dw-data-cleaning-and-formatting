#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import os


# In[17]:


def load_csv_data(file_path):
    # Input validation
    if not isinstance(file_path, str):
        raise ValueError("Input 'file_path' should be a string representing the file path.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"An error occurred while loading data from {file_path}: {e}")


# In[18]:


def load_excel_data(file_path):
    # Input validation
    if not isinstance(file_path, str):
        raise ValueError("Input 'file_path' should be a string representing the file path.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        raise ValueError(f"An error occurred while loading data from {file_path}: {e}")


# In[19]:


def save_data(cleaned_data, file_path):
    # Input validation
    if not isinstance(cleaned_data, pd.DataFrame):
        raise ValueError("Input 'cleaned_data' is not a valid DataFrame.")
    
    if not isinstance(file_path, str):
        raise ValueError("Input 'file_path' should be a string representing the file path.")
    
    try:
        cleaned_data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path} successfully.")
    except Exception as e:
        raise ValueError(f"An error occurred while saving data to {file_path}: {e}")


# In[20]:


# To ensure consistency and ease of use, standardize the column names of the dataframe. 
def clean_column_names(raw_data):
    # Input validation
    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError("Input 'raw_data' is not a valid DataFrame.")
    
    # Create a defensive copy of the DataFrame
    cleaned_data = raw_data.copy()
    
    # Clean column names
    cleaned_data.columns = cleaned_data.columns.str.strip().str.replace(' ', '_').str.lower().str.replace('[^\w]', '', regex=True)
    
    return cleaned_data


# In[21]:


# Renames the column names of the given date frame with according to the specified column_names dictionary
# Example column_names={'st': 'state'}

def rename_column_names(raw_data, column_names):
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise ValueError("Input 'raw_data' must be a non-empty DataFrame.")
    
    if not isinstance(column_names, dict):
        raise ValueError("Input 'column_names' must be a dictionary.")
    
    existing_columns = set(raw_data.columns)
    
    new_columns = set(column_names.values())
    
    if not new_columns.isdisjoint(existing_columns):
        raise ValueError("New column names should not overlap with existing column names.")
    
    for key in column_names:
        if key not in existing_columns:
            raise ValueError(f"Column '{key}' does not exist in the DataFrame.")
    
    raw_data = raw_data.rename(columns=column_names)
    return raw_data


# In[22]:


def fix_numeric_column_types(data):
    # Input validation
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input 'data' should be a non-empty DataFrame.")
    
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=[int, float]).columns
    
    # Convert numeric columns to appropriate numeric types
    for col in numeric_columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            print(f"Warning: Unable to convert '{col}' to numeric. It contains non-numeric values.")
    
    return data


# In[36]:


def identify_and_convert_numeric_column(data, column):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    
    if column not in data.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    # Extract the column and remove any leading/trailing whitespaces
    column_data = data[column].str.strip()
    
    # Check if all values in the column are numeric
    if column_data.str.isnumeric().all():
        data[column] = pd.to_numeric(column_data)
    
    return data


# In[37]:


def identify_and_convert_all_numeric_columns(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    
    for column in data.columns:
        data = identify_and_convert_numeric_column(data, column)
    
    return data


# In[ ]:





# In[23]:


#Remove rows with any missing values (NaN) from a pandas DataFrame.

def remove_empty_raws(data):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Remove rows with all missing values (NaN)
        data.dropna(how='all', inplace=True)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[24]:


# Drop rows with null values in specific columns.

def drop_raws_with_na_values(data, columns):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'columns' is a list of valid column names in 'data'
        invalid_columns = set(columns) - set(data.columns)
        if invalid_columns:
            raise ValueError(f"Invalid columns: {invalid_columns}. The DataFrame does not have these columns.")

        # Drop rows with NA values in the specified columns
        data.dropna(subset=columns, inplace=True)

        # Return the cleaned DataFrame
        return data

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[35]:


def get_duplicate_rows(data, columns=None):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        if columns is not None and not isinstance(columns, list):
            raise ValueError("Input 'columns' must be a list of column names.")

        if columns is None:
            duplicate_rows = data[data.duplicated()]
        else:
            if not set(columns).issubset(data.columns):
                missing_columns = set(columns) - set(data.columns)
                raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")

            duplicate_rows = data[data.duplicated(subset=columns)]

        return duplicate_rows
    
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[25]:


# Drop duplicate rows based on specific columns.

def drop_duplicates(data, columns, keep):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'columns' is a list of valid column names in 'data'
        invalid_columns = set(columns) - set(data.columns)
        if invalid_columns:
            raise ValueError(f"Invalid columns: {invalid_columns}. The DataFrame does not have these columns.")

        # Check if 'keep' argument is valid
        valid_keep_values = ['first', 'last', False]
        if keep not in valid_keep_values:
            raise ValueError(f"Invalid value for 'keep': {keep}. Valid values are 'first', 'last', and False.")

        # Drop duplicate rows based on the specified columns and 'keep' option
        data.drop_duplicates(subset=columns, keep=keep, inplace=True)

        # Return the cleaned DataFrame
        return data

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[27]:


# Replace inconsistent values with their correct counterparts
def replace_inconsistent_values(data, column, mapping):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'column' is a valid column name in 'data'
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if 'mapping' is a dictionary
        if not isinstance(mapping, dict):
            raise ValueError("'mapping' argument must be a dictionary.")

        # Replace inconsistent values in the specified column with their correct counterparts
        data[column] = data[column].replace(mapping)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[28]:


# Fill null values in a specific column of a pandas DataFrame with either the mean or median.

def fill_null_with_mean_or_median(data, column, method='mean'):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'column' is a valid column in the DataFrame
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if 'method' is valid
        if method not in ['mean', 'median']:
            raise ValueError("'method' must be one of 'mean' or 'median'.")

        # Get the data type of the column
        dtype = data[column].dtype

        # Check if the column is numeric (int64 or float64) for mean or median calculation
        if dtype != 'int64' and dtype != 'float64':
            raise ValueError(f"Column '{column}' is not numeric (int64 or float64).")

        # Fill null values based on the specified method
        if method == 'mean':
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == 'median':
            data[column].fillna(data[column].median(), inplace=True)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[29]:


# Fill null values in a pandas DataFrame with appropriate values based on column data types.


def fill_all_null_values(data):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Loop through each column
        for col in data.columns:
            # Get the data type of the column
            dtype = data[col].dtype

            # Fill null values based on data type
            if dtype == 'object':
                data[col].fillna(data[col].mode()[0], inplace=True)  # Fill with mode
            elif dtype == 'int64' or dtype == 'float64':
                data[col].fillna(data[col].mean(), inplace=True)     # Fill with mean

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[30]:


def fill_null_with_previous_or_next_value(data, column, method='previous'):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'column' is a valid column in the DataFrame
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        # Check if 'method' is valid
        if method not in ['previous', 'next']:
            raise ValueError("'method' must be one of 'previous' or 'next'.")

        # Fill null values based on the specified method
        if method == 'previous':
            data[column].fillna(method='ffill', inplace=True)
        elif method == 'next':
            data[column].fillna(method='bfill', inplace=True)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[31]:


def fill_nulls_in_dataset_with_previous_or_next(data, method='previous'):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'method' is valid
        if method not in ['previous', 'next']:
            raise ValueError("'method' must be one of 'previous' or 'next'.")

        # Iterate through each column and fill null values with the specified method
        for column in data.columns:
            fill_null_with_previous_or_next_value(data, column, method=method)

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[32]:


# Check if there are null values in a pandas DataFrame and return the list of columns with null values.

def check_null_values(data):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Get the list of columns with null values
        columns_with_nulls = data.columns[data.isnull().any()].tolist()

        return columns_with_nulls

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[33]:


# Get all rows from a pandas DataFrame that have null values in the specified list of columns.
# If the 'columns' list is empty, it will consider all columns for checking null values.

def get_rows_with_null_values(data, columns=[]):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Check if 'columns' is a list
        if not isinstance(columns, list):
            raise ValueError("Input 'columns' must be a list of column names.")

        # If 'columns' is empty, consider all columns for checking null values
        if not columns:
            rows_with_nulls = data[data.isnull().any(axis=1)]
        else:
            rows_with_nulls = data[data[columns].isnull().any(axis=1)]

        return rows_with_nulls

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")


# In[ ]:





# In[34]:


# def clean_data(raw_data):

#     return raw_data


# In[ ]:





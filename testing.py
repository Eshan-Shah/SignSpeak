import pandas as pd

def test_columns():
    # Load the CSV file
    df = pd.read_csv("data/gestures.csv")
    
    # Print out all column names
    print("Column names in the 'gestures.csv' file:")
    print(df.columns.tolist())  # List of column names

test_columns()
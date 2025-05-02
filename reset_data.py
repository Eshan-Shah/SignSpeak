def resetData():
    import pandas as pd

    # Read the CSV file
    df = pd.read_csv("data/gestures.csv")

    df = pd.DataFrame()  

    df.to_csv("data/gestures.csv", index=False)

    print("Data reset complete.")

resetData()
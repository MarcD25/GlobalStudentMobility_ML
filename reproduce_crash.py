
import pandas as pd
import streamlit as st

# Simulate the failure in load_and_merge_data
def load_and_merge_data_simulated():
    print("Simulating load failure...")
    return pd.DataFrame()

try:
    mobility_df = load_and_merge_data_simulated()
    print(f"Mobility DF shape: {mobility_df.shape}")
    # This line mirrors line 142 in app.py
    country_avg_mobility = mobility_df.groupby('country')[['inbound_rate', 'outbound_rate']].mean().dropna()
    print("Success (unexpected)")
except KeyError as e:
    print(f"Caught expected crash: KeyError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {e}")

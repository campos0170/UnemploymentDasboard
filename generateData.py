import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

# Import Data

f1 = "data/SeriesReport-20250708063052_26075d.xlsx"
f2 = "data/SeriesReport-20250708063208_9d0581.xlsx"
f3 = "data/SeriesReport-20250708063221_53a206.xlsx"
f4 = "data/SeriesReport-20250708063233_db80f1.xlsx"
f5 = "data/SeriesReport-20250708063242_4955e8.xlsx"
#f6 = "data/SeriesReport-20250708063253_4b0170.xlsx"
f6 = "data/SeriesReport-20250708063302_3ab447.xlsx"
f7 = "data/SeriesReport-20250708063313_7ee205.xlsx"
f8 = "data/SeriesReport-20250708063322_5528c9.xlsx"
f9 = "data/SeriesReport-20250708063333_8c9aff.xlsx"
f10 = "data/SeriesReport-20250708063343_177541.xlsx"
f11 = "data/SeriesReport-20250708063354_4a1357.xlsx"
f12 = "data/SeriesReport-20250708063405_cc98c3.xlsx"


# Organize Data

class dframe:
    """
    Converts the Excel file into a pandas DataFrame.
    Extracts the 'Age' group metadata and adds it as a column.
    Provides both wide and long format access.
    """

    def __init__(self, filename):
        self.filename = filename
        self._wide_df = None
        self._age_group = None

    def _load_wide(self):
        if self._wide_df is not None:
            return self._wide_df

        try:

            raw_data = pd.read_excel(self.filename, sheet_name="BLS Data Series", header=None)

            age_row = raw_data[raw_data.iloc[:, 0] == "Age:"]
            self._age_group = age_row.iloc[0, 1] if not age_row.empty else None

            header_index = raw_data[raw_data.iloc[:, 0] == "Year"].index[0]

            df = pd.read_excel(self.filename, sheet_name="BLS Data Series", skiprows=header_index)
            df["AgeGroup"] = self._age_group

            self._wide_df = df
            return df

        except Exception as e:
            print(f" Error processing {self.filename}: {e}")
            return pd.DataFrame()

    def to_wide(self):
        """
        Returns the wide-format DataFrame (original layout).
        """
        return self._load_wide()

    def to_long(self):
        """
        Returns the long-format DataFrame,
        melted by month with a Date column.
        """
        df = self._load_wide()
        if df.empty:
            return df

        df_long = df.melt(
            id_vars=["Year", "AgeGroup"],
            var_name="Month",
            value_name="Unemployment"
        )
        df_long["Date"] = pd.to_datetime(df_long["Year"].astype(str) + "-" + df_long["Month"], format="%Y-%b")
        return df_long.sort_values("Date")


df2 = dframe(f2).to_long()
df3 = dframe(f3).to_long()
df4 = dframe(f4).to_long()
#df5 = dframe(f5).to_long()
df5 = dframe(f5).to_long()
#df6 = dframe(f6).to_long();
df6 = dframe(f6).to_long()
df7 = dframe(f7).to_long()
df8 = dframe(f8).to_long()
df9 = dframe(f9).to_long()
df10 = dframe(f10).to_long()
df11 = dframe(f11).to_long()
df12 = dframe(f12).to_long()

df_l = [df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]

# Combine DataFrames

df_all = pd.concat(df_l, ignore_index=True)
df_all["Date"] = pd.to_datetime(df_all["Date"])
df_all["Unemployment"] = pd.to_numeric(df_all["Unemployment"], errors="coerce")
df_all["MonthNum"] = df_all["Date"].dt.month
df_all["Year"] = df_all["Date"].dt.year

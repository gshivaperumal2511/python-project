# Sales Data Analysis Using NumPy

## Project Title

Sales Data Analysis Using NumPy Array Operations

## Project Description

This project analyzes a supermarket sales dataset using Python and
NumPy. It performs numerical computations, statistical analysis,
indexing, slicing, broadcasting, and vectorized operations to understand
sales performance.

## Objectives

-   Load CSV dataset
-   Convert columns into NumPy arrays
-   Calculate total and average sales
-   Extract records using indexing and slicing
-   Apply statistical functions (mean, median, variance, standard
    deviation)
-   Use broadcasting to compute revenue
-   Perform vectorized computations

## Technologies Used

-   Python
-   NumPy
-   Pandas

## Dataset Contents

-   Product category
-   Quantity sold
-   Unit price
-   Total sales
-   Profit / gross income

## How to Run

1.  Install libraries: pip install numpy pandas

2.  Place CSV file in project folder

3.  Run: python sales_analysis.py

## Expected Output

-   Total and average sales
-   Dataset samples
-   Statistical values
-   Revenue calculations
-   Profit margins

## Conclusion

NumPy enables fast and efficient numerical analysis of large sales
datasets using array operations and vectorized computation.

## Author

Student Name: Course: Submission Date:



#project titanic

import pandas as pd
import numpy as np

print("Libraries Imported Successfully\n")

df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset Loaded Successfully\n")

print("Displaying First 10 Rows of the Dataset:\n")
print(df.head(10))

print("\nTotal Rows and Columns in Dataset:")
print(df.shape)

print("\nCreating Series from Columns...\n")

age_series = df["Age"]
fare_series = df["Fare"]
survival_series = df["Survived"]

print("Age Series:\n", age_series.head())
print("\nFare Series:\n", fare_series.head())
print("\nSurvival Series:\n", survival_series.head())

print("\nDataset Information:\n")
df.info()

print("\nStatistical Summary:\n")
print(df.describe())

print("\nMissing Values in Dataset:\n")
print(df.isnull().sum())

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

print("\nDetails of First Passenger using loc:\n")
print(df.loc[0])

print("\nFirst 5 Passengers using iloc:\n")
print(df.iloc[0:5])

print("\nSelected Columns (Age, Fare, Survived):\n")
print(df[["Age", "Fare", "Survived"]].head())

print("\nRenaming Column Sex to Gender\n")
df.rename(columns={"Sex": "Gender"}, inplace=True)

print("\nPivot Table: Survival based on Gender and Class\n")
pivot_table = df.pivot_table(values="Survived", index="Gender", columns="Pclass")
print(pivot_table)

print("\nPassengers Age > 30\n")
print(df[df["Age"] > 30].head())

print("\nPassengers Fare > 50\n")
print(df[df["Fare"] > 50].head())

print("\nFemale Passengers who Survived\n")
print(df[(df["Gender"] == "female") & (df["Survived"] == 1)].head())

print("\nSorting by Fare\n")
print(df.sort_values(by="Fare", ascending=False).head())

print("\nSorting by Age\n")
print(df.sort_values(by="Age").head())

print("\nSurvival Rate by Gender\n")
print(df.groupby("Gender")["Survived"].mean())

print("\nAverage Age by Passenger Class\n")
print(df.groupby("Pclass")["Age"].mean())

print("\nSurvival Count by Embarked Location\n")
print(df.groupby("Embarked")["Survived"].sum())

print("\nCreating Additional DataFrame for Class Information\n")

class_info = pd.DataFrame({
    "Pclass": [1, 2, 3],
    "Class_Type": ["First Class", "Second Class", "Third Class"]
})

merged_df = pd.merge(df, class_info, on="Pclass")

print("\nMerged Dataset Sample\n")
print(merged_df.head())

print("\nAverage Age of Survivors vs Non-Survivors\n")
print(df.groupby("Survived")["Age"].mean())

print("\nHighest Fare Paid:")
print(df["Fare"].max())

print("\nLowest Fare Paid:")
print(df["Fare"].min())

print("\nSurvival Percentage:")
survival_percentage = (df["Survived"].sum() / len(df)) * 100
print(round(survival_percentage, 2), "%")

print("\nProject Analysis Completed Successfully!")

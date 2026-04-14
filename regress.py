"""

Performs linear regression between petal length and sepal length
for each Iris species and generates scatter plots with fitted lines.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def perform_linear_regression(x_values, y_values):
    """
    Perform linear regression on two variables.

    Parameters:
        x_values (pd.Series): Independent variable.
        y_values (pd.Series): Dependent variable.

    Returns:
        tuple: slope, intercept
    """
    regression_result = linregress(x_values, y_values)
    return regression_result.slope, regression_result.intercept


def plot_regression(x_values, y_values, slope, intercept, species_name):
    """
    Create and save a scatter plot with a regression line.

    Parameters:
        x_values (pd.Series): Independent variable.
        y_values (pd.Series): Dependent variable.
        slope (float): Regression slope.
        intercept (float): Regression intercept.
        species_name (str): Name of the Iris species.
    """
    plt.figure()

    plt.scatter(x_values, y_values, label='Data')
    plt.plot(x_values, slope * x_values + intercept,
             color='orange', label='Fitted line')

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(f"{species_name} Regression")
    plt.legend()

    filename = f"{species_name.lower()}_regression.png"
    plt.savefig(filename)
    plt.close()


def analyze_species(dataframe):
    """
    Perform regression analysis for each species in the dataset.

    Parameters:
        dataframe (pd.DataFrame): Iris dataset.
    """
    species_list = dataframe['species'].unique()

    for species in species_list:
        species_data = dataframe[dataframe['species'] == species]

        x = species_data['petal_length_cm']
        y = species_data['sepal_length_cm']

        slope, intercept = perform_linear_regression(x, y)
        plot_regression(x, y, slope, intercept, species)

        print(f"{species}: slope={slope:.3f}, intercept={intercept:.3f}")


def main():
    """
    Main function to execute the analysis.
    """
    file_path = "iris.csv"
    iris_df = load_data(file_path)
    analyze_species(iris_df)


if __name__ == "__main__":
    main()

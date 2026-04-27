import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def make_country_data():

    data = {
        "country": [
            "Bangladesh",
            "India",
            "China",
            "United States",
            "Germany",
            "Brazil",
            "South Africa",
            "Nigeria",
            "Japan",
            "Norway",
            "Qatar",
            "Australia",
        ],
        "code": [
            "BGD",
            "IND",
            "CHN",
            "USA",
            "DEU",
            "BRA",
            "ZAF",
            "NGA",
            "JPN",
            "NOR",
            "QAT",
            "AUS",
        ],
        "co2_per_person": [
            0.6, 1.8, 7.4, 13.0, 7.7, 2.0,
            6.8, 0.6, 8.0, 7.0, 35.6, 15.0,
        ],
        "gdp_per_person": [
            1961, 1913, 10409, 63528, 46749, 6796,
            5753, 2074, 40193, 67294, 52315, 51868,
        ],
        "renewable_percent": [
            27.9, 36.0, 14.9, 8.8, 16.5, 48.4,
            10.5, 86.0, 10.4, 71.6, 0.1, 10.0,
        ],
    }

    return pd.DataFrame(data)

def cluster_countries(country_data):

    columns = [
        "co2_per_person",
        "gdp_per_person",
        "renewable_percent",
    ]

    cluster_data = country_data[columns].copy()

    scaler = StandardScaler()
    normalised_data = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=3, random_state=7, n_init=10)
    country_data["cluster"] = kmeans.fit_predict(normalised_data)

    centres = scaler.inverse_transform(kmeans.cluster_centers_)
    centre_data = pd.DataFrame(centres, columns=columns)

    order = centre_data["co2_per_person"].sort_values().index
    new_names = {}

    for new_number, old_number in enumerate(order, start=1):
        new_names[old_number] = new_number

    country_data["group"] = country_data["cluster"].map(new_names)
    centre_data["group"] = centre_data.index.map(new_names)
    centre_data = centre_data.sort_values("group")

    return country_data, centre_data

def plot_clusters(country_data, centre_data, output_folder):

    plt.figure(figsize=(8, 5))

    for group in sorted(country_data["group"].unique()):
        group_data = country_data[country_data["group"] == group]

        plt.scatter(
            group_data["gdp_per_person"],
            group_data["co2_per_person"],
            s=80,
            label=f"Group {group}",
        )

        label_offsets = {
            "BGD": (5, -12),
            "NGA": (5, 6),
            "IND": (5, 12),
        }

        for _, row in group_data.iterrows():
            offset = label_offsets.get(row["code"], (5, 5))

            plt.annotate(
                row["code"],
                (row["gdp_per_person"], row["co2_per_person"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
            )

    plt.scatter(
        centre_data["gdp_per_person"],
        centre_data["co2_per_person"],
        s=220,
        marker="X",
        color="black",
        label="Cluster centres",
    )

    plt.xscale("log")
    plt.xlabel("GDP per person (current US$, log scale)", fontsize=11)
    plt.ylabel("CO2 emissions per person", fontsize=11)
    plt.title("Country clusters using World Bank indicators", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    filename = os.path.join(output_folder, "cluster_plot.png")
    plt.savefig(filename, dpi=200)
    plt.close()

def make_china_time_series():

    data = {
        "year": [
            1990, 1995, 2000, 2005, 2010, 2015,
            2018, 2019, 2020, 2021, 2022,
        ],
        "co2_per_person": [
            2.15, 2.76, 2.70, 4.55, 6.33, 7.17,
            7.36, 7.61, 7.41, 8.05, 8.15,
        ],
    }

    return pd.DataFrame(data)

def quadratic_model(year, a_value, b_value, c_value):

    x_value = year - 1990

    return a_value * x_value ** 2 + b_value * x_value + c_value

def err_ranges(x_data, function, parameters, parameter_errors):

    lower_parameters = parameters - parameter_errors
    upper_parameters = parameters + parameter_errors

    lower = function(x_data, *lower_parameters)
    upper = function(x_data, *upper_parameters)

    lower_range = np.minimum(lower, upper)
    upper_range = np.maximum(lower, upper)

    return lower_range, upper_range

def fit_china_data(china_data):

    years = china_data["year"].to_numpy()
    co2_values = china_data["co2_per_person"].to_numpy()

    parameters, covariance = curve_fit(
        quadratic_model,
        years,
        co2_values,
    )

    parameter_errors = np.sqrt(np.diag(covariance))

    prediction_years = np.array([2030, 2040])
    predictions = quadratic_model(prediction_years, *parameters)

    lower, upper = err_ranges(
        prediction_years,
        quadratic_model,
        parameters,
        parameter_errors,
    )

    prediction_data = pd.DataFrame(
        {
            "year": prediction_years,
            "prediction": predictions,
            "lower_range": lower,
            "upper_range": upper,
        }
    )

    return parameters, parameter_errors, prediction_data

def plot_china_fit(china_data, parameters, parameter_errors, output_folder):

    years = china_data["year"].to_numpy()
    co2_values = china_data["co2_per_person"].to_numpy()

    plot_years = np.linspace(years.min(), 2040, 200)
    fitted_values = quadratic_model(plot_years, *parameters)

    lower, upper = err_ranges(
        plot_years,
        quadratic_model,
        parameters,
        parameter_errors,
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(years, co2_values, s=60, label="World Bank data")
    plt.plot(plot_years, fitted_values, label="Quadratic fit")
    plt.fill_between(plot_years, lower, upper, alpha=0.2,
                     label="Confidence range")

    plt.xlabel("Year", fontsize=11)
    plt.ylabel("CO2 emissions per person", fontsize=11)
    plt.title("Simple fitted model for China", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    filename = os.path.join(output_folder, "china_fit_plot.png")
    plt.savefig(filename, dpi=200)
    plt.close()

def main():

    output_folder = "assignment3_figures"
    os.makedirs(output_folder, exist_ok=True)

    country_data = make_country_data()
    country_data, centre_data = cluster_countries(country_data)

    print("\nClustered country data:")
    print(country_data[[
        "country",
        "co2_per_person",
        "gdp_per_person",
        "renewable_percent",
        "group",
    ]].to_string(index=False))

    print("\nBack-transformed cluster centres:")
    print(centre_data.to_string(index=False))

    plot_clusters(country_data, centre_data, output_folder)

    china_data = make_china_time_series()
    parameters, parameter_errors, prediction_data = fit_china_data(
        china_data,
    )

    print("\nFitted model parameters:")
    print(parameters)

    print("\nParameter errors:")
    print(parameter_errors)

    print("\nPredictions with confidence ranges:")
    print(prediction_data.round(2).to_string(index=False))

    plot_china_fit(
        china_data,
        parameters,
        parameter_errors,
        output_folder,
    )

    print("\nFinished. The plots were saved in:")
    print(output_folder)


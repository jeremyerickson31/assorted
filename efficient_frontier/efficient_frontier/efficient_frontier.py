
# sample calculation to make the Markowitz efficient frontier

import math
import numpy as np
import numpy.random
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


def calc_portfolio_weighted_avg(weights):
    """
    weighted average portfolio return. calculated as Sum (weight_i * return_i)
    :param weights: dataframe with asset_name as row key
    """
    weighted_return = 0
    for asset_name in weights.keys():
        weighted_return += weights[asset_name] * asset_properties[asset_name]["avg_return"]

    return weighted_return


def calc_portfolio_weighted_stddev(weights):
    """
    weighted std dev of returns. calculated as SUM_i(SUM_j( w_i * w_j * stddev_i * stddev_j * Corr_ij))
    :param weights: dataframe with asset_name as row key
    """
    weighted_risk = 0
    for asset_name_i in weights.keys():
        for asset_name_j in weights.keys():
            weighted_risk += weights[asset_name_i] * weights[asset_name_j] * \
                             asset_properties[asset_name_i]["std_dev"] * asset_properties[asset_name_j]["std_dev"] * \
                             correlations[asset_name_i][asset_name_j]

    weighted_risk = math.sqrt(weighted_risk)

    return weighted_risk


def calc_portfolio_risk_return(weights):
    """
    one function to call both risk and return calcs
    :param weights: dataframe with asset_name as row key
    :return:
    """
    portfolio_risk = calc_portfolio_weighted_stddev(weights)
    portfolio_return = calc_portfolio_weighted_avg(weights)
    sharpe_ratio = portfolio_return / portfolio_risk

    return portfolio_risk, portfolio_return, sharpe_ratio


if __name__ == "__main__":

    # manually make 6 assets and some correlations
    asset_properties = {"asset_1": {"avg_return": 0.073, "std_dev": 0.05},
                        "asset_2": {"avg_return": 0.035, "std_dev": 0.011},
                        "asset_3": {"avg_return": 0.12, "std_dev": 0.09},
                        "asset_4": {"avg_return": 0.05, "std_dev": 0.02},
                        "asset_5": {"avg_return": 0.015, "std_dev": 0.02},
                        "asset_6": {"avg_return": 0.11, "std_dev": 0.05}}

    # make correlation matrix, upper triangular only
    correlations = pd.DataFrame([[+1.00, -0.08, +0.32, -0.12, -0.30, -0.03],
                                 [+0.00, +1.00, +0.02, +0.18, +0.45, -0.24],
                                 [+0.00, +0.00, +1.00, +0.04, +0.02, -0.09],
                                 [+0.00, +0.00, +0.00, +1.00, +0.13, +0.12],
                                 [+0.00, +0.00, +0.00, +0.00, +1.00, +0.21],
                                 [+0.00, +0.00, +0.00, +0.00, +0.00, +1.00],
                                 ], index=asset_properties.keys(), columns=asset_properties.keys())
    # make upper triangular matrix symmetric about diagonal for ease of lokup
    corr_upper_tri = np.triu(correlations)
    correlations = corr_upper_tri + corr_upper_tri.T - np.diag(np.diag(corr_upper_tri))
    correlations = pd.DataFrame(correlations, index=asset_properties.keys(), columns=asset_properties.keys())

    # make random asset weights for X simulations
    simulations = 100000
    weights = numpy.random.rand(len(asset_properties.keys()), simulations)  # columns are simulations, rows are assets
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    weights_df = pd.DataFrame(weights, index=asset_properties.keys())

    # use list comprehension to get tuple of resuts as [ (risk_1, return_1, sharpe_1), (risk_2, return_2, sharpe_2), ...
    combo_portfolio_risk_return_pairs = [calc_portfolio_risk_return(weights_df[index]) for index in weights_df.columns]
    stddevs = [pair[0] for pair in combo_portfolio_risk_return_pairs]
    avgs = [pair[1] for pair in combo_portfolio_risk_return_pairs]
    sharpe_ratio = [pair[2] for pair in combo_portfolio_risk_return_pairs]

    # scatter plot
    fig = plt.figure()
    plot1 = plt.subplot2grid((10, 10), (0, 0), rowspan=10, colspan=9)
    plot1.scatter(stddevs, avgs, c=sharpe_ratio, cmap="viridis", s=5)
    plot1.set_title("Efficient Frontier")
    plot1.set_xlabel("Risk")
    plot1.set_ylabel("Return")

    # color map
    plot2 = plt.subplot2grid((10, 10), (0, 9), rowspan=10, colspan=1)
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(sharpe_ratio), vmax=max(sharpe_ratio))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=plot2, orientation='vertical', label='Sharpe Ratio')
    plt.show()

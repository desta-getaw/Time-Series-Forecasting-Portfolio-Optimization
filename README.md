# Time Series Forecasting for Portfolio Management Optimization

## 📈 Business Objective

This project is undertaken for **Guide Me in Finance (GMF) Investments**, a financial advisory firm dedicated to using data-driven insights for personalized portfolio management. The primary goal is to leverage time series forecasting to predict market trends, optimize asset allocation, and ultimately enhance client portfolio performance by minimizing risks and maximizing opportunities.

As a Financial Analyst at GMF, you are tasked with analyzing historical financial data, building predictive models, and recommending portfolio adjustments based on forecasted trends.

***

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Setup and Installation](#-setup-and-installation)
* [Step-by-Step Guide](#-step-by-step-guide)
    * [Task 1: Data Preprocessing and Exploration](#task-1-data-preprocessing-and-exploration)
    * [Task 2: Develop Time Series Forecasting Models](#task-2-develop-time-series-forecasting-models)
    * [Task 3: Forecast Future Market Trends](#task-3-forecast-future-market-trends)
    * [Task 4: Optimize Portfolio Based on Forecast](#task-4-optimize-portfolio-based-on-forecast)
    * [Task 5: Strategy Backtesting](#task-5-strategy-backtesting)
* [Submission Guidelines](#-submission-guidelines)
* [Key Dates](#-key-dates)

***

## Project Overview

This challenge involves a comprehensive workflow: fetching financial data, performing exploratory data analysis, building and comparing statistical and deep learning forecasting models, optimizing a three-asset portfolio using Modern Portfolio Theory (MPT), and validating the strategy through backtesting.

This project recognizes the **Efficient Market Hypothesis**, which suggests that precise stock price prediction is highly challenging. Therefore, our models will serve as a critical component within a broader decision-making framework, focusing on forecasting trends and volatility rather than exact prices.

![Stock market chart with analysis overlays](https://placehold.co/800x400/1a202c/ffffff?text=Financial+Analysis+Dashboard)

***

## 💾 Dataset

We will use historical financial data for three key assets, sourced from **YFinance**.

* **Assets**:
    1.  `TSLA`: A high-growth, high-risk stock from the automobile manufacturing sector.
    2.  `BND`: A Vanguard Total Bond Market ETF that provides stability and income.
    3.  `SPY`: An ETF that tracks the S&P 500 Index, offering broad market exposure.
* **Time Period**: July 1, 2015, to July 31, 2025.
* **Data Fields**: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.

***

## 📂 Project Structure

Organize your project repository with the following structure for clarity and reproducibility:

.├── data/│   ├── raw/│   └── processed/├── notebooks/│   ├── 1_Data_Exploration.ipynb│   ├── 2_Modeling.ipynb│   ├── 3_Portfolio_Optimization.ipynb│   └── 4_Backtesting.ipynb├── reports/│   ├── interim_report.pdf│   └── investment_memo.pdf├── src/│   ├── data_loader.py│   ├── modeling.py│   └── optimization.py├── .gitignore├── requirements.txt└── README.md
***

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file should be created containing the necessary packages.
    ```
    pandas
    numpy
    yfinance
    matplotlib
    seaborn
    statsmodels
    pmdarima
    scikit-learn
    tensorflow
    PyPortfolioOpt
    ```
    Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

***

## 🗺️ Step-by-Step Guide

### Task 1: Data Preprocessing and Exploration

**Objective**: Load, clean, and analyze the data to uncover initial insights.

1.  **Fetch Data**: Use the `yfinance` library to download historical data for `TSLA`, `BND`, and `SPY` for the specified date range.
2.  **Data Cleaning**:
    * Check the basic statistics (`.describe()`) of the datasets.
    * Verify data types and handle any missing values through interpolation or filling.
3.  **Exploratory Data Analysis (EDA)**:
    * Visualize the closing prices over time for each asset.
    * Calculate and plot the daily percentage change (daily returns) to observe volatility.
    * Plot rolling means and standard deviations to analyze short-term trends.
4.  **Stationarity Test**:
    * Perform the Augmented Dickey-Fuller (ADF) test on the closing prices and daily returns to check for stationarity.
    * Discuss the results. Non-stationary series will require differencing for ARIMA models.
5.  **Risk Metrics**:
    * Calculate foundational metrics such as **Value at Risk (VaR)** and the **Sharpe Ratio** to assess historical risk and return.

### Task 2: Develop Time Series Forecasting Models

**Objective**: Build, compare, and evaluate at least two different forecasting models to predict TSLA's stock price.

1.  **Data Splitting**:
    * Split the data chronologically into training and testing sets. **Do not shuffle the data.**
    * Example Split: Train on data from 2015-2023 and test on 2024-2025.
2.  **Model 1: ARIMA/SARIMA (Statistical Model)**:
    * Use `pmdarima.auto_arima` to automatically find the optimal `(p, d, q)` parameters for the model.
    * Train the ARIMA model on the training dataset.
3.  **Model 2: LSTM (Deep Learning Model)**:
    * **Scale the data** (e.g., using `MinMaxScaler`) as neural networks are sensitive to the scale of input data.
    * Create sequences of data (e.g., use the last 60 days to predict the next day).
    * Build, compile, and train the LSTM model. Experiment with different architectures (layers, neurons) and hyperparameters (epochs, batch size).
4.  **Model Evaluation**:
    * Generate forecasts from both models on the test set.
    * Compare their performance using metrics like **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Percentage Error (MAPE)**.
    * Discuss which model performed better and why.

### Task 3: Forecast Future Market Trends

**Objective**: Use your best-performing model from Task 2 to generate a 6-12 month forecast for `TSLA`.

1.  **Generate Forecast**: Use the trained model to predict future stock prices.
2.  **Visualize Forecast**: Plot the historical data alongside the forecasted prices. Include **confidence intervals** to represent the uncertainty in the prediction.
3.  **Interpret Results**:
    * **Trend Analysis**: Identify the overall forecasted trend (upward, downward, or stable).
    * **Risk Analysis**: Discuss the width of the confidence intervals. Wider intervals imply greater uncertainty, especially for long-term forecasts.
    * **Opportunities & Risks**: Based on the forecast, outline potential market opportunities and risks for TSLA.

### Task 4: Optimize Portfolio Based on Forecast

**Objective**: Construct an optimal portfolio using the principles of Modern Portfolio Theory (MPT).

1.  **Define Expected Returns**:
    * For **TSLA**, use the return forecast from your best model.
    * For **BND** and **SPY**, use their historical average daily returns (annualized).
2.  **Calculate Covariance Matrix**: Compute the historical covariance matrix for all three assets to understand how they move in relation to each other.
3.  **Generate Efficient Frontier**:
    * Using the expected returns and the covariance matrix, run an optimization to generate the Efficient Frontier with a library like `PyPortfolioOpt`.
    * Plot the frontier with risk (volatility) on the x-axis and return on the y-axis.
4.  **Identify Key Portfolios**:
    * On the plot, mark the **Maximum Sharpe Ratio Portfolio** and the **Minimum Volatility Portfolio**.
5.  **Recommend a Portfolio**:
    * Select an optimal portfolio and justify your choice based on risk-return objectives.
    * Summarize the final recommended weights for TSLA, BND, and SPY, along with the portfolio's expected annual return, volatility, and Sharpe Ratio.

### Task 5: Strategy Backtesting

**Objective**: Validate your proposed portfolio strategy by simulating its performance on historical data and comparing it against a benchmark.

1.  **Define Backtesting Period**: Use the last year of your data (e.g., August 1, 2024 - July 31, 2025).
2.  **Define Benchmark**: Use a static **60% SPY / 40% BND** portfolio as a simple benchmark.
3.  **Simulate Strategy**:
    * Start with the optimal weights determined in Task 4.
    * Simulate holding this portfolio over the backtesting period. For simplicity, you can hold the initial weights for the full year without rebalancing.
4.  **Analyze Performance**:
    * Plot the cumulative returns of your strategy against the benchmark.
    * Calculate the final total return and Sharpe Ratio for both portfolios.
    * Conclude with a summary of whether your strategy outperformed the benchmark and what this suggests about its viability.

***

## 🏆 Submission Guidelines

### Interim Submission

* **Deadline**: 20:00 UTC on Sunday, 10 Aug 2025.
* **Requirements**:
    * An interim report covering all of **Task 1**.
    * A link to your GitHub repository with the code for Task 1.

### Final Submission

* **Deadline**: 20:00 UTC on Tuesday, 12 Aug 2025.
* **Requirements**:
    * A professional **Investment Memo** (PDF) or a detailed technical blog post that presents your methodology, findings, and final recommendation.
    * A link to your completed GitHub repository. Ensure the code is clean, well-commented, and the repository is well-organized.

***

## 🗓️ Key Dates

* **Discussion on the case**: Wednesday, 06 Aug 2025.
* **Interim Submission**: Sunday, 10 Aug 2025, 20:00 UTC.
* **Final Submission**: Tuesday, 12 Aug 2025, 20:00 UTC.

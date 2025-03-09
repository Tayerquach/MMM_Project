Train a Time-Series Forecasting Model for Ad Spend vs. Conversions
========================

# [![Python][Python.py]][Python-url] [![Jupyter][jupyterlab.ipynb]][jupyterlab-url]

## 💡 Use Case: Predict how future ad spend will impact Sales using historical data.
- To understand the problem, please refer to `problem.md`.
- This repository contains the source code for the entire project with step-by-step instructions as follows:
    * [Environment Setup](#environment-setup)
    * [Simulated Data](#simulated-data)
    * [Run Preprocessing and EDA](#preprocessing)
    * [Reproduce Results](#reproduce-results)

## Environment Setup

1. Install Python (<a target="_blank" href="https://wiki.python.org/moin/BeginnersGuide">Setup instruction</a>).
   **Note**: In this project, we used Python 3.11.4
2. Install Conda (<a target="_blank" href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html">Conda Installation</a>) or similar environment systems
3. Create a virtual enviroment
```console 
conda create --name [name of env] python==[version]
```
For example,
```console 
conda create --name mmm python==3.11.4
```
4. Activate enviroment
```console 
conda activate [name of env]
``` 

## Dowload the project (git)
Clone the project
```consolde
git clone https://github.com/Tayerquach/MMM_Project.git
```

After cloning repository github, going to the `MMM_PROJECT` folder and do the steps as follows

**Install Python packages**
```console 
pip3 install -r requirements.txt 
```

## Simulated Data
We apply a basic media mix modelling structure to generate the data in our case study.

- **Outcome** = *Newly sale each week*

- **Predictors** = *weekly marketing spend in each marketing channel, transformed*
  - *nonlinear* functional transformation to account for diminishing returns on spend
  - *adstock* transformation of channel spend to account for lagged effects of advertising

- **Control variables** = *seasonal, discounting tactics, etc.*

Output is in `data` folder under `raw_three_year_data.csv` which contains **three years** of weekly data with **three channels** from different parts of the marketing funnel. Each channel has a different adstock, saturation and contribution. This dataset contains **weekly marketing, demand, and sales data** for **EggBuddy Cafe**, used for **Marketing Mix Modeling (MMM)** to analyze the impact of advertising spend on demand and revenue.

### **Schema Structure**

| Column Name            | Data Type | Description |
|------------------------|-----------|-------------|
| `date`                | `DATE`    | The week-ending date (typically a Sunday), representing the time period for the recorded data. |
| `demand`              | `FLOAT`   | The actual observed demand for EggBuddy Cafe, measured in orders placed. |
| `demand_proxy`        | `FLOAT`   | A proxy measure for demand, possibly based on online engagement, foot traffic estimations, or search trends. |
| `tv_ad_spend_raw`     | `FLOAT`   | The raw amount spent on **TV advertisements** for that week, measured in currency (e.g., dollars). |
| `social_ad_spend_raw` | `FLOAT`   | The raw amount spent on **social media ads** (e.g., Facebook, Instagram, TikTok) promoting EggBuddy Cafe. |
| `search_ad_spend_raw` | `FLOAT`   | The raw amount spent on **search engine ads** (e.g., Google Ads) to attract potential customers. |
| `sales`               | `FLOAT`   | The **total revenue generated** by EggBuddy Cafe in that week, reflecting the effectiveness of marketing efforts. |

## Run Preprocessing and EDA


















<!-- MARKDOWN LINKS & IMAGES -->
[Python.py]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

[jupyterlab.ipynb]: https://shields.io/badge/JupyterLab-Try%20GraphScope%20Now!-F37626?logo=jupyter
[jupyterlab-url]: https://justinbois.github.io/bootcamp/2020_fsri/lessons/l01_welcome.html
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

def sales_behavior_holidays(train_data):
    # Sales behavior before, during, and after holidays
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data['IsHoliday'] = np.where(train_data['StateHoliday'] != '0', 1, 0)

    holiday_sales = train_data.groupby('IsHoliday')['Sales'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=holiday_sales, x='IsHoliday', y='Sales')
    plt.title('Average Sales: Holiday vs Non-Holiday')
    plt.xlabel('Is Holiday')
    plt.ylabel('Average Sales')
    plt.show()
    logging.info('Analyzed sales behavior during holidays.')

def seasonal_purchase_behaviors(train_data):
    # Seasonal behavior (Christmas, Easter, etc.)
    train_data['Month'] = train_data['Date'].dt.month
    seasonal_sales = train_data.groupby('Month')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=seasonal_sales, x='Month', y='Sales')
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.show()
    logging.info('Analyzed seasonal purchase behaviors.')

def correlation_sales_customers(train_data):
    # Correlation between sales and number of customers
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=train_data, x='Customers', y='Sales')
    plt.title('Sales vs. Number of Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

    correlation = train_data['Sales'].corr(train_data['Customers'])
    print(f'Correlation between Sales and Customers: {correlation}')
    logging.info(f'Calculated correlation between sales and number of customers: {correlation}')

def impact_of_promotions(train_data):
    # Effect of promos on sales
    promo_sales = train_data.groupby('Promo')['Sales'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=promo_sales, x='Promo', y='Sales')
    plt.title('Average Sales with and without Promo')
    plt.xlabel('Promo')
    plt.ylabel('Average Sales')
    plt.show()
    logging.info('Analyzed the impact of promotions on sales.')
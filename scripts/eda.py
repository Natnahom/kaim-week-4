import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='eda_log.txt', level=logging.INFO)

def distribution_of_promotions(train_data, store_data):
    # Distribution of promotions in training and test sets
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_data, x='Promo')
    plt.title('Distribution of Promotions in Training Set')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.countplot(data=store_data, x='Promo2')
    plt.title('Distribution of Promotions in Test Set')
    plt.show()
    logging.info('Displayed distribution of promotions in training and test sets.')
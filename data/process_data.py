import sys

import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    reads data from csv files and merges them into a single dataframe based on id
    :param messages_filepath: string
    :param categories_filepath: string
    :return: dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='outer', on='id')


def clean_data(df):
    """
    cleans the category column, removes duplicates
    :param df: dataframe
    :return: dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').str[0]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # iterate through the category columns in df to keep only the last
    # character of each string (the 1 or 0)
    for column in categories:
        #     set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # removing '2' from 'related' column
    df.drop(df[df['related'] == 2].index, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    saves dataframe to sqlite database
    :param df: dataframe
    :param database_filename: string
    :return: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('clean_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

# Disaster Response Pipeline Project

In this project, we will analyze disaster data from 
[Figure Eight](https://appen.com/) to build a model for an API that classifies 
disaster messages.
In the data folder, there are csv files containing real messages that were sent 
during disaster events. We will be creating a machine learning pipeline to 
categorize these events so that messages can be delivered to an appropriate 
disaster relief agency.

This project includes a web app where an emergency worker can input a new 
message and get classification results in several categories. 
The web app will also display visualizations of the data. 

## Table of Contents
1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Components](#project-components)
4. [Project Structure](#project-structure)
5. [Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)

## Installation
### Dependencies
* Python >= 3.7
* matplotlib >= 3.4.2
* numpy >= 1.21.1
* pandas >= 1.3.1
* Flask >= 2.0.1

### User Installation
Assuming you already have Python>3.7 up and running, the easiest way to install 
the requirements is using **pip** or **conda**. This tutorial assumes using _pip_
for the installation purposes.

Create a virtual environment to set up the project. (Optional Step)

``
python3 -m venv <my_env>
``

cd to the root directory of the code and install the requirements using

``
pip install -r requirements.txt
``

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Components
There are three components of this project.

### ETL Pipeline
The file `process_data.py` contains a data cleaning pipeline that:

* Loads the messages and categories datasets
*Merges the two datasets
* Cleans the data
* Stores it in a SQLite database 

### ML Pipeline
The Python script, `train_classifier.py`, contains machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file 

### Flask Web App

A basic flask app that displays visualizations of data as well as prints out 
results for a message entered by the user.

### Graphs Folder

This folder contains screenshots of the data visualizations as appeared on the 
index of the web app.

## Project Structure

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- graphs
|- msg_category.png
|- Msg_genre.png
|- news_msg_category.png

- README.md

```

## Licensing, Authors, and Acknowledgements
All credits to Figure Eight for making the data available to Udacity. Many 
thanks to udacity for this project. Feel free to use all or any part of project
citing me, udacity and/or Figure Eight accordingly.

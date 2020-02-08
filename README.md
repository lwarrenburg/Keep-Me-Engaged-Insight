# Keep-Me-Engaged
Finding Customized Recommendations to Improve e-Learner Engagement

## Table of Contents

- [Project Description](#description)
- [Data Process](#process)
- [Web App Tool](#tool)
- [Modeling User Engagement](#modeling)
- [Results](#results)
- [Contact](#contact)

___
## Description
### Summary
For my Insight Data Science project, I consulted with `Thought Industries`, a software company that provides an easy-to-use **online learning platform** for over 7,000 companies of all sizes. Although millions of users across the world have used e-Learning platforms, many learners drop out or do not complete the course. An important way to increase user retention is to ensure that the e-Learning sites are engaging.

Because Thought Industries hosts over 200,000 online courses, it would be difficult for a team of people to provide feedback on current website engagement and suggest recommendations to increase learner engagement. I created a tool, called `Keep Me Engaged`, that uses machine learning to provide customized recommendations to increase user engagement for over 30,000 of Thought Industries' online courses.

To create Keep-Me-Engaged, I built a series of Jupyter notebooks to summarize 1.6 million data points of _administrative, learner, instructor,_ and _website features_ to model learner engagement. The web app is expected to be implemented as part of Thought Industries’ Premium Services in the future.

### Who are the users of Keep-Me-Engaged?
This tool is designed to help **course developers and instructors**. These users will learn how to improve engagement levels for future iterations of their course (like the next semester or the next on-boarding session). Keep-Me-Engaged is _not_ to help predict engagement levels for a brand new course.

### GitHib files
In this GitHub repository, you will find the following information:
- **Jupyter Notebooks to describe the feature engineering and modeling processes**
  - Thought Industries--Feature Engineering.ipynb
  - Thought Industries--Modeling.ipynb
  - _Note: The Data Generation script is not posted for privacy reasons_
  
- **Files to run and deploy the Keep-Me-Engaged web app**
  - keep-me-engaged-streamlit.py (Python file using Streamlit to create the app)
  - requirements.txt (lists the required files for the app to be deployed on Heroku)
  - Procfile (required for the app to be deployed on Heroku)
  
- **Deidentified data files used by the Keep-Me-Engaged Python script**
  - X.csv (CSV of all the predictor variables for the machine learning model)
  - y.csv (CSV of the dependent variable: Engagement Scores)
  - featureimportance.csv (CSV listing the relative variable importances for the model)
  - fifteen.csv (subset of X where each company has 15 or less courses -- makes app more intuitive)
  - finalizedmodel.sav (pickle file of a Random Forest regression, used by Heroku to deploy the web app)
 
- **Descriptions of the Insight Project**
  - Presentation folder (includes slides for a 5-min demo of the project)
  - Images folder (includes pictures of the web app for Keep-Me-Engaged and of the feature importances)

___
## Process

### Overview
  1.  Access & Download Data (AWS Redshift, PostgreSQL)
  2.  Feature Engineering & Model Selection (Python, pandas, scikit-learn)
  3.  Web App Creation & Deployment (Streamlit, Heroku)
 
### Operationlizing "Engagement"
Although there are many ways of defining leaner engagement, I decided to operationalize engagement in the following way:
 > _**Engagement:** The percent of users that completed the entire course_
  
  For example, an Engagement Score of 80 means that--of all of the users that started the course--80% of learners completed the course.

### Feature Selection
There are four main kinds of features in Thought Industries' data that could affect Learner Engagement.
  1.  _**Administrative Features**_ (examples: Price of Course, Length/Duration of Course)
  2.  _**Learner Features**_ (examples: Average Quiz Grade, Time Spent on Course)
  3.  _**Instructor Features**_ (examples: percent of non-graded quizzes, number of collaborations)
  4.  _**Website Features**_ (examples: percent of interactive pages, percent of pages that include videos)

All of these features were used in my model, but only the latter two (_Instructor_ and _Website_ Features) are present on the web app. Since the app is designed for course developers and instructors, the app focuses on features that they can _control themselves_.

___
## Tool
[Keep-Me-Engaged App!](https://keep-me-engaged.herokuapp.com/)

Keep-Me-Engaged is an online tool for **e-Learning course developers** to find customized recommendations on how to increase user engagement in their online courses. Features of the Keep-Me-Engaged include the following:

#### Selecting a course and company
- Select your **company name** from a drop-down list 
  _Company names have been replaced with random words from the dictionary to ensure confidentiality_
- Select a **Course ID #** from the list of courses offered by your company

#### Compare Engagement Scores across the courses offered by your company
- View a bar chart plotting the Engagement Scores for all company courses
- Selected tool is highlighted in a different color
- Hover tool allows users to immediately see the Engagement Scores and number of users in each course
- Prominent display of the _Current Engagement Score_ for the selected course

#### Discover the most important features of Learner Engagement
- Blog-style text that summarizes feature importances across 30,000 Thought Industries courses
- Table with feature name, rank, and descriptions
- Option of displaying the scores of these features for the selected course and course averages 

#### Use interactive slider bars to see how your Engagement Scores can change in the future
- Seven sliders where the user can change the value of a feature (like "Percent of Non-Graded Assessments")
- Button to calculate the _Updated Engagement Score_ to see how the changes made with the slider bars affect User Engagement.

![image](/images/sample_graph.png)

_Sample picture of the bar chart with the Hover Tool active_
___
## Modeling
### 1. Data Generation
**Using AWS Redshift & PostgreSQL:**
  - Choose variables using top-down and bottom-up methods
  - Access specific variables
  - Examine their distributions
  - Transform the variables appropriately
  - Merge them into a single data frame
  - Export that dataframe to a CSV file. 

_To ensure the confidentiality of this process and protect company information, this Jupyter notebook is not publicly available on my GitHub._

### 2. Feature Engineering
**Using Python, pandas, scikit-learn:**
  - Read in data frame exported from the Data Generation script
  - Continue visualizing variable distributions and correlations
      _Delete or modify variables according to these results_
  - Constrain scope of the problem 
      _Focus on Engagement Scores from 1-99%)_
  - Examine and transforming data according to missing values
      _Delete some observations (like observations with a missing Engagement Score)_
      _Replace some missing values with 0 (where a missing value means 0)_
      _Transform related columns with many missing values into a single column with binary values (it does or does not contain this feature)_
      _impute with median where appropriate_
  - Check for outliers & apply log(1+x)

### 3. Modeling
**Using Python, pandas, scikit-learn:**
  - Train/test split (80/20)
  - Random Forest Regression
    - Grid search & hyperparameter tuning
    - Compare _base_ and _tuned_ Random Forest models
    - Evaluate & plot feature importances
    - Create Shapley plots using Shapley values
  - Linear Regression (logged and not-logged predictors)
  - Regularized (Ridge) Regression (logged and not-logged predictors)

___
## Results
### Model Evaluation
- Random Forest Regression with hyperparameter tuning: _**0.62**_
- Linear Regression with logged variables: _**0.16**_
- Ridge Regression with logged variables: _**0.16**_

### Feature Importance
_Top seven features:_
  1.  Length/Duration of Course
  2.  Number of Users
  3.  Company Rating
  4.  Percent of Content Pages
  5.  Price (dollars)
  6.  Percent of Non-Graded Assessments
  7.  Percent of Learners who Earn Certificates

### Examples of Interesting Findings for Instructors
This table shows whether _**increasing**_ or _**decreasing**_ the following features result in **_higher_** engagement scores.

Increasing | Decreasing
:-------------: | :-------------:
Certificates | Content Pages
Non-graded Assessments |  
Teacher Comments |  
Interactive Pages |  

![image](/images/feature_importances.png)

_Feature importances from the Random Forest model_

___
## Contact
Feel free to reach out in the following ways!
- Personal website: <a href="https://www.lindsaywarrenburg.com/" target="_blank">`https://www.lindsaywarrenburg.com/`</a>
- LinkedIn: <a href="https://www.linkedin.com/in/lindsay-a-warrenburg/" target="_blank">`https://www.linkedin.com/in/lindsay-a-warrenburg/`</a>


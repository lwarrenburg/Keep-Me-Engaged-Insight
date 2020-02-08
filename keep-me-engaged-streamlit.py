import streamlit as st
import pandas as pd
from scipy.stats import beta
import numpy as np
import pickle

import matplotlib.pyplot as plt
import holoviews as hv
import hvplot
import hvplot.pandas # noqa: F401
from holoviews import opts
hv.extension('bokeh', logo=False)

import bokeh.models as bmo
from bokeh.plotting import figure, show
from bokeh.palettes import PuBu, Spectral5, Spectral6
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, CategoricalColorMapper
from bokeh.transform import factor_cmap, factor_mark

import sklearn
from sklearn import preprocessing, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report


##############################################################################

#### Read in data from streamlit folder ####

##############################################################################

data = pd.read_csv('fifteen.csv')
data = data[np.isfinite(data['Engagement Score'])]

featureimportance = pd.read_csv('featureimportance.csv')

##############################################################################

#### Writing title and intro ####

##############################################################################

st.title("Keep-Me-Engaged")

st.markdown("A tool for **e-Learning course developers** to find customized recommendations on how to increase user engagement!")

st.header("Want to increase user engagement in your online courses?")

st.markdown("""
Learner engagement is something to strive for. Although engagement can mean different things to different people, we consider _**engagement**_ to be **the percent of learners that complete an e-Learning course.**

To use this app, please select your _company_ and _Course ID #_ from the sidebar on the left side of the page. The following features are available on this tool:""")

st.write("""
1. Learn how your course engagement score compares to other courses offered by your company. 
2. Discover the most important features that you can influence to increase learner engagement.
3. Use interactive slider bars to see how your engagement score can change in the future.
""")


##############################################################################

#### Sidebars ####

##############################################################################

st.sidebar.markdown(
"""
# Control Panel
"""
)

# pick company
company_info = data['companyid_words'].unique()
company_selection = st.sidebar.selectbox(
	'Which company would you like to select?',
	(company_info)
)

# pick course number
subset = data[data['companyid_words'] == company_selection]


course_info = subset['courseid_num'].unique()
course_selection = st.sidebar.selectbox(
	'Which course would you like to select?',
	(course_info)
)

subset3 = subset.drop('companyid_words', axis=1)
subset3["Engagement Score"] = pd.to_numeric(subset3["Engagement Score"])

##############################################################################

#### Add text about selected courses/company ####

##############################################################################

st.subheader("Comparing engagement scores across courses of the selected company")

st.write("_Course ID #**", str(course_selection), "**is highlighted. Use the hover tool to see the engagement score and number of users for each course._")

subset3["Engagement Score"] = subset3["Engagement Score"].round()
subset3 = subset3.sort_values(by ='Engagement Score', ascending=False)

subset3 = subset3.reset_index(drop=True)
subset3["Course Currently Selected"] = "No"

rowLoc = subset3[subset3['courseid_num']==course_selection].index.values.astype(int)[0]
subset3["Course Currently Selected"][rowLoc] = "Yes"


##############################################################################

#### Plot selected course along with all other courses from the selected company ####

##############################################################################

subset3["Course ID #"] = subset3["courseid_num"]

plot_opts = dict(show_legend=False, color_index='Course Currently Selected', title="Engagement Scores Across Your Company Courses", width=600, xlabel='Course ID #', 
ylabel='Engagement Score', ylim=(0, 110), xrotation=45, tools=['hover'])

style_opts = dict(box_color=hv.Cycle(['#30a2da', '#fc4f30']))

bars = hv.Bars(subset3, hv.Dimension('Course ID #'), ['Engagement Score', 'Course Currently Selected','Number of Users'])

bars = bars.opts(plot=plot_opts, style=style_opts)

st.write(hv.render(bars))

##############################################################################

#### Writing current engagement score ####

##############################################################################

subset2 = subset[subset['courseid_num'] == course_selection]
subset2.reset_index(drop=True, inplace=True)

engagementscore = subset2['Engagement Score']
engagementscore = pd.DataFrame(engagementscore)
engagementscore.columns = ["Your Current Engagement Score:"]

st.write("The **current** engagement score for Course ID #**", str(course_selection), "** is:")
st.markdown('<span style="color:#30a2da; font-size:42px">' + str(engagementscore.iloc[0,0]) + '</span>', unsafe_allow_html=True)

##############################################################################

#### Discuss which features are important for user engagement ####

##############################################################################

st.header("""
What makes for an engaging e-Learning course?
""")

st.markdown("""
At Thought Industries, we have found that there are some aspects of online courses that foster **_engagement_** among all learners. Here, you can see the seven most important features that you can influence to increase completion rates (ranked from most important to least important). The desciptions of the features correspond to widgets and website features used across Thought Industries' e-Learning platforms.
""")

##############################################################################

#### Select the top 7 features that lead to user engagement (that can be controlled by instructors) #### 

##############################################################################

test = featureimportance[featureimportance['controllable'].isin(['Instructor Features', 'Website Features'])]

test = test.reset_index(drop = True)

topfeatures = test.iloc[0:7,0]

##############################################################################

#### Summarize this information in a dataframe; add descriptions of the features ####

##############################################################################

explain = pd.DataFrame()

explain['Feature'] = ['Content Pages', 'Not-Graded Assessments', 'Certificates', 'Interactive Pages', 'Graded Assessments', 'External Pages', 'Teacher Comments']

explain['Importance']=[1,2,3,4,5,6,7]

explain['Description']=['Text, slideshows, presentations, videos, list rolls, PDF viewers, ads, recipes, audio files, and articles', 'Surveys, tallies, and workbooks', 'Certificates given after course completion', 'Assignments, flip card sets, notebooks, highlight zone sets, highlight zone quizzes, match pair sets, discussion boards, social share card sets, and images', 'Tests and quizzes', 'LTI, shareable content objects, API objects, survey gizmos, embedded features, in-person events, and meeting information', 'Number of times the instructor comments on an assignment']

explain = explain.set_index('Importance')

st.table(explain)

##############################################################################

#### Find out descriptive measures of the variables across ALL courses ####

##############################################################################

feat1loc = data.columns.get_loc(topfeatures[0])
feat2loc = data.columns.get_loc(topfeatures[1])
feat3loc = data.columns.get_loc(topfeatures[2])
feat4loc = data.columns.get_loc(topfeatures[3])
feat5loc = data.columns.get_loc(topfeatures[4])
feat6loc = data.columns.get_loc(topfeatures[5])
feat7loc = data.columns.get_loc(topfeatures[6])

avgs = data.iloc[:,[feat1loc, feat2loc, feat3loc, feat4loc, feat5loc, feat6loc, feat7loc]]
avgs.fillna(avgs.mean())

avg1 = avgs.iloc[:,0].mean().round()
avg2 = avgs.iloc[:,1].mean().round()
avg3 = avgs.iloc[:,2].mean().round()
avg4 = avgs.iloc[:,3].mean().round()
avg5 = avgs.iloc[:,4].mean().round()
avg6 = avgs.iloc[:,5].mean().round()
avg7 = avgs.iloc[:,6].mean().round()

min1 = avgs.iloc[:,0].min().round()
min2 = avgs.iloc[:,1].min().round()
min3 = avgs.iloc[:,2].min().round()
min4 = avgs.iloc[:,3].min().round()
min5 = avgs.iloc[:,4].min().round()
min6 = avgs.iloc[:,5].min().round()
min7 = avgs.iloc[:,6].min().round()

max1 = avgs.iloc[:,0].max().round()
max2 = avgs.iloc[:,1].max().round()
max3 = avgs.iloc[:,2].max().round()
max4 = avgs.iloc[:,3].max().round()
max5 = avgs.iloc[:,4].max().round()
max6 = avgs.iloc[:,5].max().round()
max7 = avgs.iloc[:,6].max().round()

##############################################################################

#### Examining the selected course's feature scores ####

##############################################################################

st.subheader("Click the button below to see how the selected course scores on these seven features compared to other Thought Industries courses:")
st.markdown("<br>", unsafe_allow_html = True)

new = subset2.T
new.reset_index(level=0, inplace=True)
new = new[new['index'].isin(featureimportance['colname'])]
new = new.rename(columns={"index": "Feature", 0: "Your Score"})

new['Your Score'] = pd.to_numeric(new['Your Score'])
new['Your Score'] = new['Your Score'].round()
new = new.T

new.columns = new.iloc[0]

feat1 = new.columns.get_loc(topfeatures[0])
feat2 = new.columns.get_loc(topfeatures[1])
feat3 = new.columns.get_loc(topfeatures[2])
feat4 = new.columns.get_loc(topfeatures[3])
feat5 = new.columns.get_loc(topfeatures[4])
feat6 = new.columns.get_loc(topfeatures[5])
feat7 = new.columns.get_loc(topfeatures[6])

scores = new.iloc[1:,[feat1, feat2, feat3, feat4, feat5, feat6, feat7]]

avgscores = scores.append(pd.Series([avg1, avg2, avg3, avg4, avg5, avg6, avg7], index=scores.columns), ignore_index=True)

avgscores = avgscores.set_index([pd.Index(['Selected Course', 'All Courses'])])

##############################################################################

#### Make button so they can see the comparison between their features and the average across all Thought Industries' courses if they want ####

##############################################################################

if st.button('Let me see!'):
	st.table(avgscores)

new = new.drop(new.index[0])

##############################################################################

#### What if we change the important features? ####

##############################################################################

st.header("""
Change engagement score by changing features of your online course!
""")

st.markdown("""
_Use the sliders below to see how your engagement score will change if you change the following features of your course._ 

_When you are finished using the slider bars, you can click the button below to see your **updated** engagement score:_
""")

feat1slider = st.slider(
	topfeatures[0],
	min_value=0,
	max_value=100)

feat2slider = st.slider(
	topfeatures[1],
	min_value=0,
	max_value=100)

feat3slider = st.slider(
	topfeatures[2],
	min_value=0,
	max_value=100)

feat4slider = st.slider(
	topfeatures[3],
	min_value=0,
	max_value=100)

feat5slider = st.slider(
	topfeatures[4],
	min_value=0,
	max_value=100)

feat6slider = st.slider(
	topfeatures[5],
	min_value=0,
	max_value=100)

feat7slider = st.slider(
	topfeatures[6],
	min_value=0,
	max_value=10)

##############################################################################

#### Random Forest to predict new engagement score ####

##############################################################################

# load in RF model 
loadedmodel = pickle.load(open('finalizedmodel.sav', 'rb'))

# read in data
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

# get user input from sliders
newprediction = pd.DataFrame(new)

locationFeat1 = newprediction.columns.get_loc(topfeatures[0])
locationFeat2 = newprediction.columns.get_loc(topfeatures[1])
locationFeat3 = newprediction.columns.get_loc(topfeatures[2])
locationFeat4 = newprediction.columns.get_loc(topfeatures[3])
locationFeat5 = newprediction.columns.get_loc(topfeatures[4])
locationFeat6 = newprediction.columns.get_loc(topfeatures[5])
locationFeat7 = newprediction.columns.get_loc(topfeatures[6])

newprediction.iloc[:,locationFeat1] = feat1slider
newprediction.iloc[:,locationFeat2] = feat2slider
newprediction.iloc[:,locationFeat3] = feat3slider
newprediction.iloc[:,locationFeat4] = feat4slider
newprediction.iloc[:,locationFeat5] = feat5slider
newprediction.iloc[:,locationFeat6] = feat6slider
newprediction.iloc[:,locationFeat7] = feat7slider

# fill in blanks with mean across all companies
newprediction = newprediction.fillna(X.mean())

# predict new engagement score
predengagement = loadedmodel.predict(newprediction)
predengagement = predengagement.round(1)
predengagement = pd.DataFrame(predengagement)
predengagement.columns = ["Updated Engagement Score:"]

##############################################################################

#### Writing updated engagement score ####

##############################################################################

if st.button('Calculate Updated Engagement Score'):
	st.write("The **updated** engagement score for Course ID #**", str(course_selection), "** is:")
	st.markdown('<span style="color:#30a2da; font-size:42px">' + str(predengagement.iloc[0,0]) + '</span>', unsafe_allow_html=True)


##############################################################################

#### Contact information ####

##############################################################################

st.sidebar.markdown(
"""
Created by Lindsay Warrenburg, Data Science Fellow at Insight in Boston, MA.
"""
)

st.sidebar.markdown(
"""
"""
)

st.sidebar.markdown(
"""
_If you have more questions about course development on Thought Industries platforms, please contact 
**Samantha Wickman** 
at **samantha.wickman@thoughtindustries.com**._
"""
)





# Master-Project-2

## Drug Recommendation System Based on Sentiment Analysis Using Machine Learning

### ABSTRACT
This project uses the Drug Review Dataset to understand user reviews regarding multiple drugs for different conditions. These reviews are in the form of numeric ratings and textual reviews. The textual reviews are analyzed to predict the polarity of its sentiment and classified into one of 2 classifications. Two classification models have been tested: LGBM and Random Forest. The highest accuracy is 90.8%, given by the Random Forest. The numeric rating is used to recommend the highly-rated drugs for a given condition to the user.

### DATASET
https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018

### METHODOLOGY
Initially the data has been cleaned of null and empty values (preferred over imputation as imputing reviews desensitizes the aim of the study) and then sorted based on the user ratings. We have extracted the unique conditions to help base the recommendation system here along with extracting the top 10 user preferred drugs (based on highest, here 10 pointer).

Textblob, a python library for NLP technique, is used to give a sentiment polarity and subjectivity of the review. TextBlob returns polarity and subjectivity of the given user review in sentence. The polarity lies between [-1,1], -1 defines a negative sentiment and 1 defines a positive sentiment.The subjectivity lies between [0,1], 0 defines a objective sentiment and 1 defines a subjective sentiment Negation words reverse the polarity. TextBlob has semantic labels that help with fine-tuned analysis. The correlation matrix for the cleaned and uncleaned reviews shows that the removal of stopwords and snowball stemmers are impacting the review to be having a completely different sentiment and hence cleaning is done without the stopwords removal.

The correlation matrix plotted as a heat map, shows the linear dependence of each feature with all the other features in the dataset.

The final step in preprocessing is the Label encoding of the drug name and the condition into numeric values to help aid in the machine learning of this data. We have used label encoding albeit with the one disadvantage that cannot be avoided (also a drawback of this study) is that Label Encoding classifies the data into numbers, and this causes an interpretation of the numbers to be ranked. One hot encoding solves this issue in a general case but cannot be implemented here as there are 3600+ unique values indicating that the dataset will increase with as many dimensions and these created dummy variables create a trap to multicollinearity (dependence between independent variables).

Machine learning models such as LGBM, Random Forest that have been trained with massive amounts of data provided in the dataset helped us to investigate the viability of using machine learning to categorize user ratings based on their textual review in order to discover areas of contingency in this project.

The models are trained to predict the target output,'Review Sentiment'.

### RESULTS
LGBM has accuracy 90.6% with a TP of 28615. We can calculate the TN, FN, FP accordingly.

Random Forest with Random Feature Elimination has an accuracy of about 90.8% with a higher TP of 28747(more sensitivity for this model).

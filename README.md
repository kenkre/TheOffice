# The Office

I wanted to check if there is anything more than the obvious contributing to the success of the TV show, The Office. For this I used Schrute Dataset and IMDB rating to build and compare different models trained with Natural Language Processing techniques. I used mean squared error to score my models and choose the best one: a Random Forest Regressor with no stopwords, no stemmer or lemmatizer, no limits on features (words), and no ngrams. 

## Exploring The Data

The data Schrute dataset consisted of over 55,000 rows of script for 177 episodes over nine years. The IMDB rating dataset was 7.5 milion rows and nine columns. After joining tables and cleansing data, my working dataset was 51792 rows and 9 columns.  

The following chart shows the average rating of the show by season. Michael Scott, the protagonist of the show was played by Steve Carell. NBC didn't renew Steve Carell's contract after the season 7 and a decline in the ratings on the show is obvious. 

<p align="center">
<img src="Images/popularity.png" width="800" height="275">
<p/>

Here is a chart of Top and Bottom 5 shows. four of the bottom shows were in season 8. 

<p align="center">
<img src="TopBottom5.png" width="800" height="275">
<p/>

This chart shows percent of dialogues spoken by the leading characters per season. 

<p align="center">
<img src="Top_6_Characters_Dialog_Percentage.png" width="800" height="275">
<p/>

Here is a word cloud for the best and worst show. 

<p align="center">
<img src="Images/bestshow_cloud.png" width="800" height="350">
<p/>

<p align="center">
<img src="Images/worstshow_cloud.png" width="800" height="350">
<p/>

Looking at these word clouds, it seems the best show used theme more in the scripts and the worst show used name of characters more often in the script. 

## Modeling

To prep for modeling, I split my data into a train (80%) and test (20%) set so that I would have some data to test my winning model on that it had not seen before. I decided to use mean squared error to score my models.

My tri-grams didn't pick up any of the catch phrases from the show as for the most part every word of the phrase was a stop word. I trained and tested my model without using stopwords and punctuations. 

The base mode score with mean Average Rating was 0.297122. My score with Randomforest Regressor removing stop words and punctuations was 0.288765. Score with Randomforest Regressor and Lasso keeping stop words and punctuations was not any different.  
With this I have concluded that there was any trend or peculiarity in the script that contributed to the success of the show. 

## Next Steps

- Do a grid search to refine my models. 
- Graph the ROC curves of my models to compare their shapes.
- Do a little feature importance and figure out what words help indicate a lie.
- Consider separating the data in different ways:
    - by democrats and republicans to see if there are differences in how they lie and about what.
    - by speakers who are individuals and speakers who are not (facebook posts, bloggers, etc.). Looking at the data it seems those labeled as individuals are statements that are spoken whereas facebook posts and blogs are written. There is likely a difference in words used when speaking vs writing.

## Main Python Libraries

- pandas
- numpy
- matplotlib
- sklearn
  - RandomForestRegressor
  - TfidfVectorizer
  - CountVectorizer
- nltk
  - SnowballStemmer
  - WordNetLemmatizer

## Code Files

- 01_The_Office_Data_prep.ipynb
- 02_The_Office_EDA.ipynb
- 03_TheOfficeEDA_2.ipynb
- 04_The_Office_Model.ipynb
- 05_The_Office_NLP_TFIDF_manual.ipynb
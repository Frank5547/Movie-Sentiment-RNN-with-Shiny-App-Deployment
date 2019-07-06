# Movie-Sentiment-RNN-with-Shiny-App-Deployment
Background Code behind my deployed R Shiny app located at http://thalamus.shinyapps.io/movie_sentiment_rnn_with_shiny_app_deployment containing an LSTM model capable of predicting the sentiment of movie reviews. A couple examples of how this app works can be seen below:

![Negative Review](Shiny_App_Example_1.png)
![Positive Review](Shiny_App_Example_2.png)

The first folder contains the code which I used to actually deploy the R Shiny app in www.shinyapps.io. It contains some slight modifications from the original code that were needed for the Shiny app to work inside shinyapps.io. For instance,the vocabulary_to_int.pkl got renamed to vocabulary_to_int_py_2.pkl since I needed to use protocol 2 instead of 3 inside pickle.dump(). This was due to the fact that Python 2.7 was picked as the default inside shinyapps.io.

The second folder above represents the orginal code that I used to build and test the app locally in my computer. This includes the training of the LSTM model and the testing with a separate test set that yielded an 88% test accuracy. If you want to run this app locally, I suggest that you use the files in this folder. Just run "Movie_Review_Sentiment_Predictions_App.R" inside RStudio and you should be good to go!

Please be advised, it may take a few seconds for the Shiny app to load once you click on the URL above, and, once you type a movie review, to give the first prediction. If anyone notices any errors, please let me know, so I can fix them. It would be much appreciated!

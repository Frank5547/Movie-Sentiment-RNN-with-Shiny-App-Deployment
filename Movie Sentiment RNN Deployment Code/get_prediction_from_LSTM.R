#####################################
# get_prediction_from_LSTM.R
# Inputs
#   - review: A string with a movie review
#   - vocabulary_path: A string containing the path to the Pickle (.pkl) file that has the vocabulary to integer mapping.
#     that was used when training the LSTM model.
#   - best_model_path: A string containing the path to the .pt file containing the training checkpoint of the LSTM model with the lowest validation loss.
# Outputs
#   - prediction: an integer containing the prediction from the LSTM model (1 if positive, 0 if negative)
# Creator: Francisco Javier Carrera Arias
# Date Created: 06/19/2019
####################################
library(reticulate)

get_prediction_from_LSTM <- function(review,
                                     vocabulary_path = "vocabulary_to_int_py_2.pkl",
                                     best_model_path = "Sentiment_Best_Model.pt"){
  # Import the New_prediction Python file using reticulate's source Python
  
  source_python("New_Prediction.py")
  
  # Get the LSTM's model prediction invoking the predict_new_review function
  prediction <- predict_new_review(review, vocabulary_path, best_model_path)
  
  # Return the prediction
  return(prediction[1])
}


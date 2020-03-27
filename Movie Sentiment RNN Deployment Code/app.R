library(shiny)
library(shinythemes)
library(shinyjs)
library(reticulate)
virtualenv_create(envname = "python_environment", python = "python3")
virtualenv_install("python_environment", packages = c("numpy","nltk","torch"))
reticulate::use_virtualenv("python_environment", required = TRUE)
source("get_prediction_from_LSTM.R")

# Define UI for application
ui <- fluidPage(theme = shinytheme("superhero"),
   # Application title
   titlePanel(HTML("<h1><center><font size=6> Movie Review Sentiment Predictor </font></center></h1>")),
   tags$h5(align = "center","Just type your movie/show review below and click the button to get the sentiment prediction.
      If you want to submit a new review, simply type your new review and click the button
      again for a new prediction."),
   
   # Create main text box input for the movie review
   fluidRow(
     column(width = 12, wellPanel(textAreaInput(inputId = "review", label = "Input the text of the movie review:", rows = 10)))
  ),
  fluidRow(
    column(width = 12, align = "center", actionButton("button", "Submit Review for Sentiment Prediction"))
  ),
  fluidRow(
    br(),
    column(width = 12, align = "center", htmlOutput("prediction"))
  )
)

server <- function(input, output) {
  
   pred <- eventReactive(input$button,{
        prediction <- get_prediction_from_LSTM(input$review)
        if(prediction == 0){
          tags$div(
            HTML(paste("This review is ", tags$strong(tags$span(style="color:red", "NEGATIVE!")), sep = ""))
          )
        }
        else {
          tags$div(
            HTML(paste("This review is ", tags$strong(tags$span(style="color:green", "POSITIVE!")), sep = ""))
          )
        }
   })
   
   output$prediction <- renderUI({
     pred()
   })
  
}

# Run the application 
shinyApp(ui = ui, server = server)


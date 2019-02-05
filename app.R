library(shiny)
library(ggplot2)
library(dplyr)
library(caret)
library(mboost)
library(scales)

file_name <- "house_prices.csv"
if (!file.exists(file_name)) {
  unzip(paste0(file_name, ".zip"))
}
house_df <- read.csv(file_name)
set.seed(12345)

house_train_idx <- createDataPartition(house_df$SalePrice, p = 0.75, list = FALSE, times = 1)
house_train_df <- house_df[house_train_idx, ]
house_validation_df <- house_df[-house_train_idx, ]
train_set <- (1:nrow(house_df)) %in% unlist(house_train_idx)

# SalePrice ~ GrLivArea + Neighborhood + BldgType + OverallQual + OverallCond + SaleType + SaleCondition + GarageCars + ExterQual + YearBuilt + YearRemodAdd + LotArea
variables <- list(
  "GrLivArea" = "Gross living area",
  "Neighborhood" = "Neighborhood",
  "BldgType" = "Building type",
  "OverallQual" = "Overall quality",
  "OverallCond" = "Overall condition",
  "SaleType" = "Sale type",
  "SaleCondition" = "Sale condition",
  "GarageCars" = "Garage # of cars",
  "ExterQual" = "Exterior quality",
  "YearBuilt" = "Built year",
  "YearRemodAdd" = "Remodel year",
  "LotArea" = "Lot area",
  "YrSold" = "Year sold"
)

ui <- fluidPage(
   titlePanel("House prices prediction (GLMBoost)"),
   
   sidebarLayout(
      sidebarPanel(
        actionButton("help", "Show help"),
        checkboxGroupInput("selectedParameters",
                           label = h3("Model parameters"),
                           choiceNames = as.character(variables),
                           choiceValues = names(variables)),
         h3("Tuning parameters"),
         sliderInput("mstop", "Number of boosting iterations:",
                    min = 1,
                    max = 30,
                    value = 25,
                    step = 1)
      ),
      mainPanel(
        tabsetPanel(
          tabPanel("Data", plotOutput("chartPlot")),
          tabPanel("Prediction", plotOutput("predictionPlot")),
          tabPanel("Residuals", plotOutput("residualPlot"))
        ),
        conditionalPanel("output.show_model_details",
          h2("Model summary"),
          p("Final model"),
          tableOutput("summaryTable"),
          p("Training performance"),
          tableOutput("trainSummaryTable"),
          p("Validation performance"),
          tableOutput("validationSummaryTable")
        )
      )
   )
)

server <- function(input, output) {
   
   selected_parameters <- debounce(reactive({
     input$selectedParameters
   }), 1000)
   
   tuning_settings <- debounce(reactive({
     expand.grid(
       mstop = c(input$mstop),
       prune = TRUE
       )
   }), 1000)
   
   frm <- reactive({
     paste("SalePrice ~", paste(as.character(selected_parameters()), collapse = " + "))
   })
   
   mdl <- reactive({
     if (length(selected_parameters()) > 0) {
       withProgress(message = "Model is Training", value = 1.0, { 
         set.seed(12345)
         train(as.formula(frm()), 
             house_train_df,
             method = "glmboost",
             tuneGrid = tuning_settings())
       })
     }
   })
   
   pred <- reactive({
     m <- mdl() 
     if (!is.null(m)) {
       predict(m, house_df)
     }
   })
   
   pred_df <- reactive({
     p <- pred()
     if (!is.null(p)) {
       data.frame(GrLivArea = house_df$GrLivArea, SalePrice = p)
     }
   })
   
   valid_pred_df <- reactive({
     m <- mdl() 
     if (!is.null(m)) {
       data.frame(obs = house_validation_df$SalePrice, pred = predict(m, house_validation_df))
     }
   })
   
   output$predictionPlot <- renderPlot({
     p_df <- pred_df()
     if (!is.null(p_df)) {
       ggplot(p_df, aes(x = GrLivArea, y = SalePrice)) +
         geom_point() +
         scale_y_continuous(labels = comma)
     }
   })
   
   output$summaryTable <- renderTable({
     p_df <- pred_df()
     if (!is.null(p_df)) {
       bind_rows(defaultSummary(data.frame(
         obs = house_df$SalePrice,
         pred = p_df$SalePrice
       )))
     }
   })
   
   output$trainSummaryTable <- renderTable({
     m <- mdl()
     if (!is.null(m)) {
       getTrainPerf(m)
     }
   })
   
   output$validationSummaryTable <- renderTable({
     v_df <- valid_pred_df()
     if (!is.null(v_df)) {
       bind_rows(defaultSummary(v_df))
     }
   })
   
   output$residualPlot <- renderPlot({
     p_df <- pred_df()
     if (!is.null(p_df)) {
       ggplot(p_df, aes(x = GrLivArea, y = house_df$SalePrice - SalePrice)) +
         geom_point() +
         scale_y_continuous(labels = comma)       
     }
   })
  
   output$chartPlot <- renderPlot({
     ggplot(house_df, aes(x = GrLivArea, y = SalePrice, color = train_set)) +
       geom_point() +
       scale_y_continuous(labels = comma)
   })
   
   output$show_model_details <- reactive({
     !is.null(mdl())
   })
   
   outputOptions(output, "show_model_details", suspendWhenHidden = FALSE)

   helpModal <- function() {
     modalDialog(
       title = "Help",
       "This application demonstrates how selecting different variables ",
       "and tuning parameters affects the output of a GLMBoost model.",
       h4("Dataset"),
       "The example dataset is the",
       a(
         href = "https://www.kaggle.com/c/house-prices-advanced-regression-techniques",
         "Kaggle House Prices dataset"
       ),
       ", which contains data of the sale price and 80 other parameters for 1460 house sales.",
       h4("Model variables and tuning parameters"),
       "The actual dataset contains 81 variables. Out of this list, ",
       length(variables),
       " variables are selected to be used for demonstration purposes. Selecting a number of ",
       "variables from the left pane builds a model with the formula ",
       code("SalePrice ~ variable1 + variable2 + ..."),
       ". In addition to model variables, a tuning parameter that controls the number of boosting ",
       "rounds applied to the model can be controlled.",
       h4("Assessing model performance"),
       "Once a number of variables has been chosen, a GLMBoost model is automatically trained on",
       "a 75% split of the dataset. 25% of the data is set aside for validation. When the training",
       "is complete, a performance metric (RMSE, R",
       tags$sup("2"),
       " & MAE) is calculated on:",
       tags$ol(
        tags$li("the full dataset"),
        tags$li("the training dataset, and"),
        tags$li("the validation dataset.")
       ),
       "In addition, a plot of predicted sales prices vs. gross living area and residuals ",
       code("(predicted sale price - actual sale price)"),
       "are shown on the second and third tab."
     )
   }

   observeEvent(input$help, {
     showModal(helpModal())
   })

   showModal(helpModal())
}

# Run the application 
shinyApp(ui = ui, server = server)


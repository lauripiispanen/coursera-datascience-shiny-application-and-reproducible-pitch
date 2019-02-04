library(shiny)
library(ggplot2)
library(dplyr)
library(caret)
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
  "LotArea" = "Lot area"
)

tuningGrid <- list(
  "glm" = expand.grid(),
  "svmLinear3" = expand.grid(cost = 0.25,
                             loss = 'L1'),
  "xgbTree" = expand.grid(eta = 0.3,
                          gamma = 0,
                          colsample_bytree = 0.8,
                          min_child_weight = 1,
                          subsample = 1,
                          nrounds = c(1:80),
                          max_depth = 3)
)

ui <- fluidPage(
   titlePanel("House prices prediction"),
   
   sidebarLayout(
      sidebarPanel(
        checkboxGroupInput("selectedParameters",
                           label = h3("Model parameters"),
                           choiceNames = as.character(variables),
                           choiceValues = names(variables)),
         h3("Tuning parameters"),
         sliderInput("nrounds", "Number of boosting rounds:",
                    min = 1,
                    max = 120,
                    value = 80,
                    step = 1),
         sliderInput("eta", "Learning rate:",
                     min = 0.001,
                     max = 1,
                     value = 0.3,
                     step = 0.05),
         sliderInput("gamma", "Minimum loss reduction:",
                     min = 0,
                     max = 100,
                     value = 0,
                     step = 1),
         sliderInput("max_depth", "Maximum tree depth:",
                     min = 0,
                     max = 15,
                     value = 3,
                     step = 1),
         sliderInput("min_child_weight", "Minimum child weight:",
                     min = 0,
                     max = 15,
                     value = 1,
                     step = 1),
         sliderInput("subsample", "Subsampling ratio:",
                     min = 0,
                     max = 1,
                     value = 1,
                     step = 0.1),
         sliderInput("colsample_bytree", "Column subsample by tree:",
                     min = 0,
                     max = 1,
                     value = 0.8,
                     step = 0.1)
      ),
      mainPanel(
        tabsetPanel(
          tabPanel("Data", plotOutput("chartPlot")),
          tabPanel("Prediction", plotOutput("predictionPlot")),
          tabPanel("Residuals", plotOutput("residualPlot")),
          tabPanel("Learning", plotOutput("learningPlot"))
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
       eta = input$eta,
       gamma = input$gamma,
       min_child_weight = input$min_child_weight,
       subsample = input$subsample,
       nrounds = c(1:input$nrounds),
       max_depth = input$max_depth,
       colsample_bytree = input$colsample_bytree
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
             method = "xgbTree",
             tuneGrid = tuning_settings(),
             verbose = TRUE)
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
   
   output$learningPlot <- renderPlot({
     m <- mdl() 
     if (!is.null(m)) {
       plot(m)
     }
   })
   
   output$show_model_details <- reactive({
     !is.null(mdl())
   })
   
   outputOptions(output, "show_model_details", suspendWhenHidden = FALSE)
}

# Run the application 
shinyApp(ui = ui, server = server)


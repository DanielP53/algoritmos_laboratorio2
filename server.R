
library(shiny)
library(reticulate)
use_condaenv(condaenv = "base")

source_python("algoritmos.py")

np <- import("numpy")


#tableOut, soluc = newtonSolverX(-5, "2x^5 - 3", 0.0001)

shinyServer(function(input, output) {
    
    #Evento y evaluación de metodo de newton para ceros
    bisectionCalculate<-eventReactive(input$bisectionResolve, {
        eq<-input$bisectionEquation[1]
        print(eq)
        initInterval<-input$bisectionInterval[1]
        print(initInterval)
        bisectionKmax <- input$bisectionKmax[1]
        print(bisectionKmax)
        error<-input$bisectionTolerance[1]
        print(error)
        outs<-bisectionMethod(eq, initInterval, error, bisectionKmax)
        outs
    })
    
    #Evento y evaluación de diferencias finitas
    newtonCalculate<-eventReactive(input$newtonMethodResolve, {
      eq<-input$newtonFunction[1]
      print(eq)
      initSol<-input$newtonInitialSol[1]
      print(initSol)
      newtonKmax <- input$newtonMethodMaxIter[1]
      print(newtonKmax)
      error<-input$newtonTolerance[1]
      print(error)
      outs<-newtonRaphsonMethod(eq, initSol, error, newtonKmax)
      outs
    })
    
    rosenbrocCalculate<-eventReactive(input$rosenMethodSolve, {
      x0 <- input$rosenX0Input[1]
      print(x0)
      stepSize = input$rosenStepSize[1]
      print(stepSize)
      outs<-runRosenbrock(x0, stepSize)
      outs
    })

    #Evento y evaluacion de gradient descent
    gdCalculate<-eventReactive(input$gdMethodResolve, {

      matrixString <- input$gdMatrix
      filas <- strsplit(matrixString, " ")[[1]]
      Q <- do.call(rbind, lapply(filas, function(fila) as.numeric(unlist(strsplit(fila, ",")))))
      cString <- input$gdC
      c <- np$array(as.numeric(unlist(strsplit(cString, ","))))
      xString <- input$gdInitial
      x <- np$array(as.numeric(unlist(strsplit(xString, ","))))
      epsilon <- as.numeric(input$gdTolerance)
      max_iter <- as.integer(input$gdMethodMaxIter)
      stepSize <- as.character(input$gdStepSize)
      alphaValue <- as.numeric(input$gdAlphaValue)
      
      if (stepSize == 'constant') {
        outs<-gradient_descent(Q, c, x, epsilon, max_iter, step_size_type=stepSize, alpha_value=alphaValue)
      } else {
        outs<-gradient_descent(Q, c, x, epsilon, max_iter, step_size_type=stepSize)
      }
        
      
      outs <- do.call(rbind, lapply(outs, function(row) {
        data.frame(
          Iteration = row[[1]],
          x_k = paste(row[[2]], collapse = ", "),
          p_k = paste(row[[3]], collapse = ", "),
          Norm_Gradient = row[[4]]
        )
      }))
      
      #outs<- Q
      outs
    })
    
    
    #REnder metodo de Newton
    output$salidaTabla<-renderTable({
      bisectionCalculate()
    })
    
    #Render Diferncias Finitas
    output$salidaNewton<-renderTable({
      newtonCalculate()
    })
    
    output$salidaRosen<-renderTable({
      rosenbrocCalculate()
    })
    
    #Render metodo GD
    output$salidaGradient<-renderTable({
      gdCalculate()
    })
    
})

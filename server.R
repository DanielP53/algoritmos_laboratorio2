
library(shiny)
library(reticulate)
use_condaenv(condaenv = "base")

source_python("algoritmos.py")

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
      matriz <- do.call(rbind, lapply(filas, function(fila) as.numeric(unlist(strsplit(fila, ",")))))
      print(filas)
      c <- input$gdC
      x0 <- input$gdInitialSol
      outs<-matriz
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


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
    
    
    #REnder metodo de Newton
    output$salidaTabla<-renderTable({
      bisectionCalculate()
    })
    
    #Render Diferncias Finitas
    output$salidaNewton<-renderTable({
      newtonCalculate()
    })
    
    
})

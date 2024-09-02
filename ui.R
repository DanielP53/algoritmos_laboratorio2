library(shiny)
library(shinydashboard)

# Define UI for application that draws a histogram

# Bisection Method
tabIdBisectionMethod = "Biseccion"
tabBisectionTitle = "Método de la bisección"

bisectionEquationId = "bisectionEquation"
bisectionEquationLabel = "Ingrese la ecuación"

bisectionIntervalId = "bisectionInterval"
bisectionIntervalLabel = "Intervalo [a,b]. Escribir como a,b sin los corchetes"

bisectionKMaxId = "bisectionKmax"
bisectionKMaxLabel = "Máximo de iteraciones $$k_{max}$$"

bisectionToleranceId = "bisectionTolerance"
bisectionToleranceLabel = "Tolerancia"

bisectionSolveButtonId = "bisectionResolve"
bisectionSolveButtonLabel = "Resolver por método de bisección"
# Bisection Method

# Newton Raphson
newtonMethodTabId = "NewtonRaphson"
newtonMethodTabTitle = "Método Newton-Raphson"

newtonMethodFunctionId = "newtonFunction"
newtonMethodFunctionLabel = "Función diferenciable"

newtonMethodInitialSolId = "newtonInitialSol"
newtonMethodInitialSolLabel = "Solución inicial $$x_0$$"

newtonMethodMaxIterationId = "newtonMethodMaxIter"
newtonMethodMaxIterationLabel = "Máximo de iteraciones $$k_{max}$$"

newtonMethodTolId = "newtonTolerance"
newtonMethodTolLabel = "Tolerancia"

newtonMethodSolveButtonId = "newtonMethodResolve"
newtonMethodSolveButtonLabel = "Resolver por Netwon-Raphson"

# Netwon Rapshon


# Gradient Descent
newtonMethodTabId = "Gradient Descent"
newtonMethodTabTitle = "Método Gradient Descent"

newtonMethodFunctionId = "newtonFunction"
newtonMethodFunctionLabel = "Función diferenciable"

newtonMethodInitialSolId = "gdInitial"
newtonMethodInitialSolLabel = "Solución inicial $$x_0$$"

newtonMethodMaxIterationId = "gdMethodMaxIter"
newtonMethodMaxIterationLabel = "Máximo de iteraciones $$k_{max}$$"

newtonMethodTolId = "newtonTolerance"
newtonMethodTolLabel = "Tolerancia"

newtonMethodSolveButtonId = "newtonMethodResolve"
newtonMethodSolveButtonLabel = "Resolver por Netwon-Raphson"

dashboardPage(
    dashboardHeader(title = "Algoritmos en la Ciencia de Datos"),
    dashboardSidebar(
        sidebarMenu(
            menuItem(tabBisectionTitle, tabName = tabIdBisectionMethod),
            menuItem(newtonMethodTabTitle, tabName = newtonMethodTabId)
        )
    ),
    dashboardBody(
        tabItems(
            tabItem(tabIdBisectionMethod,
                    h1(tabBisectionTitle),
                    # textInput(id, label)
                    box(
                        textInput(bisectionEquationId, bisectionEquationLabel),
                        textInput(bisectionIntervalId, bisectionIntervalLabel),
                        textInput(bisectionKMaxId, withMathJax(bisectionKMaxLabel)),
                        textInput(bisectionToleranceId, bisectionToleranceLabel),
                        actionButton(bisectionSolveButtonId, bisectionSolveButtonLabel)
                    ),
                    tableOutput("salidaTabla")),
            
            tabItem(newtonMethodTabId,
                    h1(newtonMethodTabTitle),
                    box(
                    textInput(newtonMethodFunctionId, newtonMethodFunctionLabel),
                    textInput(newtonMethodInitialSolId, withMathJax(newtonMethodInitialSolLabel)),
                    textInput(newtonMethodMaxIterationId, withMathJax(newtonMethodMaxIterationLabel)),
                    textInput(newtonMethodTolId, newtonMethodTolLabel),
                    actionButton(newtonMethodSolveButtonId, newtonMethodSolveButtonLabel)
                    ),
                    tableOutput("salidaNewton"))
        )
    )
)

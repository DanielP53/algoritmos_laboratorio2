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

# Gradient Descent
gradientMethodTabId = "GradientDescent"
gradientMethodTabTitle = "Método Gradient Descent"

gradientMethodMatrixId = "gdMatrix"
gradientMethodMatrixLabel = HTML("Matriz Q, Para ingresar la matriz ingrese la linea separada por comas y espacios en blanco: 1,2,3&nbsp&nbsp4,5,6&nbsp&nbsp7,8,9")

gradientMethodCId = "gdC"
gradientMethodCLabel = "Vector C, ingresar separado por comas: 1, 4, 5"

gradientMethodInitialSolId = "gdInitial"
gradientMethodInitialSolLabel = "Solución inicial $$x_0$$solo ingresar separados por comas: 2, 1, 3"

gradientMethodStepSizeId = "gdStepSize"
gradientMethodStepSizeLabel = "Tipo de Stepsize, 'exact', 'variable', 'constant'"

gradientMethodAlphaValueId = "gdAlphaValue"
gradientMethodAlphaValueLabel = "Alpha Value para Step Size exacto'"

gradientMethodMaxIterationId = "gdMethodMaxIter"
gradientMethodMaxIterationLabel = "Máximo de iteraciones $$k_{max}$$"

gradientMethodTolId = "gdTolerance"
gradientMethodTolLabel = "Tolerancia"

gradientMethodSolveButtonId = "gdMethodResolve"
gradientMethodSolveButtonLabel = "Resolver por Gradient Descent"

rosenMethodTabTitle = "Función Rosenbrock"
rosenMethodTabId = "rosenFunction"
rosenx0InputId = "rosenX0Input"
rosenx0InputLabel = "x_0, Ingresalo como x,y sin parentesis"
rosenStepSizeId = "rosenStepSize"
rosenStepSizeLabel = "Step size"
rosenMethodSolveButtonId = "rosenMethodSolve"
rosenMethodSolveButtonLabel = "Resolver función Rosen"

dashboardPage(
    dashboardHeader(title = "Algoritmos en la Ciencia de Datos"),
    dashboardSidebar(
        sidebarMenu(
            menuItem(tabBisectionTitle, tabName = tabIdBisectionMethod),
            menuItem(newtonMethodTabTitle, tabName = newtonMethodTabId),
            menuItem(gradientMethodTabTitle, tabName = gradientMethodTabId),
            menuItem(rosenMethodTabTitle, tabName = rosenMethodTabId)
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
                    tableOutput("salidaNewton")),

            tabItem(gradientMethodTabId,
                    h1(gradientMethodTabTitle),
                    box(
                      textInput(gradientMethodMatrixId, gradientMethodMatrixLabel),
                      textInput(gradientMethodCId, gradientMethodCLabel),
                      textInput(gradientMethodInitialSolId, withMathJax(gradientMethodInitialSolLabel)),
                      textInput(gradientMethodStepSizeId, gradientMethodStepSizeLabel),
                      textInput(gradientMethodAlphaValueId, gradientMethodAlphaValueLabel),
                      textInput(gradientMethodMaxIterationId, withMathJax(gradientMethodMaxIterationLabel)),
                      textInput(gradientMethodTolId, gradientMethodTolLabel),
                      actionButton(gradientMethodSolveButtonId, gradientMethodSolveButtonLabel)
                    ),
                    tableOutput("salidaGradient")),
            tabItem(rosenMethodTabId,
                    h1(rosenMethodTabTitle),
                    box(
                      textInput(rosenx0InputId, rosenx0InputLabel),
                      textInput(rosenStepSizeId, rosenStepSizeLabel),
                      actionButton(rosenMethodSolveButtonId, rosenMethodSolveButtonLabel)
                    ),
                    tableOutput("salidaRosen"))
        )
    )
)

import pandas as pd
import sympy as sym
import re

def processEquation(equation):
    strOut = equation.replace('^', '**')
    x = sym.symbols('x')
    expr = sym.sympify(strOut)
    print(expr)
    def f(value):
      return float(expr.subs(x, value))

    return f


def bisectionMethod(equation, interval, tol, maxIter):
    try:
        f = processEquation(equation)
        intervalItems = interval.split(',')
        a = float(intervalItems[0])
        b = float(intervalItems[1])
        tol = float(tol)
        maxIter = int(maxIter)
        results = []
    except ValueError as e:
        print(e)
    for k in range(maxIter):
        c = (a + b) / 2
        fc = f(c)
        error = (abs(a - b) / 2)
        results.append({
            'iteration': k + 1,
            'x_k': fc,
            'punto medio': c,
            'a': a,
            'b': b,
            'Error': error
        })
        if (abs(fc) < tol):
            break
        elif (f(a) * fc < 0):
            b = c
        else:
            a = c

    bisectionOutput = pd.DataFrame(results)

    return bisectionOutput
    

def newtonRaphsonMethod(equation, x0, tol, maxIter):
  results = []
  f = processEquation(equation)
  x0 = float(x0)
  tol = float(tol)
  maxIter = int(maxIter)
  x = sym.symbols('x')
  equation = sym.sympify(equation)
  for k in range(maxIter):
    fx = f(x0)
    dfx = sym.diff(equation, x)
    print(dfx)
    dfx = sym.lambdify(x, dfx)
    dfx = dfx(x0)
    error = abs(fx)
    if (error < tol):
      break

    x0 = x0 - (fx / dfx)

    results.append({
      'iteration': k + 1,
      'x_k': x0,
      'Error': error
    })
    

  df = pd.DataFrame(results)
  return df


#Evaluación REGREX
def evaluate_Fx(str_equ, valX):
  x = valX
  #strOut = str_equ
  strOut = str_equ.replace("x", '*(x)')
  strOut = strOut.replace("^", "**")
  out = eval(strOut)
  print(strOut)
  return out

#Deferencias finitas para derivadas
def evaluate_derivate_fx(str_equ, x, h):
  strOut = str_equ.replace("x", '*(x + h)')
  strOut = strOut.replace("^", "**")
  strOut = "-4*(" + strOut + ")"
  out = eval(strOut)
  
  strOut = str_equ.replace("x", '*(x + 2*h)')
  strOut = strOut.replace("^", "**")
  out = out + eval(strOut)
  
  strOut = str_equ.replace("x", '*(x)')
  strOut = strOut.replace("^", "**")
  strOut = "3*(" + strOut + ")"
  out = out + eval(strOut)
  
  out = -out/(2*h)
  print(out)
  return out

#Resolverdor de Newton
def newtonSolverX(x0, f_x, eps):
  x0 = float(x0)
  eps = float(eps)
  xn = x0
  error = 1
  arrayIters = []
  arrayF_x = []
  arrayf_x = []
  arrayXn = []
  arrayErr = []
  
  i = 0
  h = 0.000001
  while(error > eps):
    print("...")
    x_n1 = xn - (evaluate_Fx(f_x, xn)/evaluate_derivate_fx(f_x, xn, h))
    error = abs(x_n1 - xn)
    i = i + 1
    xn = x_n1
    arrayIters.append(i)
    arrayXn.append(xn)
    arrayErr.append(error)
    solution = [i, xn, error]

  print("Finalizo...")
  TableOut = pandas.DataFrame({'Iter':arrayIters, 'Xn':arrayXn, 'Error': arrayErr})
  return TableOut

def add(a, b):
  a = int(a)
  b = int(b)
  resultado = a + b
  return "El resultado es: " + str(resultado)

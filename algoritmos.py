import pandas as pd
import sympy as sym
import re
import numpy as np

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

#Gradient Descent

def gradient_descent(Q, c, x0, epsilon, max_iter, step_size_type='constant', alpha_value=0.1):
    x = np.array(x0)
    results = []  # Lista para almacenar los resultados
    
    for k in range(max_iter):
        grad = gradient_quadratic(x, Q, c)
        norm_grad = np.linalg.norm(grad)
        
        if norm_grad < epsilon:
            results.append((k, x.tolist(), (-grad).tolist(), norm_grad))
            break

        if step_size_type == 'exact':
            alpha = exact_step_size(Q, c, x, grad)
        elif step_size_type == 'constant':
            alpha = alpha_value
        elif step_size_type == 'variable':
            alpha = 1 / (k + 1)
        else:
            raise ValueError("Tipo de step size no reconocido")

        p_k = -grad
        x = x - alpha * grad
        
        # Almacena los resultados de la iteración
        results.append((k, x.tolist(), p_k.tolist(), norm_grad))
        
    print("Results:", results)  # Añadido para depuració

    return results

def quadratic_function(x, Q, c):
    return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c.T, x)

def gradient_quadratic(x, Q, c):
    return np.dot(Q, x) + c

def exact_step_size(Q, c, x, grad):
    alpha = sym.symbols('alpha')

    x_new = x - alpha * grad

    f_expr = 0.5 * sym.Matrix(x_new).T * sym.Matrix(Q) * sym.Matrix(x_new) + sym.Matrix(c).T * sym.Matrix(x_new)
    f_expr = f_expr[0]

    f_prime = sym.diff(f_expr, alpha)

    f_prime_numeric = sym.lambdify(alpha, f_prime, 'numpy')

    alpha_opt = sym.solve(f_prime, alpha)

    for opt in alpha_opt:
        if opt.is_real and opt > 0:
            return float(opt)

    return 1.0


# Rosenbrock

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosenbrock(x):
	df_dx1 = -400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
	df_dx2 = 200 * (x[0] - x[1]**2)
	return np.array([df_dx1, df_dx2])

def clamp_values(value, threshold=1e6):
    return round(value, 4)

def check_invalid(value):
    if np.isnan(value) or np.isinf(value):
        return "Error"
    return clamp_values(value)

def validatedArray(arr):
   return [check_invalid(arr[0]), check_invalid(arr[1])]

def array_to_string(arr):
    return ', '.join(map(str, arr))

def rosen_gradient_descent(f, grad_f, x0, alpha = 0.5, tol = 1e-8, max_iter = 1000):
   xk = x0
   results = []
   for k in range(max_iter):
      grad = grad_f(xk)
      pk = grad
      #print(f"grad = {grad}")
      #print()
      xk = xk - alpha * grad
      #print(f"xk = {xk}")
      grad_norm = np.linalg.norm(grad)

      results.append([k+1, array_to_string(validatedArray(xk.copy())), array_to_string(validatedArray(pk)), check_invalid(grad_norm)])

      if (grad_norm < tol):
         break
   df = pd.DataFrame(results, columns=['Iteración', 'x_k', 'p_k', '||grad_f(x_k)||'])
   print(df)
   return df
   
def runRosenbrock(x_0, alpha):
   x_0 = x_0.split(',')
   print(x_0)
   x_0Arr = np.array([float(x_0[0]), float(x_0[1])])
   alpha = float(alpha)

   df = rosen_gradient_descent(rosenbrock, grad_rosenbrock, x0=x_0Arr, alpha=alpha)

   return df

#runRosenbrock('0,0', '0.05')
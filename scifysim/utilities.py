import sympy as sp
import numpy as np
from kernuller import fprint

import logging

logit = logging.getLogger(__name__)

def vec2diag(vec):
    thelen = vec.shape[0]
    A = sp.eye(thelen)
    for i in range(thelen):
        A[i,i]= vec[i]
    return A

class source(object):
    def __init__(self, xx, yy, ss):
        self.xx = xx
        self.yy = yy
        self.ss = ss

class ee(object):
    def __init__(self, expression):
        """
        ee is for Executable expression
        Encapsulates an expression for flexible lambda evaluation.
        expression  : a sympy expression to 
        """
        self.expr = expression
        if self.expr.is_Matrix:
            self.outshape = self.expr.shape
            self.outlen = np.prod(self.outshape)
            #self.callfunction = self.setup_matrixfunction()
        else :
            self.outlen = 1
            self.outshape = (1)
        
    def lambdify(self, args, modules="numexpr"):
        """
        Creates the lambda function. Currently, lambdification to numexpr
        does not support multiple outputs, so we create a list of lambda functions.
        args    : a tuple of sympy for symbols to use as inputs
        modules : The module to use for computation.
        """
        #Here, we have to decide if we need the function to all 
        if ("numexpr" in modules) and (self.outlen is not 1):
            thefuncs = []
            for i in range(self.outlen):
                thefuncs.append(sp.lambdify(args, sp.flatten(self.expr)[i], modules=modules))
            self.funcs = thefuncs
            self.callfunction = self.numexpr_call
            
        else :
            self.funcs = sp.lambdify(args, self.expr, modules=modules)
            self.callfunction = self.numpy_call
            
            
    def numexpr_call(self,*args):
        """
        The evaluation call for funcions for the numexpr case
        Evaluating the list of functions and returning it as an array
        """
        return np.stack([np.asarray(self.funcs[i](*args)) for i in range(self.outlen)])
    
    def numpy_call(self, *args):
        """
        Just the normal call of the lambda function
        """
        return self.funcs(*args) # Here, we flatten for consitency?
    
    def __call__(self,*args):
        return self.callfunction(*args)
    def fprint(self):
        fprint(self.expr)
    
def test_ex():
    x, y = sp.symbols("x y")
    f1 = x**2 + y**2
    objf1 = ee(f1)
    objf1.fprint()
    
    objf1.lambdify((x, y), modules="numpy")
    b = objf1(10.,11.)
    print("Result with numpy :\n", b)
    
    objf1.lambdify((x, y), modules="numexpr")
    b = objf1(10.,11.)
    print("Result with numexpr :\n", b)
    
    
    f2 = sp.Matrix([[x**2 + y**2],
                    [x**2 + y**2]])
    objf2 = ee(f2)
    objf2.fprint()
    
    objf2.lambdify((x, y), modules="numpy")
    b = objf2(10.,11.)
    print("Result with numpy :\n", b)
    
    objf2.lambdify((x, y), modules="numexpr")
    b = objf2(10.,11.)
    print("Result with numexpr :\n", b)
    
    
    f3 = sp.Matrix([[x**2 + y**2, x**2 - y**2],
                    [x**2 + 2*y**2, x**2 - 2*y**2]])
    objf3 = ee(f3)
    objf3.fprint()
    
    objf3.lambdify((x, y), modules="numpy")
    b = objf3(10.,11.)
    print("Result with numpy :\n", b)
    
    objf3.lambdify((x, y), modules="numexpr")
    print(objf3.outlen)
    b = objf3(10.,11.)
    print("Result with numexpr :\n", b)
    
    print("")
    print("Broadcasting example")
    print("====================")
    
    objf2.lambdify((x, y), modules="numpy")
    b = objf2(10.*np.arange(3)[:,None], 11.*np.arange(4)[None,:])
    print("broadcas with numpy :\n", b)
    print("shape:", b.shape)
    
    objf2.lambdify((x, y), modules="numexpr")
    b = objf2(10.*np.arange(3)[:,None], 11.*np.arange(4)[None,:])
    print("broadcas with numexpr :\n", b)
    print("shape:", b.shape)
    
    return b


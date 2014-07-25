# Conjugate Gradient Descent Algorithm
# Testing runtime -- push results to SQLite 3 DB

import numpy
import sys
import math
import sqlite3 
from scipy.sparse.linalg.isolve import _iterative
from scipy.sparse.linalg.isolve.utils import make_system
import scipy.sparse.linalg.cg

def NumCGIterations(A, b, x0=None, tol=1e-12, maxiter=None, xtype=None, M=None, callback=None):
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'cgrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    work = np.zeros(4*n,dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
        ijob = 2

    if info > 0 and iter_ == maxiter and resid > tol:
        # info isn't set appropriately otherwise
        info = iter_

    return iter_

scipy.sparse.linalg.cg = NumCGIterations

# globals
error = sys.maxint
matrix_size = 1
experimentno = 0
# Conjugate Gradient Descent




# create and connect to DB
conn = sqlite3.connect('test.db')
c = conn.cursor()
# set up DB 
# ENTRY: eigenvalues: multiplicity | disttribution or clustering type | length | # iterations
c.execute('''CREATE TABLE IF NOT EXISTS Experiments (ExperimentNumber int, Size int, Distribution_or_Cluster varchar(255), NumberOfIterations int);''')
conn.commit()


## CREATE MATRICES
# execute every experiment from 1 to 10,000

# -- CLUSTERS -- 

# -- DISTRIBUTIONS -- 

# Logistic 

while (matrix_size < 10000):
    loc, scale = 10, 1
    s = numpy.random.logistic(loc, scale, matrix_size)
    s.round()
    b = numpy.random.random_integers(0, high=matrix_size, size=(matrix_size, 1.))
    A = numpy.diag(s)
    x = scipy.sparse.linalg.cg(A,b)
    print x
    matrix_size*=10

matrix_size = 1
#count, bins, ignored = plt.hist(s, bin=50)


#while (error > epsilon): 
    
conn.close()
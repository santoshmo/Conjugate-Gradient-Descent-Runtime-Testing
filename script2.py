import numpy as np
import sys
import math
import sqlite3 
import scipy
from scipy.sparse.linalg.isolve import _iterative
from scipy.sparse.linalg.isolve.utils import make_system
import scipy.sparse.linalg
import random

def cgr(A, b, k, eps):
	A = np.matrix(A)
	b = np.matrix(b)
	n = length(b)
	residuals = np.zeroes(k,1)
	x = np.zeroes(n,1)
	r0 = b
	p = r0
	for x in range(1,k):
		pp = A*p 
		alpha = np.transpose(r0)*r0
		print alpha
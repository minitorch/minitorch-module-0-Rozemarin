"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, TypeVar

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
# Multiplies two numbers
def mul(x: float, y: float) -> float:
    return x * y

# Returns the input unchanged
def id(x: float) -> float:
    return x

# Adds two numbers
def add(x: float, y: float) -> float:
    return x + y

# Negates a number
def neg(x: float) -> float:
    return -x

# Checks if one number is less than another
def lt(x: float, y: float) -> bool:
    return x < y

# Checks if two numbers are equal
def eq(x: float, y: float) -> bool:
    return x == y

# Returns the larger of two numbers
def max(x: float, y: float) -> float:
    return x if x > y else y

# Checks if two numbers are close in value
def is_close(x: float, y: float, tol: float = 1e-5) -> bool:
    return abs(x - y) <= tol

# Calculates the sigmoid function
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

# Applies the ReLU activation function
def relu(x: float) -> float:
    return max(0, x)

# Calculates the natural logarithm
def log(x: float) -> float:
    return math.log(x)

# Calculates the exponential function
def exp(x: float) -> float:
    return math.exp(x)

# Calculates the reciprocal
def inv(x: float) -> float:
    return 1 / x if x != 0 else 0.0

# Computes the derivative of log times a second argument
def log_back(x: float, d: float) -> float:
    return d / x if x != 0 else 0.0  

# Computes the derivative of reciprocal times a second argument
def inv_back(x: float, d: float) -> float:
    return -d / (x * x) if x != 0 else 0.0 

# Computes the derivative of ReLU times a second argument
def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type

# Higher-order function that applies a given function to each element of an iterable
def map(func: Callable[[T], U], iterable: Iterable[T]) -> List[U]:
    arr = []
    for el in iterable:
        arr.append(func(el))
    return arr

# Higher-order function that combines elements from two iterables using a given function
def zipWith(func: Callable[[T, T], U], iterable1: Iterable[T], iterable2: Iterable[T]) -> List[U]:
    arr = []
    for el1, el2 in zip(iterable1, iterable2):
        arr.append(func(el1, el2))
    return arr

# Higher-order function that reduces an iterable to a single value using a given function
def reduce(func: Callable[[T, T], T], iterable: Iterable[T], initial: T) -> T:
    ans = initial
    for el in iterable:
        ans = func(ans, el)
    return ans

# Negate all elements in a list using map
def negList(lst: List[float]) -> List[float]:
    return map(neg, lst)

# Add corresponding elements from two lists using zipWith
def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    return zipWith(add, lst1, lst2)

# Sum all elements in a list using reduce
def sum(lst: List[float]) -> float:
    return reduce(add, lst, 0.0)

# Calculate the product of all elements in a list using reduce
def prod(lst: List[float]) -> float:
    return reduce(mul, lst, 1.0)

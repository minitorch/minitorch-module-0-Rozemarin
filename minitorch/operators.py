"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, List, TypeVar

# ## Task 0.1
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


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Check if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-5) -> bool:
    """Check if two numbers are close in value, within a tolerance."""
    return abs(x - y) <= tol


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply the ReLU activation function."""
    return max(0, x)


def log(x: float) -> float:
    """Calculate the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of a number, returning 0 if x is 0."""
    return 1 / x if x != 0 else 0.0


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log with respect to x, scaled by d."""
    return d / x if x != 0 else 0.0


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of reciprocal with respect to x, scaled by d."""
    return -d / (x * x) if x != 0 else 0.0


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of ReLU with respect to x, scaled by d."""
    return d if x > 0 else 0.0


# ## Task 0.3
# Small practice library of elementary higher-order functions.

T = TypeVar("T")  # Input type
U = TypeVar("U")  # Output type


def map(func: Callable[[T], U], iterable: Iterable[T]) -> List[U]:
    """Apply a function to each element in an iterable."""
    arr = []
    for el in iterable:
        arr.append(func(el))
    return arr


def zipWith(
    func: Callable[[T, T], U], iterable1: Iterable[T], iterable2: Iterable[T]
) -> List[U]:
    """Apply a function to pairs of elements from two iterables."""
    arr = []
    for el1, el2 in zip(iterable1, iterable2):
        arr.append(func(el1, el2))
    return arr


def reduce(func: Callable[[T, T], T], iterable: Iterable[T], initial: T) -> T:
    """Reduce an iterable to a single value by iteratively applying a function."""
    ans = initial
    for el in iterable:
        ans = func(ans, el)
    return ans


def negList(lst: List[float]) -> List[float]:
    """Return a list with each element negated."""
    return map(neg, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Return a list with elements from two lists added element-wise."""
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, lst, 0.0)


def prod(lst: List[float]) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, lst, 1.0)

---
layout: post
title: Explore Data
date: 2025-10-05
#image: https://placehold.it/900x300
lead: "Exploring data is an essential component of model building."
categories: Python
subtitle: Learn more Data exploration
---

This code defines a decorator called random_state. Here's a breakdown of what it does.

```python
from collections import namedtuple

import contextlib
from functools import wraps
import hashlib

from joblib import Parallel, delayed
from importlib import import_module
import warnings
```

```python
@contextlib.contextmanager
def set_random_states(random_states, method_name, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_states (dict):
            Dictionary mapping each method to its current random state.
        method_name (str):
            Name of the method to set the random state for.
        set_model_random_state (function):
            Function to set the random state for the method.
    """
    original_np_state = np.random.get_state()
    random_np_state = random_states[method_name]
    np.random.set_state(random_np_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        set_model_random_state(current_np_state, method_name)

        np.random.set_state(original_np_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        method_name = function.__name__
        with set_random_states(self.random_states, method_name, self.set_random_state):
            return function(self, *args, **kwargs)

    return wrapper
```

This code defines a decorator called random_state. Here's a breakdown of what it does:

- `@wraps(function)`: This is a decorator from the functools module. It's used within other decorators to preserve the original function's metadata (like its name, docstring, etc.) when the decorator wraps it.

- `def wrapper(self, *args, **kwargs)`: This defines the inner function that will replace the original function when the @random_state decorator is applied. It takes self (assuming it's a method of a class), positional arguments (\*args), and keyword arguments (\*\*kwargs).

- `if self.random_states is None`: This checks if a random_states attribute of the object (self) is None. If it is, the original function is called directly without any random state management.

- `method_name = function.__name__` This gets the name of the original function that the decorator is applied to.

- `with set_random_states(self.random_states, method_name, self.set_random_state)`: This is the core of the decorator. It uses the set_random_states context manager (defined in the cell above) to manage the random state before executing the original function.
  It passes the self.random_states dictionary (presumably holding random states for different methods), the method_name, and self.set_random_state (a function to set the random state for the model).
  The with statement ensures that the random state is properly handled before and after the original function is executed.

- `return function(self, *args, **kwargs)`: This calls the original function with the provided arguments and returns its result. This happens within the set_random_states context manager, so the random state is set before this call and restored afterwards.

- `return wrapper`: The decorator returns the wrapper function, which will be executed instead of the original function whenever the decorated function is called.

In essence, the @random_state decorator provides a mechanism to set and manage the random state specifically for methods within a class that have a random_states attribute and a set_random_state method. This is useful for ensuring reproducibility when dealing with functions that rely on random number generation.

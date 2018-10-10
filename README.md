# The mpoints package

The `mpoints` package is a machine learning tool that implements the class of state-dependent Hawkes processes.
Its key features include both simulation and estimation (statistical inference).
It also contains a module with specialised plotting services.

State-dependent Hawkes processes belong to the class of hybrid marked point processes,
a class that models the arrival in time of random events and their interaction with the state of a system.

We strongly recommend to first read the [tutorial](https://mpoints.readthedocs.io/en/latest/tutorial.html).
It contains an introduction to this statistical model and illustrates the main services offered by the `mpoints` package.

For additional mathematical details, please consult the [documentation](https://mpoints.readthedocs.io) and the 
[references](https://mpoints.readthedocs.io/en/latest/references.html).

## Installation

The package can be easily installed via `pip`, a popular package management system for Python. In the terminal, simply enter the following command.

```
    pip install mpoints
```
If you are using virtual environments (with conda), make sure that you install the package in the intended environment.

Note: when installing `mpoints` on Windows, you may be prompted to install Visual C++ Build Tools if this is not already installed.
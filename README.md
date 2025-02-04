# amplitudes-meets-ML
Shared repo for some physicists to code up some Boltzmann Brains. I'm thinking we just start with some simple projects:

1) Use PyTorch to fit BRK.A ticker to an exponential curve.
2) Use a neural nets (feed forward, CNN, Transformers) to classify digits numerically.
3) Use transformer networks to predict tokens (letters/words/integers) in sequential data.

But first, some best practices for version control so we can speak the langauge of industry data science and ML. 

## Step 1: Setting up git
In order to use the full functionality of GitHub, we first need to setup Git... and even before that, need to ask Paolo what machine he uses (PC, Mac, etc.)

## Step 2: Setting up SSH key
When hosting a shared repository on GitHub, it is useful to setup a SSH cryptographic key, so you don't have to re-enter your password all the time.

## Step 3: Git Clone, Push, Pull, etc.
Best practices for version control when working in a group. Forking a branch, pull requests, etc.

## Step 4: Setting up virtual environments
Sometimes project collaborators will have different versions of the project depencies downloaded on their local machine. For this reason, it is useful to set up a virtual environment where dependencies (like PyTorch, MatplotLib, etc.) can be accessed locally, and shared across machines.

```
python3 -m venv venv
source venv/bin/activate
pip3 install requirements.txt
```
requirements.txt file is just a line by line list of libraries

## Definitions

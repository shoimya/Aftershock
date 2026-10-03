# Set up the Colab notebook

Type: task
Status: open
Blocked by: none

## Goal

Make an empty notebook in Google Colab and check that all the tools we need are already there.

## Downloads

| What | From | Goes to |
|---|---|---|
| Nothing is installed | | |
| The notebook file | You create it | Your Google Drive, folder `Colab Notebooks` (Colab saves it there for you) |

Nothing goes on your laptop. You only need a web browser and a Google account.

## Steps

1. Go to https://colab.research.google.com and sign in with your Google account.
2. Click **New notebook**.
3. Click the name at the top (`Untitled0.ipynb`) and rename it to `aftershock_v1.ipynb`.
4. Add a text cell at the top: `# Aftershock v1` and one line from the project description.
5. Add a code cell that loads every tool and prints its version:

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import sklearn
   import lightgbm
   import yfinance as yf

   for lib in [pd, np, sklearn, lightgbm, yf]:
       print(lib.__name__, lib.__version__)
   ```

6. Run the cell (Shift + Enter).
7. If a library is missing, add a cell above it with `!pip install yfinance` (or the missing name) and run it. This installs on Google's computer, not yours. You must run this cell again each time you open the notebook.

## Done when

- The notebook is called `aftershock_v1.ipynb` and is in your Google Drive.
- The version cell prints 5 lines and shows no errors.

# Make the three models

Type: task
Status: open
Blocked by: 04

## Goal

Write the three "guessers". Each one learns from past rows (`fit`) and then guesses the target for new rows (`predict`). All three work in log form, the same as the target.

## Downloads

None. `scikit-learn` and `lightgbm` are already in Colab (checked in ticket 01).

## The three models

| Model | How it guesses | Why we have it |
|---|---|---|
| **Naive** | Next 5 days will look like the last 5 days. The guess is just `vol_5d`. | The minimum. A real model must do better than this. |
| **HAR** | A straight-line formula: `target = b0 + b1 × vol_1d + b2 × vol_5d + b3 × vol_22d`. It learns the best `b` numbers from the past. | The classic textbook model (Corsi, 2009). The main model to beat. |
| **LightGBM** | A machine learning model made of many small decision trees. It uses all four inputs and can find curved patterns that a straight line misses. | The machine learning challenger. |

## Steps

1. Add a text cell: `## 4. Models`.
2. Add the models. They all have the same two functions, so the test in ticket 06 can treat them the same:

   ```python
   from sklearn.linear_model import LinearRegression
   from lightgbm import LGBMRegressor

   class Naive:
       def fit(self, X, y):
           return self  # nothing to learn
       def predict(self, X):
           return X["vol_5d"].values

   class HAR:
       cols = ["vol_1d", "vol_5d", "vol_22d"]
       def fit(self, X, y):
           self.model = LinearRegression().fit(X[self.cols], y)
           return self
       def predict(self, X):
           return self.model.predict(X[self.cols])

   class LGBM:
       def fit(self, X, y):
           self.model = LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                      num_leaves=15, random_state=42, verbose=-1)
           self.model.fit(X[FEATURES], y)
           return self
       def predict(self, X):
           return self.model.predict(X[FEATURES])

   MODELS = {"Naive": Naive, "HAR": HAR, "LightGBM": LGBM}
   ```

3. Quick smoke test on the first half of the data only (this is **not** the real test):

   ```python
   half = len(data) // 2
   X, y = data[FEATURES], data["target"]
   for name, Model in MODELS.items():
       m = Model().fit(X.iloc[:half], y.iloc[:half])
       print(name, m.predict(X.iloc[half:half + 3]).round(3))
   ```

## Important rule

Do **not** change the LightGBM settings after you see the test results in ticket 07. Changing settings to make the test score better is a hidden form of cheating. The settings above are fixed for v1.

## Done when

- The smoke test prints 3 numbers for each of the 3 models, with no errors.

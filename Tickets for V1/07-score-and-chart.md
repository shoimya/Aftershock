# Score the models and draw charts

Type: task
Status: open
Blocked by: 06

## Goal

Find out which model guessed best, and show it in a table and in charts.

## Downloads

None. Charts appear inside the notebook.

## Steps

1. Add a text cell: `## 6. Results`.
2. **Score table.** The score is MAE (mean absolute error): on average, how many percentage points the guess was off. Smaller is better.

   ```python
   scores = {}
   for name in MODELS:
       scores[name] = (preds_pct[name] - preds_pct["actual"]).abs().mean()
   score_table = pd.Series(scores).round(2).sort_values()
   print("Average error (percentage points):")
   print(score_table)
   ```

   Example of how to read it: "HAR 4.10" means HAR was off by about 4.1 points on average. If the real volatility was 20%, HAR's guess was usually somewhere around 16% to 24%.

3. **Error per year.** This shows if a model was only good in calm years:

   ```python
   yearly = preds_pct.groupby(preds_pct.index.year).apply(
       lambda g: pd.Series({n: (g[n] - g["actual"]).abs().mean() for n in MODELS}))
   yearly.plot(kind="bar", figsize=(12, 4), title="Average error per year (lower is better)")
   plt.show()
   ```

4. **Guess against reality, full period:**

   ```python
   ax = preds_pct["actual"].plot(figsize=(12, 4), color="lightgray", label="actual")
   preds_pct[["HAR", "LightGBM"]].plot(ax=ax)
   ax.set_title("Volatility over next 5 days: guess vs actual (% per year)")
   plt.show()
   ```

5. **Zoom on the 2020 crash** (Feb to May 2020), same chart with `preds_pct.loc["2020-02":"2020-05"]`. This shows how fast each model reacts to a shock.

## Done when

- The score table shows 3 numbers.
- Both HAR and LightGBM have a smaller error than Naive. (If not, something is wrong. Check tickets 03 to 06 again.)
- The 3 charts are in the notebook.

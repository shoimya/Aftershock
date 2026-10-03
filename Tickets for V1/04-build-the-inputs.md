# Build the four inputs

Type: task
Status: open
Blocked by: 03

## Goal

Make the four inputs (also called "features") that the models use to guess the target. Each input on day `t` must use **only** data up to the end of day `t`. Using any later data is cheating, and it makes the results look better than they really are.

## Downloads

None.

## The four inputs

| Name | Meaning |
|---|---|
| `vol_1d` | How much SPY moved today (yearly %) |
| `vol_5d` | How much SPY moved over the last 5 days, including today |
| `vol_22d` | How much SPY moved over the last 22 days (about one month) |
| `ret_1d` | Today's return. Falling days usually lead to more volatility than rising days, so this helps. |

## Steps

1. Add a text cell: `## 3. Inputs`.
2. Add a small helper and make the inputs:

   ```python
   def past_vol(n):
       # volatility over the last n days, including today
       return np.sqrt(252 / n * df["r2"].rolling(n).sum())

   TINY = 1e-4  # a few days have a return of exactly 0, and log(0) is not allowed
   df["vol_1d"] = np.log(past_vol(1).clip(lower=TINY))
   df["vol_5d"] = np.log(past_vol(5).clip(lower=TINY))
   df["vol_22d"] = np.log(past_vol(22).clip(lower=TINY))
   df["ret_1d"] = df["r"]

   FEATURES = ["vol_1d", "vol_5d", "vol_22d", "ret_1d"]
   ```

   The three volatility inputs are in log form, the same as the target.

3. Remove rows that are not complete:
   - the first 22 rows (not enough history for `vol_22d`)
   - the last 5 rows (no target yet)

   ```python
   data = df.dropna(subset=FEATURES + ["target"]).copy()
   print("Rows left:", len(data))
   print(data[FEATURES + ["target"]].describe().round(3))
   ```

4. Cheating check. Change one future price and confirm that today's inputs do not change:

   ```python
   test = df.copy()
   test.iloc[200, test.columns.get_loc("r2")] *= 10  # change day 200
   before = np.sqrt(252 / 22 * df["r2"].rolling(22).sum()).iloc[199]
   after = np.sqrt(252 / 22 * test["r2"].rolling(22).sum()).iloc[199]
   print("Input on day 199 did not change:", before == after)
   ```

## Done when

- `data` has no empty values.
- The cheating check prints `True`.

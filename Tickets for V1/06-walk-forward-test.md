# Run the walk-forward test

Type: task
Status: open
Blocked by: 05

## Goal

Test the models in the same way as real life: a model can only learn from the past, and then it must guess a future it has never seen. This is the most important ticket in the project.

## Downloads

None.

## How it works (simple)

We go one year at a time, from 2005 to today:

1. On the first trading day of the year (call it day `T`), train each model on **all** rows before `T`.
2. **The gap.** Each row's target looks 5 days into the future. So the last 5 rows before `T` have targets that use days on or after `T`. On day `T` those are not known yet. Remove those 5 rows from training.
3. Use the trained models to guess every day of that year.
4. Move to the next year and repeat. Each year the training data gets bigger.

The test years include the 2008 crash, the 2020 crash, and 2022. The models never see these years before they guess them.

## Steps

1. Add a text cell: `## 5. Walk-forward test`.
2. Add the loop:

   ```python
   results = []
   for year in range(2005, data.index[-1].year + 1):
       test_rows = data[data.index.year == year]
       if len(test_rows) == 0:
           continue
       T = test_rows.index[0]                 # first trading day of the year
       pos_T = data.index.get_loc(T)
       train_rows = data.iloc[: pos_T - H]   # the gap: drop the last H rows before T

       # safety check: the target of the last training row must end before T
       last_train_day = train_rows.index[-1]
       target_end_day = df.index[df.index.get_loc(last_train_day) + H]
       assert target_end_day < T, f"leak in {year}"

       out = pd.DataFrame({"actual": test_rows["target"]}, index=test_rows.index)
       for name, Model in MODELS.items():
           m = Model().fit(train_rows[FEATURES], train_rows["target"])
           out[name] = m.predict(test_rows[FEATURES])
       results.append(out)
       print(year, "train rows:", len(train_rows), "test rows:", len(test_rows))

   preds = pd.concat(results)
   ```

   Note: this uses row positions in `data`, which are trading days. `data` starts 22 days after the first price, but that does not change the gap, because the gap only looks back from `T`.

3. Turn everything back from log form into normal % per year:

   ```python
   preds_pct = np.exp(preds) * 100
   print(preds_pct.head())
   ```

## Done when

- The loop prints one line per year from 2005 to this year, and the training rows grow each year.
- The safety check never stops the loop.
- `preds_pct` has 4 columns: `actual`, `Naive`, `HAR`, `LightGBM`.

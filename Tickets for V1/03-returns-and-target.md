# Calculate returns and the target

Type: task
Status: open
Blocked by: 02

## Goal

Turn prices into daily returns. Then calculate the **target**: the number the models must guess. The target is how much SPY will move over the **next 5 trading days**, as a yearly percentage.

## Downloads

None. This step only does math on the data from ticket 02.

## Background (simple)

- **Daily return:** how much the price changed from yesterday. We use the "log return": `r = ln(price today / price yesterday)`. For small moves this is almost the same as the percent change.
- **Volatility:** the size of the moves. We square each return (so up and down moves both count as positive), add them up, and take the square root.
- **Yearly percentage:** a year has about 252 trading days. We scale by 252 so the number is easy to compare. SPY is usually around 15% to 20% a year.

## Steps

1. Add a text cell: `## 2. Returns and target`.
2. Make a table to hold everything:

   ```python
   H = 5  # forecast horizon: the next 5 trading days

   df = pd.DataFrame({"price": prices})
   df["r"] = np.log(df["price"] / df["price"].shift(1))
   df["r2"] = df["r"] ** 2
   ```

3. Calculate the target. For each day `t`, add up the squared returns of days `t+1` to `t+5`:

   ```python
   future_sum = df["r2"].rolling(H).sum().shift(-H)
   df["target_vol"] = np.sqrt(252 / H * future_sum)
   df["target"] = np.log(df["target_vol"])  # the models guess the log, because it is more stable
   ```

   - `rolling(H).sum()` on day `t+5` covers days `t+1` to `t+5`. `shift(-H)` moves it back to day `t`.
   - The last 5 days have no target yet (the future is not known). They are empty. That is correct.

4. Check one day by hand:

   ```python
   i = 100
   by_hand = np.sqrt(252 / H * sum(df["r"].iloc[i+1:i+1+H] ** 2))
   print(by_hand, df["target_vol"].iloc[i])  # the two numbers must be the same
   ```

5. Look at the target:

   ```python
   (df["target_vol"] * 100).plot(title="Volatility over the next 5 days (% per year)", figsize=(10, 4))
   plt.show()
   print("Average:", round(df["target_vol"].mean() * 100, 1), "%")
   ```

## Done when

- The hand check prints two equal numbers.
- The average is roughly 15% to 20%.
- The chart has tall spikes in late 2008 and March 2020.

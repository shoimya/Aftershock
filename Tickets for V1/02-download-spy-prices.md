# Download SPY prices

Type: task
Status: open
Blocked by: 01

## Goal

Get every daily price of SPY (a fund that follows the S&P 500) from its first day in 1993 until today.

## Downloads

| What | From | Goes to |
|---|---|---|
| SPY daily prices (about 8,000 rows, under 1 MB) | Yahoo Finance servers, through `yfinance` | The memory of the Colab session (a table called `prices`) |

Nothing goes on your laptop. When you close Colab, the data is gone. That is fine, because the download takes only a few seconds and you can run it again.

## Steps

1. Add a text cell: `## 1. Get the data`.
2. Add a code cell:

   ```python
   raw = yf.download("SPY", start="1993-01-29", auto_adjust=True, progress=False)
   prices = raw["Close"].squeeze().dropna()
   prices.name = "price"
   print(prices.head())
   print(prices.tail())
   print("Rows:", len(prices))
   ```

   - `auto_adjust=True` gives prices that include dividends. This is important. Without it, the price drops on each dividend day look like market moves.
   - `.squeeze()` turns the one-column table into a simple list of prices with dates.

3. Draw the price to see it:

   ```python
   prices.plot(title="SPY price (with dividends)", figsize=(10, 4))
   plt.show()
   ```

## Done when

- The first date is 1993-01-29 and the last date is the most recent trading day.
- There are about 8,000 rows or more, with no empty values.
- The chart goes up over time, with clear drops in 2000-2002, 2008 and 2020.

If the download fails or returns 0 rows, wait one minute and run it again. Yahoo sometimes blocks too many requests for a short time.

import yfinance as yf

def test_yf_options():
    ticker = yf.Ticker("SPY")

    print("Checking available expirations...")
    exps = ticker.options
    print("Expirations returned:", exps)

    if not exps:
        print("\n❌ No expirations returned.")
        print("➡️ This usually happens on weekends or when Yahoo's options API is down.")
        return

    # Try first expiration
    exp = exps[0]
    print(f"\nTrying first expiration: {exp}")

    try:
        chain = ticker.option_chain(exp)
        print("\nCalls head:")
        print(chain.calls.head())

        print("\nPuts head:")
        print(chain.puts.head())

        print("\n✅ yfinance options are working.")
    except Exception as e:
        print("\n❌ Failed to fetch the chain.")
        print("Error:", e)

if __name__ == "__main__":
    test_yf_options()
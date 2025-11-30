
The prices of options you see aren't based on what people actually think will happen to a stock. They're based on what a hypothetical risk neutral investor would pay. That's useful for pricing options, but it doesn't tell you what's going to happen in the real world.
In 2015, Stephen Ross published a theorey: he showed that under certain conditions, you can actually extract real-world probabilities from option prices. Not risk-neutral probabilities.

This project implements that theorem and visualizes it in 3D.

<img width="1486" height="634" alt="image" src="https://github.com/user-attachments/assets/4797a268-0d23-4aa3-bfcf-0439866ae166" />    
<img width="1431" height="622" alt="image" src="https://github.com/user-attachments/assets/de4ffd4e-a606-4402-8614-7a84c3965377" />  


## What This App Does

This is a Streamlit app that:

1. Pulls options data for major assets (SPY, QQQ, tech stocks, crypto)
2. Builds an implied volatility surface from that data
3. Computes the risk-neutral probability distribution using Breeden-Litzenberger (the classic method from 1978)
4. Applies the Recovery Theorem to convert those risk-neutral probabilities into *physical* probabilities

Here is a video of me going through the program and explaining with very simply example of how one might inteperate the plots: https://youtu.be/DxrStSYuIiY

## Installation

```bash
pip install -r requirements.txt
streamlit run recovery_theorem.py
```

## How to Read the Visualizations

### 1. Implied Volatility Surface

This is the starting point. It shows implied volatility across:
- **X-axis**: Strike prices
- **Y-axis**: Days to expiration  
- **Z-axis**: Implied volatility

What you're seeing is the famous "volatility smile". Out-of-the-money options have higher implied vol because of tail risk and skewness. The surface should be relatively smooth - if it's jagged, there's not enough liquidity in those contracts.
If you would like a futher explination I would suggest watching this video:
https://www.youtube.com/watch?v=G7gf-oXptxE  
https://www.youtube.com/watch?v=YH0tWpBaKGs

**What to look for:**
- The smile gets more pronounced further from the current price
- Near-dated options usually have lower volume
- If you see spikes or holes, that's illiquid strikes with bad pricing

### 2. Risk-Neutral Probability Density

This uses the Breeden-Litzenberger formula:

```
PDF(K) = e^(rT) × ∂²C/∂K²
```

Take the second derivative of call prices with respect to strike. That gives you the probability density implied by option prices.

**BUT** - and this is crucial - this is the *risk-neutral* density. It's the probability distribution in a world where everyone is risk-neutral and the expected return on everything is the risk free rate. Obviously, that's not reality.

**What to look for:**
- Where is the peak? That's the most likely outcome according to options markets
- Is it shifted left or right from the current spot price? (It's usually shifted slightly down due to volatility risk premium)
- How fat are the tails? Fatter tails = market pricing in more tail risk

### 3. Physical Probability Surface (Recovery Theorem)

- If the market is perfect (complete) and there exists a single valid discounting rule (the pricing kernel), then you can work backwards from the risk neutral probabilities to find the real world probabilities.

How? By building a Markov transition matrix from the risk-neutral densities across multiple expirations, finding the stationary distribution, and using that to back out real-world probabilities.

**What you're seeing:**
- **X-axis**: Strike prices
- **Y-axis**: Time to expiration
- **Z-axis**: Physical probability density

This surface tells you what the market *actually* thinks will happen, adjusted for risk aversion.

**Key differences from risk-neutral:**
- Usually shifted slightly higher (equity risk premium)
- Different tail behavior - the physical density accounts for crash risk differently
- Time decay looks different because you're seeing actual expected paths, not risk-neutral ones

**What to look for:**
- Compare this to the risk-neutral PDF. Where do they differ?
- The areas of highest probability are where the market thinks the asset will actually trade
- If the physical probabilities are significantly different from risk-neutral, that's telling you something about risk premia and market sentiment

## The Math (Briefly)

### Breeden-Litzenberger (1978)

They showed that the second derivative of the call price with respect to strike using black scholes formula gives you the risk-neutral density:

```
q(K) = e^(rT) × ∂²C(K,T)/∂K²
```

We smooth the call prices first (using Gaussian filters) because real market data is noisy, then take numerical derivatives.

### Ross Recovery Theorem (2015)

The idea: build a discrete-state transition matrix Q where Q[i,j] is the risk-neutral probability of moving from state i to state j. 

Then solve for the stationary distribution:

```
Q^T π = λ π
```

Where π is the principal eigenvector (eigenvalue closest to 1). This eigenvector is proportional to the physical probability distribution.

The key insight is that the pricing kernel relates physical and risk-neutral measures, and if you have a complete set of Arrow-Debreu securities (which options approximate), you can invert that relationship.

**Assumptions:**
- Complete markets (or close enough)
- Time-separable preferences  
- Stationary transition probabilities
- No arbitrage

These are strong assumptions. In practice, they're never perfectly satisfied, so take the results with a grain of salt.

**Recovery Theorem limitations** - This is a simplified implementation. The real Ross approach uses higher-dimensional state spaces and more sophisticated numerical methods. This version uses ~15-20 states for computational tractability. ( I am limited by the number of requests I can make)

## Why This Matters

If the Recovery Theorem actually works (and that's still debated in academic circles), it means you can extract market expectations about future returns directly from option prices. 

That's huge for:
- Tail risk hedging
- Understanding market sentiment
- Calibrating economic models
- Portfolio allocation based on market-implied probabilities

Lots of people think it's overly theoretical and the assumptions don't hold in reality. But even if it's just an approximation, it's a cool way to think about what option prices are really telling you.

## References

**Papers:**
- Breeden, D. T., & Litzenberger, R. H. (1978). "Prices of State-Contingent Claims Implicit in Option Prices." *Journal of Business*.
- Ross, S. A. (2015). "The Recovery Theorem." *Journal of Finance*, 70(2), 615-648.
- Carr, P., & Yu, J. (2012). "Risk, Return, and Ross Recovery." *Journal of Derivatives*, 20(1), 38-59.

**Critiques:**
- Borovička, J., Hansen, L. P., & Scheinkman, J. A. (2016). "Misspecified Recovery." *Journal of Finance*, 71(6), 2493-2544.
  - (They argue the assumptions are too strong and misspecification leads to garbage results)


# Gambling Odds To Outcome probabilities Conversion (goto_conversion)

The most common method used to convert betting odds to probabilities is to normalise the inverse odds (Multiplicative conversion). However, this method does not consider the favourite-longshot bias. 

To the best of our knowledge, there are two existing methods that attempt to consider the favourite-longshot bias. (i) Shin conversion [[1](#1),[2](#2),[3](#3)] maximises the expected profit for the bookmakers assuming a small proportion of bettors have inside information. (ii) Power conversion [[4](#4)] raises all inverse odds to the same constant power. However, both of these methods require iterative computation to convert betting odds to probabilities.

Our proposed method, **G**ambling **O**dds **T**o **O**utcome probabilities **Conversion** (`goto_conversion`) is significantly more efficient than Shin and Power conversion because it converts betting odds to probabilities directly without iterative computation.

The `goto_conversion` reduces all inverse odds by the same units of standard error. This attempts to consider the favourite-longshot bias by utilising the proportionately wider standard errors implied for inverses of longshot odds and vice-versa.

Furthermore, our table of experiment results show that the `goto_conversion` converts betting odds to probabilities more accurately than all three of these existing methods.

The favourite-longshot bias is not limited to gambling markets, it exists in stock markets too. Thus, we applied the original `goto_conversion` to stock markets by defining the `zero_sum` variant. Under the same philosophy as the original `goto_conversion`, `zero_sum` adjusts all predicted stock prices (e.g. weighted average price) by the same units of standard error to ensure all predicted stock prices relative to the index price (e.g. weighted average nasdaq price) sum to zero. This attempts to consider the favourite-longshot bias by utilising the wider standard errors implied for predicted stock prices with low trade volume and vice-versa.

# Presentation at the Royal Statistical Society

- Link to Presentation Recording: https://youtu.be/M00osEjcp_4?si=_WZv09311q3UoS9t&t=411

# Applications on Kaggle

To the best of my knowledge, on Kaggle, at least four gold medal solutions and many other medal solutions have publicly stated that they leveraged `goto_conversion`:
- [Gold Medal Winning (3rd out of 821) Solution from 2024 March Mania Kaggle Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/495101)
- [Gold Medal Winning (4th out of 821) Solution from 2024 March Mania Kaggle Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/494407)
- [1xGold and 2xSilver Medal Winning Solution from 2019 to 2022 March Mania Kaggle Competition](https://www.kaggle.com/code/kaito510/1xgold-2xsilvers-key-ingredient)
- [15xBronze Medal Winning (86th to 100th place out of 821) Solution from 2024 March Mania Kaggle Competition](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient)
- [Silver Medal Winning (38th out of 821) Solution from 2024 Match Mania Kaggle Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/485888#2740879)
- [Most Voted Solution from 2023 Optiver Kaggle Competition](https://www.kaggle.com/code/ravi20076/optiver-baseline-models?scriptVersionId=152991375)
- [Gold Medal Winning (14th out of 3225) Solution from 2023 Optiver Kaggle Competition (the `zero_sum` variant)](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/462653)

# Installation

Requires Python 3.7 or above.

```
pip install goto-conversion
```

# Usage

## Decimal Odds

```python
import goto_conversion
goto_conversion.goto_conversion([1.2, 3.4, 5.6])
```

```
[0.7753528189788175, 0.17479473292721065, 0.04985244809397199]
```

## American Odds

```python
import goto_conversion
goto_conversion.goto_conversion([-500, 240, 460], isAmericanOdds = True)
```

```
[0.7753528189788175, 0.17479473292721065, 0.04985244809397199]
```

# Pseudo Code

![alt text](https://github.com/gotoConversion/goto_conversion/blob/main/PseudoCode.png?raw=true)

# Experiment Results

Brier Score was mainly used to evaluate the accuracy of the probabilities implied by each conversion method. The Brier Score is essentially the mean squared error of the probabilities relative to the ground truth. Ranked Probability Score (RPS) was additionally used to evaluate the probabilities for football betting odds because the outcome is ordinal (home win, draw and away win). RPS is essentially the Brier Score on the cumulative probabilities.

The experiment results table below is based on the same 6,000 football matches' betting odds (home win, draw or away win) across four different bookmakers. `goto_conversion` outperforms all other conversion methods for all four bookmakers under both Brier Score and RPS. Kaggle notebook to reproduce the table below can be found [here](https://www.kaggle.com/code/kaito510/novel-conversion-of-football-betting-odds).

![alt text](https://github.com/gotoConversion/goto_conversion/blob/main/FballExperiment.png?raw=true)

# References

<a id="1">[1]</a> 
[H. S. Shin, “Prices of State Contingent Claims with Insider
traders, and the Favorite-Longshot Bias”. The Economic
Journal, 1992, 102, pp. 426-435.](https://doi.org/10.2307/2234526)

<a id="2">[2]</a>
[E. Štrumbelj, "On determining probability forecasts from betting odds".
International Journal of Forecasting, 2014, Volume 30, Issue 4,
pp. 934-943.](https://doi.org/10.1016/j.ijforecast.2014.02.008)

<a id="3">[3]</a>
[M. Berk, "Python implementation of Shin's method for calculating implied probabilities from bookmaker odds"](https://github.com/mberk/shin)

<a id="4">[4]</a>
[S. Clarke, S. Kovalchik, M. Ingram, "Adjusting bookmaker’s odds to allow for
overround". American Journal of Sports Science, 2017, Volume 5, Issue 6,
pp. 45-49.](https://doi.org/10.11648/j.ajss.20170506.12)

# Contact Me

via LinkedIn Message: https://www.linkedin.com/in/goto/

# Q&A

Q1. I want to know whether the teams in the csv file named mensProbabilitiesTable in the 538 data you created are in 2024 or 2023?

A1. 2024 but it is NOT 538 data, it is my data displayed in a format inspired by 538.

# goto_conversion (Novel Conversion of Betting Odds to Probabilities)

The most common method used to convert betting odds to probabilities is to normalise the inverse odds (Multiplicative conversion). However, this method does not consider the favourite-longshot bias. 

To the best of our knowledge, there are two existing methods that attempt to consider the favourite-longshot bias. (i) Shin conversion [[1](#1),[2](#2),[3](#3)] maximises the expected profit for the bookmakers assuming a small proportion of bettors have inside information. (ii) Power conversion [[4](#4)] raises all inverse odds to the same constant power. However, both of these methods require iterative computation to convert betting odds to probabilities.

Our proposed method (goto_conversion) is significantly more efficient than Shin and Power conversion because it converts betting odds to probabilities directly without iterative computation.

The goto_conversion reduces all inverse odds by the same units of standard error. This attempts to consider the favourite-longshot bias by utilising the proportionately wider standard errors implied for inverses of longshot odds and vice-versa.

Furthermore, our tables of experiment results below show that the goto_conversion converts betting odds to probabilities more accurately than all three of these existing methods.

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

The experiment results table directly below is based on the same 6,000 football matches' betting odds (home win, draw or away win) across four different bookmakers.

![alt text](https://github.com/gotoConversion/goto_conversion/blob/main/FballExperiment.png?raw=true)

Kaggle notebook to reproduce the table directly above: https://www.kaggle.com/code/kaito510/novel-conversion-of-football-betting-odds

The experiment results table directly below is based on 6,348 horse races' betting odds for the win and place markets.

![alt text](https://github.com/gotoConversion/goto_conversion/blob/main/RacingExperiment.png?raw=true)

Kaggle notebook to reproduce the table directly above: https://www.kaggle.com/code/kaito510/novel-conversion-of-horse-racing-odds

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

LinkedIn Message: https://www.linkedin.com/in/goto/

Kaggle Message: https://www.kaggle.com/kaito510/competitions

Or fire an issue on this repo.

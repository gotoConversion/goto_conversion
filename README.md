# `goto_conversion` - Powered over 10 Gold Medals and 100 Medals on Kaggle

`goto_conversion` has powered over 10 :1st_place_medal: gold-medal-winning solutions and 100 :2nd_place_medal: :3rd_place_medal: medal-winning solutions on Kaggle [[6](#6),[7](#7)]. They include:
- 1x :1st_place_medal: 2x :2nd_place_medal: 2x :3rd_place_medal: [5x Medals including 1x Gold (all solo) the Founder of `goto_conversion` won from 2019 to 2025 Basketball Outcome Prediction Competitions](https://www.kaggle.com/kaito510) :basketball:
- :1st_place_medal: [Gold Medal (2nd out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572528) :basketball:
- :1st_place_medal: [Gold Medal (3rd out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572553) :basketball:
- :1st_place_medal: [Gold Medal (5th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition (referred to as `kaito510` solution)](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572909) :basketball:
- :1st_place_medal: [Gold Medal (6th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572482) :basketball:
- :1st_place_medal: [Gold Medal (7th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572540) :basketball:
- :1st_place_medal: [Gold Medal (8th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572535) :basketball:
- 3x :1st_place_medal: [3xGold Medal (10th to 12th out of 1727) Solutions from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/code/kaito510/who-used-goto-conversion) :basketball:
- :1st_place_medal: [Gold Medal (14th out of 3225) Solution from 2023 Stock Market Prediction Competition (the `zero_sum` variant)](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/462653) :chart_with_upwards_trend:
- :1st_place_medal: [Gold Medal (3rd out of 821) Solution from 2024 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/495101) :basketball:
- :1st_place_medal: [Gold Medal (4th out of 821) Solution from 2024 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/494407) :basketball:
- :1st_place_medal: [Most Voted Solution from 2023 Stock Market Prediction Competition](https://www.kaggle.com/code/ravi20076/optiver-baseline-models?scriptVersionId=152991375) :chart_with_upwards_trend:
- 75x :2nd_place_medal: [75xSilver Medal (14th to 83th place out of 1727) Solution from 2025 Basketball Outcome Prediction Competition (`Akshar Patidar` and `Best overfitting` was a team of 2 and 5 respectively)](https://www.kaggle.com/code/kaito510/updated-goto-conversion-winning-solution) :basketball:
- :2nd_place_medal: [Silver Medal (38th out of 821) Solution from 2024 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/485888#2740879) :basketball:
- 19x :3rd_place_medal: [19xBronze Medal (86th to 100th place out of 821) Solution from 2024 Basketball Outcome Prediction Competition (`CV_conda` was a team of 5)](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient) :basketball:
- :microphone: [Presentation at the Royal Statistical Society](https://youtu.be/M00osEjcp_4?si=_WZv09311q3UoS9t&t=411) :book:
- :white_check_mark: [Approved by PySport](https://opensource.pysport.org/project/goto_conversion) :trophy:

# Ease of Use

To use `goto_conversion`, it does not require historical data for model fit, advanced domain knowledge, nor paid computational resources.
Linked below provides five examples of how to use `goto_conversion` in the freely available, Google Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xdo-4uZu0XFdbFuqZbV0gKUGs4L2rCAt?usp=sharing)

# Abstract

The most common method used to convert betting odds to probabilities is to normalise the inverse odds (Multiplicative conversion). However, this method does not consider the favourite-longshot bias. 

To the best of our knowledge, there are two existing methods that attempt to consider the favourite-longshot bias. (i) Shin conversion [[1](#1),[2](#2),[3](#3)] maximises the expected profit for the bookmakers assuming a small proportion of bettors have inside information. (ii) Power conversion [[4](#4)] raises all inverse odds to the same constant power.

Our proposed method `goto_conversion` reduces all inverse odds by the same units of standard error. This attempts to consider the favourite-longshot bias by utilising the proportionately wider standard errors implied for inverses of longshot odds and vice-versa. Our experiments show `goto_conversion` converts betting odds to probabilities more robustly than all three of these existing methods.

This package is an implementation of `goto_conversion` as well as `efficient_shin_conversion`. The Shin conversion is originally a numerical solution but according to Kizildemir 2024 [[5](#5)], we can enhance its efficiency by reduction to an analytical solution. We have implemented the enhanced Shin conversion proposed by Kizildemir 2024 as `efficient_shin_conversion` in this package.

The favourite-longshot bias is not limited to betting markets, it exists in stock markets too. Thus, we applied the original `goto_conversion` to stock markets by defining the `zero_sum` variant. Under the same philosophy as the original `goto_conversion`, `zero_sum` adjusts all predicted stock prices (e.g. weighted average price) by the same units of standard error to ensure all predicted stock prices relative to the index price (e.g. weighted average nasdaq price) sum to zero. This attempts to consider the favourite-longshot bias by utilising the wider standard errors implied for predicted stock prices with low trade volume and vice-versa.

# Pseudo Code

![alt text](https://github.com/gotoConversion/goto_conversion/blob/main/PseudoCode.png?raw=true)

# References

<a id="1">[1]</a> 
[H. S. Shin, “Prices of State Contingent Claims with Insider
traders, and the Favorite-Longshot Bias”. The Economic
Journal, 1992, 102, pp. 426-435.](https://doi.org/10.2307/2234526)

<a id="2">[2]</a>
[E. Štrumbelj, "On determining probability forecasts from gambling odds".
International Journal of Forecasting, 2014, Volume 30, Issue 4,
pp. 934-943.](https://doi.org/10.1016/j.ijforecast.2014.02.008)

<a id="3">[3]</a>
[M. Berk, "Python implementation of Shin's method for calculating implied probabilities from bookmaker odds"](https://github.com/mberk/shin)

<a id="4">[4]</a>
[S. Clarke, S. Kovalchik, M. Ingram, "Adjusting bookmaker’s odds to allow for
overround". American Journal of Sports Science, 2017, Volume 5, Issue 6,
pp. 45-49.](https://doi.org/10.11648/j.ajss.20170506.12)

<a id="5">[5]</a>
[Kizildemir, M., Akin, E., & Alkan, A. (2024). A Family of Solutions Related to Shin’s Model For Probability Forecasts. Cambridge Open Engage](https://doi.org/10.33774/coe-2024-dwb6t)

<a id="6">[6]</a>
[goto_conversion's Kaggle Profile](https://www.kaggle.com/kaito510)

<a id="7">[7]</a>
[Kaggle Main Page](https://www.kaggle.com)

# Contact Me

via LinkedIn Message: https://www.linkedin.com/in/goto/

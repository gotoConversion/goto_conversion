# `goto_conversion` - Powered $47,000 of prize money, over 10 Gold Medals and 100 Medals on Kaggle

**LATEST UPDATES ARE ON MY SUBSTACK:**

[![Substack](https://img.shields.io/badge/Substack-%23006f5c.svg?style=for-the-badge&logo=substack&logoColor=FF6719)](https://gotoconversion.substack.com/)

# Wins

`goto_conversion` has powered over 10 :1st_place_medal: gold-medal-winning solutions and 100 :2nd_place_medal: :3rd_place_medal: medal-winning solutions on Kaggle [[1](#1)]. They include:
- 1x :1st_place_medal: 2x :2nd_place_medal: 2x :3rd_place_medal: [5x Medals including 1x Gold (all solo) the Founder of `goto_conversion` won from 2019 to 2025 Basketball Outcome Prediction Competitions](https://www.kaggle.com/kaito510) :basketball:
- 75x :2nd_place_medal: [75xSilver Medal (14th to 83th place out of 1727) Solution from 2025 Basketball Outcome Prediction Competition (`Akshar Patidar` and `Best overfitting` was a team of 2 and 5 respectively)](https://www.kaggle.com/code/kaito510/updated-goto-conversion-winning-solution) :basketball:
- 19x :3rd_place_medal: [19xBronze Medal (86th to 100th place out of 821) Solution from 2024 Basketball Outcome Prediction Competition (`CV_conda` was a team of 5)](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient) :basketball:
- :moneybag: $8,000 Winner :1st_place_medal: [Gold Medal (2nd out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572528) :basketball:
- :moneybag: $7,000 Winner :1st_place_medal: [Gold Medal (3rd out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572553) :basketball:
- :moneybag: $5,000 Winner :1st_place_medal: [Gold Medal (5th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition (referred to as `kaito510` solution)](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572909) :basketball:
- :moneybag: $5,000 Winner :1st_place_medal: [Gold Medal (6th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572482) :basketball:
- :moneybag: $5,000 Winner :1st_place_medal: [Gold Medal (7th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572540) :basketball:
- :moneybag: $5,000 Winner :1st_place_medal: [Gold Medal (8th out of 1727) Solution from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572535) :basketball:
- :moneybag: $7,000 Winner :1st_place_medal: [Gold Medal (3rd out of 821) Solution from 2024 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/495101) :basketball:
- :moneybag: $5,000 Winner :1st_place_medal: [Gold Medal (4th out of 821) Solution from 2024 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/494407) :basketball:
- :1st_place_medal: [Gold Medal (14th out of 3225) Solution from 2023 Stock Market Prediction Competition (the `zero_sum` variant)](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/462653) :chart_with_upwards_trend:
- 3x :1st_place_medal: [3xGold Medal (10th to 12th out of 1727) Solutions from 2025 Basketball Outcome Prediction Competition](https://www.kaggle.com/code/kaito510/who-used-goto-conversion) :basketball:
- :2nd_place_medal: [Silver Medal (38th out of 821) Solution from 2024 Basketball Outcome Prediction Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/485888#2740879) :basketball:
- :white_check_mark: [Approved by PySport](https://opensource.pysport.org/project/goto_conversion) :trophy:

# Ease of Use

To use `goto_conversion`, it does not require historical data for model fit, advanced domain knowledge, nor paid computational resources.
Linked below provides five examples of how to use `goto_conversion` in the freely available, Google Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xdo-4uZu0XFdbFuqZbV0gKUGs4L2rCAt?usp=sharing)

# Abstract

Our proposed method `goto_conversion` reduces all inverse odds by the same units of standard error. This attempts to consider the favourite-longshot bias by utilising the proportionately wider standard errors implied for inverses of longshot odds and vice versa.

This repository's main purpose is to implement `goto_conversion`, but also implements some other functions, such as `efficient_shin_conversion`. The Shin conversion [[2](#2)] is originally a numerical solution, but according to [[3](#3)], we can enhance its efficiency by reducing it to an analytical solution. We have implemented the enhanced Shin conversion as `efficient_shin_conversion` in this package.

The favourite-longshot bias is not limited to betting markets; it exists in stock markets too. Thus, we applied the original `goto_conversion` to stock markets by defining the `zero_sum` variant. Under the same philosophy as the original `goto_conversion`, `zero_sum` adjusts all predicted stock prices (e.g. weighted average price) by the same units of standard error to ensure all predicted stock prices relative to the index price (e.g. weighted average NASDAQ price) sum to zero. This attempts to consider the favourite-longshot bias by utilising the wider standard errors implied for predicted stock prices with low trade volume and vice versa.

# References

<a id="1">[1]</a>
[goto_conversion's Kaggle Profile](https://www.kaggle.com/kaito510)

<a id="2">[2]</a>
[E. Štrumbelj, "On determining probability forecasts from gambling odds".
International Journal of Forecasting, 2014, Volume 30, Issue 4,
pp. 934-943.](https://doi.org/10.1016/j.ijforecast.2014.02.008)

<a id="3">[3]</a>
[Kizildemir, Melis, Akin, Ertugrul and Alkan, Altug. "A family of solutions related to Shin’s model for probability forecasts" Journal of Quantitative Analysis in Sports, vol. 21, no. 2, 2025, pp. 153-158.](https://doi.org/10.1515/jqas-2024-0064)

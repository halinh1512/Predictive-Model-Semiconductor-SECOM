# Predictive-Model-Semiconductor-SECOM
SECOM is a multivariate dataset providing information of a complex modern semi-conductor manufacturing process. These information are signals collected from various sensors and measurement points, which are used to monitor the whole production line. "If we consider each type of signal as a feature, then feature selection may be applied to identify the most relevant signals. [...] the labels represent a simple pass/fail yield for in house line testing, [...]" (UCI - Machine Learning Repository (2008), http://archive.ics.uci.edu/ml/datasets/secom).

SECOM dataset includes 1567 rows (observations) with 590 columns representing 590 features/signals collected from sensors, together with the labels representing pass (-1) / fail (1) yield for in house line testing and associated date time stamp.

The key ideas are first to perform data cleaning and preparation to remove noise and deal with missing values. Then feature selection methods are selected to extract the most important variables. As the dataset contains Rare Events (Fail case) which we are interested to find out, it is necessary to apply sampling methods to deal with such imbalanceness before building the best parsimonious predictive model to predict faulty wafers.

abc123456
test pull request

# Solar_Energy_Prediction
  Being able to accurately predict the solar power hitting
the photovoltaic panels is a key challenge to integrate more
and more renewable energy sources into the grid, as the total
power generation needs to match the instantaneous consumption load. The solar power coming to our planet is predictable,
but the energy produced fluctuates with varying atmospheric
conditions. Usually, numerical weather prediction models are
used to make irradiation forecasts. This project focuses on
machine learning techniques to produce more accurate predictions for solar power (see figure 1).
Our strategy to make this prediction is:
- collect, understand and process the weather data,
- perform different machine learning techniques to make the
prediction,
- perform some feature engineering aside of the forecast features,
- analyze the results and discuss them.

#Gathering the data
The data (in netCDF4 format, very popular to manipulate
weather data) has been provided by International Solar Alliance in several
files:
1.weather data, as the values of 15 weather parameters (such as precipitation, maximum temperature, air
pressure, downward/upward short-wave radiative flux,...)
forecasted at 5 different hours of the day and provided
by 11 different ensemble forecast models. This data is
forecasted for a uniform spatial grid (16 × 9) centered on
Kamuthi Solar Power Plant and has been collected everyday from
1994 to 2007 (5113 days) for the training dataset and
from 2008 to 2012 (1400 days) for the testing dataset.
2.daily incoming solar energy data, as the total daily incoming solar energy at 98 Kamuthi Solar Power Plant sites (different from the grid points of the weather data from 1994
to 2007.

# Adapt the data to our needs
For each Kamuthi Solar Power Plant site, we have identified the four closest
GEFS weather sites. There are then two possible methods to use the data. For each Mesonet site, we
can either interpolate geographically the weather data from the
grid to the Kamuthi Solar Power Plant site, or factor all the features in our algorithms to make a prediction.



# Selecting the predictors
Because of the multiple dimensions of the weather data available, we have decided to make some grouping for the data. As
the output should be the daily expected solar power, for each
day, site and weather model, we have composed an array of the
15 weather parameters, taken for the 5 different timestamps of
the day and the 4 closest stations, which gave an array of 300
predictors for each given day, site and weather model.
After that, we we able to run algorithms on the data for each
day, site and model. In the weather prediction industry, these
models are usually equally weighted when running forecast
softwares. So, we have decided to average the power forecasts
from the different models to estimate their combined prediction. However, all the weather parameters are forecasted using
the same model, so, not to loose the correlation that we have
within the same model (11 models, 300 predictors), we could
not work simultaneously with all the models together (300×11
predictors). Therefore the steps chosen to run the algorithms:
1. take the average of each parameter on the 11 models
2. train one model over all the days and sites
3. for each site/day: estimate the incoming solar energy
At this stage, we had weather data for: 98 sites ×5113 days
×(75 + 1) parameters (distance Kamuthi Solar Power Plant site-GEFS) ×4 stations.
This boils down to: 76×4 = 304 features, and 98×5113 =
501074 samples, for 98×1796 = 176008 predictions to make.

#Understanding the data
Before running any algorithm on the massive dataset, we
wanted to get a grasp on the kind of influence some of the features had on the output. So, we took the weather parameters
that seemed the most meaningful to us and plotted heat maps.It is not the
only factor though. The West-East distribution of clouds with
the clouds being more frequent in the East, will also probably
have high negative correlation with the output. We will verify
later these correlations.Then, we also wanted to have some more quantitative preanalysis, by measuring the correlation between the factors and the response. Scatterplots were useful in giving a visual estimate of the kind of correlation between them: linear, polynomial, inverse...

#Regression methods
To be able to compare our approach to others’ (Kaggle leaderboard), we have used the MAE3
formula to calculate the error
The mean absolute error is commonly used by the renewable
energy industry to compare forecast performance. It does not
excessively punish extreme forecasts.
# Simple linear regression
We started with a simple linear regression to make our first
predictions. Hence the forecasted daily incoming solar energy
for each day and Mesonet site was:
To determine the coefficients θk we have trained our model
by minimizing
To assess the bias variance trade-off, we have divided the
training set into two subsets: the first one with the data from
1994 to 2006 (12 years) used as the training set, and the other
one with the data from 2007 to 2008 (2 years) used as the
validation set. We have trained different models by varying the
size of the training dataset and computed the corresponding
cross-validation error on the validation test. Then, we have
plotted both the training and test MAE of each model to show
the ”learning curve”, on figure 7.
The learning curve for the linear regression model shows the
evolution of the learning and testing errors. For 304 predictors,
the sample does not seem to be large enough below 10 years.
From 10 to 12 years of training samples, for a testing set of
2 years, MAEs converge and it seems that we train a model
without too much bias nor too much variance. So, it seems
that 12 years of data should be enough to train a model of 304
features.

# Random Forests
Then, we wanted to try more complex methods that could handle the large number of features and the highly non-linear and
complex relationship between the features and the response of
the data, that has been observed during the first visualizations
of the data. Tree-based methods (decision trees) seemed to be
a good match.
Random Forests builds a large number of decision trees by
generating different bootstrapped training data sets and averages all the predictions. But when building these trees, each
time a split in a tree is considered, a random sample of m predictors is chosen from the full set of p predictors. The classifiers may be weak predictors when used separately, but much
stronger when combined with other predictors. Randomness
allows weak predictors to be taken into account and uncorrelates the trees.
Two tuning parameters are needed to build a Random
Forests algorithm: the total number of trees generated and
the number of features randomly selected at each split when
building the trees. To determine optimal values for those two
parameters, we have run several cross validation models and
selected those which gave us the best results. We have started
with values around those given in the litterature[2] (Typically
a good value for m is √p which is around 17 in our case) and
explored the different learning curves given by models. Eventually, we ran our Random Forests algorithm with 15 predictors on 3000 trees.

# Models comparison
we can see that the most successful models are
given by using Random Forest methods (6% more accurate
than linear regression)

# Additional features
To help our predictions, we have tried other features: as Environmental Engineers, we know that the incoming solar radiation to the Earth heavily depends on two main parameters,
the time of the year and the location. But we have also added
other parameters:
1. time: the incoming solar energy high relies on the spatial
position of the Sun, which depends on the seasons, so we
have added a categorical feature to factor the month of
the prediction day.
2. location: the solar incident varies with the latitude, but we
have also added the longitude of the Mesonet site as we
have observed graphically that the irradiation also relies
on the longitude, even though this may not apply to other
places .
3. altitude: we have found a high correlation between longitude and altitude (and irradiation) in Kamuthi Solar Power Plant, so we
have also added the altitudes of the sites and GEFS stations.

# Conclusion
This machine learning project was our first hands-on experience with real big data. The project was about data preparation for a big part: it involved data understanding, sorting
and reframing. Then, we had to think about ways to run our
algorithm, as we needed machines capable of handling about
2GB of input data at once. It was challenging, but that made
us really think about ways to save time and resources: how to
reduce the computational load of our code, how relevant it is
to make backups of intermediate files, how useful it is to run
calibration test algorithms before launching codes that would
run for tens of hours.
It was definitely challenging to work with this big data. We
have also learnt a lot about implementing algorithms in real
life, as we were not working in a fully academic environmental
anymore.
And finally, as we tried to understand the different correlation relationships between the parameters and the forecasts, we
surprisingly also got a better understanding of solar prediction
from an energy engineer point of view.

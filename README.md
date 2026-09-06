Overview
Accurate short-term air-quality forecasts can support public warnings, pollution management, transport planning, and health-protection decisions. Fine particulate matter is particularly important because its concentration can change rapidly during pollution events.

In this competition, you will forecast the PM2.5 concentration one hour ahead across a network of urban air-quality monitoring stations.

Each row contains information available at the observation hour, including:

Current PM2.5 concentration Other pollutant measurements Temperature and atmospheric pressure Dew point and rainfall Wind direction and speed Monitoring-station identity Date and hour

The prediction target, PM2_5_next_hour, is the PM2.5 concentration recorded at the same station exactly one hour after the observation.

The competition data contain:

360,954 labelled training station-hours 51,063 hidden test station-hours 12 monitoring stations A chronological train-test split

The training data cover an earlier monitoring period, while the hidden test data come from a later period. Rows without a valid next-hour target have been excluded, but missing predictor values have been retained. Handling incomplete environmental measurements is therefore part of the challenge.

Because the split is chronological, random cross-validation may produce overly optimistic results. Time-aware validation is strongly recommended.

Submissions are evaluated using Root Mean Squared Error (RMSE):


where:

(N) is the number of observations in the hidden test set.
(y_i) is the observed next-hour PM2.5 concentration.
(\hat{y}_i) is the submitted next-hour PM2.5 prediction.
Lower scores are better.

RMSE gives greater weight to large prediction errors. This is relevant because substantially underestimating or overestimating a severe pollution event may affect public warnings and operational decisions.

Data:
id: unique competition identifier for a station and forecast target time.
observation_timestamp: hour at which predictors are observed.
station: monitoring-station name.
year, month, day, hour: components of the observation timestamp.
current_PM2_5: PM2.5 concentration at the observation hour (µg/m³).
PM10, SO2, NO2, CO, O3: contemporaneous pollutant measurements.
TEMP: temperature in degrees Celsius.
PRES: atmospheric pressure in hPa.
DEWP: dew-point temperature in degrees Celsius.
RAIN: precipitation in mm.
wd: wind direction.
WSPM: wind speed in m/s.
Missing predictor values are preserved as blank CSV fields.
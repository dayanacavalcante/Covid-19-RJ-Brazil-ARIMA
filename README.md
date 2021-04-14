# Covid-19 Rio de Janeiro, Brazil

### _Case Study_

Apply stationarity transformations to Covid-19 data in Rio de Janeiro, Brazil, and use Autoregressive Integrated Moving Average (ARIMA).

### _Preprocessing_

The data used include the period of _2020-01-13_ to _2021-03-23_.
```
       classificação_final dt_notific dt_inicio_sintomas bairro_resid_estadia  ... dt_evolucao  raca_cor Data_atualizacao sistema
0               CONFIRMADO 2020-09-18           9/3/2020            PACIENCIA  ...   9/22/2020     Preta        3/24/2021   SIVEP 
1               CONFIRMADO 2020-11-25          11/2/2020      BARRA DA TIJUCA  ...   1/12/2021    Branca        3/24/2021   SIVEP 
2               CONFIRMADO 2020-05-06           5/6/2020             CACHAMBI  ...   5/14/2020  Ignorado        3/24/2021   SIVEP 
3               CONFIRMADO 2020-11-12          11/2/2020      BARRA DA TIJUCA  ...  12/14/2020    Branca        3/24/2021   SIVEP 
4               CONFIRMADO 2020-06-13          4/26/2020      MARECHAL HERMES  ...   7/24/2020  Ignorado        3/24/2021   SIVEP 
...                    ...        ...                ...                  ...  ...         ...       ...              ...     ... 
220542          CONFIRMADO 2021-03-23          3/13/2021             FLAMENGO  ...         NaN    Branca        3/24/2021    ESUS 
220543          CONFIRMADO 2021-03-23           3/7/2021         BRAS DE PINA  ...   3/21/2021    Branca        3/24/2021    ESUS 
220544          CONFIRMADO 2021-03-23          3/22/2021               OLARIA  ...         NaN     Parda        3/24/2021    ESUS 
220545          CONFIRMADO 2021-03-23          3/10/2021     MAGALHAES BASTOS  ...   3/24/2021     Parda        3/24/2021    ESUS 
220546          CONFIRMADO 2021-03-23          3/22/2021      BARRA DA TIJUCA  ...         NaN    Branca        3/24/2021    ESUS 

[220547 rows x 12 columns]
```
![](/Charts/CasesPlot.png)

- Decomposing the Time Series data: with the _seasonal_decompose_ function, the series is broken down into trend, seasonality and residuals. It shows us a downward trend towards the end of the series. It is not seasonal. It presents a larger residual variation at the beginning of the year 2021.

![](/Charts/SeasonalDecompose_ts.png)

Most time series models work with the assumption that the time series is stationary. Which means that it's statistical properties are approximately constant over time. Statistical tests are used to confirm whether the series is stationary or not. In this case, I used the Dickey-Fuller Test.

![](/Charts/TestStationarity_1.png)

```
Results of Dickey-Fuller Test:
Test Statistic                  -2.895020
p-value                          0.045930
#Lags Used                      17.000000
Number of Observations Used    371.000000
Critical Value (1%)             -3.448100
Critical Value (5%)             -2.869362
Critical Value (10%)            -2.570937
dtype: float64
```

Understanding results of Dickey-Fuller Test:
- The H0 considers that the Time Series is not stationary;
- If the _p-value_ is below 5%: is stationary;
- If the Test Statistic is below any critical value: is stationary;

If the p value is high, it can indicate the presence of certain trends (variable average) or also seasonality, which is not the case. Despite rejecting H0, I worked on transforming the series into stationary, first analyzing the trend.

### _Transforms used for stationarizing data_

- Logarithmic transformation: basically it penalizes larger values more than smaller ones. Reduces the Trend.

![](/Charts/LogarithmicTransformation.png)

![](/Charts/TestStationarity2_Log.png)

```
Results of Dickey-Fuller Test:
Test Statistic                  -4.694037
p-value                          0.000086
#Lags Used                      16.000000
Number of Observations Used    372.000000
Critical Value (1%)             -3.448052
Critical Value (5%)             -2.869341
Critical Value (10%)            -2.570926
dtype: float64
```
With the Logarithmic Transformation the _p-value_ is well below 0.05 and the Statistical Test well below the critical values. The series can be considered to be stationary. 

- Smoothing can be used to model the Trend, using the Moving Average and Weighted Exponential Moving Average.

![](/Charts/SmoothingMovingAverage.png)

![](/Charts/TestStationarity3_Moving_Avg.png)

```
Test Statistic                  -3.899334
p-value                          0.002040
#Lags Used                      17.000000
Number of Observations Used    365.000000
Critical Value (1%)             -3.448394
Critical Value (5%)             -2.869491
Critical Value (10%)            -2.571006
dtype: float64
```

![](/Charts/SmoothingWEMA.png)

![](/Charts/TestStationarity4_WEMA.png)

```
Test Statistic                  -2.333482
p-value                          0.161397
#Lags Used                      14.000000
Number of Observations Used    374.000000
Critical Value (1%)             -3.447956
Critical Value (5%)             -2.869299
Critical Value (10%)            -2.570903
dtype: float64
```
Of the transformations made, the Weighted Exponential Moving Average had a worse performance so far because the _p-value_ is above 0.05 and the Statistical Test is also above the Critical Values.

- Remove trend and seasonality with differentiation.

![](/Charts/Differentiation.png)

![](/Charts/TestStationarity5_Diff.png)

```
Results of Dickey-Fuller Test:
Test Statistic                  -4.412541
p-value                          0.000282
#Lags Used                      17.000000
Number of Observations Used    370.000000
Critical Value (1%)             -3.448148
Critical Value (5%)             -2.869383
Critical Value (10%)            -2.570948
dtype: float64
```

### _Auto-Correlation Function (ACF) and Partial Auto-Correlation Function (PACF)_

With the ACF and PACF charts, it is possible to verify the values that will be used for the terms of the ARIMA model. You can see that the original data is not stationary and I transformed the data into stationary with differentiation.

![](/Charts/ACF.png)

![](/Charts/PACF.png)

### _ARIMA Model_

![](/Charts/ARIMA_Model(10,1,2).png)
```
ARIMA Model Results
==============================================================================
Dep. Variable:                D.Cases   No. Observations:                  271
Model:                ARIMA(10, 1, 2)   Log Likelihood                -171.447
Method:                       css-mle   S.D. of innovations              0.450
Date:                Tue, 13 Apr 2021   AIC                            370.895
Time:                        17:44:57   BIC                            421.325
Sample:                             1   HQIC                           391.143

==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
const              0.0246      0.016      1.563      0.118      -0.006       0.055
ar.L1.D.Cases     -0.3885      0.095     -4.087      0.000      -0.575      -0.202
ar.L2.D.Cases     -1.1039      0.090    -12.327      0.000      -1.279      -0.928
ar.L3.D.Cases     -0.6115      0.102     -5.983      0.000      -0.812      -0.411
ar.L4.D.Cases     -0.5067      0.102     -4.972      0.000      -0.706      -0.307
ar.L5.D.Cases     -0.3426      0.103     -3.326      0.001      -0.544      -0.141
ar.L6.D.Cases     -0.1499      0.105     -1.432      0.152      -0.355       0.055
ar.L7.D.Cases      0.2661      0.098      2.712      0.007       0.074       0.458
ar.L8.D.Cases      0.3879      0.090      4.305      0.000       0.211       0.565
ar.L9.D.Cases      0.3557      0.056      6.300      0.000       0.245       0.466
ar.L10.D.Cases     0.4966      0.057      8.782      0.000       0.386       0.607
ma.L1.D.Cases     -0.2269      0.102     -2.231      0.026      -0.426      -0.028
ma.L2.D.Cases      0.7315      0.078      9.407      0.000       0.579       0.884
                                    Roots
==============================================================================
                   Real          Imaginary           Modulus         Frequency
------------------------------------------------------------------------------
AR.1             1.1706           -0.0000j            1.1706           -0.0000
AR.2             0.6315           -0.7923j            1.0132           -0.1429
AR.3             0.6315           +0.7923j            1.0132            0.1429
AR.4             0.1350           -1.0338j            1.0426           -0.2293
AR.5             0.1350           +1.0338j            1.0426            0.2293
AR.6            -0.2395           -0.9981j            1.0265           -0.2875
AR.7            -0.2395           +0.9981j            1.0265            0.2875
AR.8            -0.8788           -0.6811j            1.1119           -0.3951
AR.9            -0.8788           +0.6811j            1.1119            0.3951
AR.10           -1.1834           -0.0000j            1.1834           -0.5000
MA.1             0.1551           -1.1588j            1.1692           -0.2288
MA.2             0.1551           +1.1588j            1.1692            0.2288
------------------------------------------------------------------------------
```

![](/Charts/ARIMA_Model(2,1,2).png)

```
ARIMA Model Results
==============================================================================
Dep. Variable:                D.Cases   No. Observations:                  271
Model:                 ARIMA(2, 1, 2)   Log Likelihood                -250.081
Method:                       css-mle   S.D. of innovations              0.606
Date:                Tue, 13 Apr 2021   AIC                            512.162
Time:                        17:54:19   BIC                            533.775
Sample:                             1   HQIC                           520.840

=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0237      0.016      1.485      0.137      -0.008       0.055
ar.L1.D.Cases     1.0838      0.049     22.088      0.000       0.988       1.180
ar.L2.D.Cases    -0.7396      0.048    -15.300      0.000      -0.834      -0.645
ma.L1.D.Cases    -1.5501      0.039    -40.072      0.000      -1.626      -1.474
ma.L2.D.Cases     0.8339      0.040     21.041      0.000       0.756       0.912
                                    Roots
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
AR.1            0.7327           -0.9029j            1.1628           -0.1415
AR.2            0.7327           +0.9029j            1.1628            0.1415
MA.1            0.9294           -0.5791j            1.0951           -0.0887
MA.2            0.9294           +0.5791j            1.0951            0.0887
-----------------------------------------------------------------------------
```
With the ARIMA Model Results (2,1,2) it had a higher AIC value than the ARIMA Model Results (10,1,2) I did not make the predictions with this model.

#### _Data used in the study_

The data were taken from the site: https://www.data.rio/datasets/f314453b3a55434ea8c8e8caaa2d8db5








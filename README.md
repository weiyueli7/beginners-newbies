# beginners-newbies



## Data Cleaning & Preprocessing

First, we noticed that the data set has 29 columns:

```
['YEAR', 'SERIAL', 'MONTH', 'HWTFINL', 'CPSID', 'ASECFLAG', 'HFLAG', 'ASECWTH', 'REGION', 'STATEFIP', 'NFAMS', 'PERNUM', 'WTFINL', 'CPSIDP', 'ASECWT', 'AGE', 'SEX', 'RACE', 'MARST', 'BPL', 'EMPSTAT', 'OCC', 'UHRSWORKT', 'WKSTAT', 'JOBCERT', 'EDUC', 'EDDIPGED', 'INCWAGE', 'OINCWAGE']
```

where the `INCWAGE` column describes the salaries we are interested in. 

### 1. Drop surveys with useless income wage

But after looking at values in the `INCWAGE` column, we found many `nan`s and several weird values like `99999999` (according to the IPUMS source, `99999999` stands for "not in universe"). 

```
total surveys: 54737
number of N.I.U wage: 5543
number of NaN wage: 30160
number of 0 wage: 7496
number of wage > 0: 11538
number of wage >= 0: 19034
```

We exclude these data from our dataframe since they cannot help us make predictions: (`data` is the raw data set, `NIU` equals to 99999999)

```python
filtered = data[(data['INCWAGE'] >= 0) & (data['INCWAGE'] < NIU)].reset_index(drop=True)
```

### 2. Adjust income for inflation

> https://cps.ipums.org/cps/cpi99.shtml

We need to inflate or deflate the dollar amounts of `INCWAGE` in order to make them comparable, so we convert the dollar amounts to constant 1999 dollars by multiplying the CPI99 constants of that data year to the `INCWAGE` values:

We first record these constants in a static `.json` file so that it can be easily loaded as a dictionary:

```python
with open(data_pth / 'cpi99_cons.json', 'w') as fout:
    json.dump(cpi99_cons, fout)
```

Then, generate a series of the corresponding CPI99 constants of that data year and multiply the constants to the raw dollar amounts of `INCWAGE`:

```python
def clean_incwage(df):
    cpi99 = df['YEAR'].apply(lambda yr : cpi99_cons[str(yr)])
    df['inc_cpi99'] = df['INCWAGE'] * df['cpi99']

    df.groupby('YEAR').mean()['INCWAGE'].plot(legend=True)
    df.groupby('YEAR').mean()['inc_cpi99'].plot(legend=True)
    
clean_wage(filtered)
```

![](./util/incwage_cpi99.png)

As shown, the adjusted values are more stable.

### 3. Choose & Adjust appropriate factors

Then, after exploring these columns and the descriptions (e.g. https://cps.ipums.org/cps-action/variables/BPL#description_section), we decided to focus on columns as below:

```
['REGION', 'RACE', 'AGE', 'SEX', 'EMPSTAT', 'OCC', 'UHRSWORKT', 'WKSTAT', 'EDUC']
```

![](./util/factors_vs_inc.png)

#### 3. Age


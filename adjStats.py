import pandas as pd
from sklearn import linear_model
## @drmartylawrenc1
## https://github.com/drmartylawrence

### --------------------------------------------------------------------------
### PART 1 - SET UP VARIABLE NAMES
### --------------------------------------------------------------------------

inFile1 = 'rawGameTeam2020.csv' # csv containing the game by game data of interest
inFile2 = 'rawSeasTeam2020.csv' # csv containing the season level data we want to group by
                                  # Ie, one column per team/player  
outFile = 'adjSeasTeam2020.csv' # Name of the output csv

# Dataframe of game-by-game we will use in the opponent adjustment
dfGame = pd.read_csv(inFile1)
dfSeas = pd.read_csv(inFile2)

# DF column names to help guide opponent adjustment function
xStr = 'team' # Column of interest, either the team/player we want to adjust
hfaStr = 'hfa' # Homefield Advantage column name
oppStr = 'opponent' # Opponent column name

# Column names for stats we want to opponent-adjust:
statStr = [
    'offPpa',
    'offPassPpa',
    'offRushPpa',
    'defPpa',
    'defPassPpa',
    'defRushPpa'
    ] 

# statStr = [
#     'allPPA',
#     'passPPA',
#     'rushPPA',
#     ] 

### --------------------------------------------------------------------------
### Part 2 - ADJUSTED STATS FUNCTION
### --------------------------------------------------------------------------
## Callable function 'adjFunc' to perform the opponent adjustment
# Input1 is 'df' a dataframe with the raw game by game stats
# Input2 is 'stat' which is a string for the raw game df column we adjust on
def adjFunc(df, stat):
        # Create dummy variables for each Team/Opponent, plus Homefield Advantage
    dfDummies = pd.get_dummies(df[[xStr, hfaStr, oppStr]])
    
    # Hyperparameter tuning if you want it. I've found a value between 1-2 works best
    rdcv = linear_model.RidgeCV(alphas = [1,1.5,2], fit_intercept = True)
    rdcv.fit(dfDummies,df[stat]);
    alf = rdcv.alpha_
    
    # Or set Alpha directly here
    # alf = 1
    
    # Set up ridge regression model parameters
    reg = linear_model.Ridge(alpha = alf, fit_intercept = True)
    

    # Run the regression
    # X values in the regression will be dummy variables each Team/Opponent, plus Homefield Advantage
    # y values will be the raw value from each game for the specific stat we're adjusting
    reg.fit(
        X = dfDummies,
        y = df[stat]
        )
    
    # Extract regression coefficients
    dfRegResults = pd.DataFrame({
        'coef_name': dfDummies.columns.values,
        'ridge_reg_coef': reg.coef_})
    
    # Add intercept back in to reg coef to get 'adjusted' value
    dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef']+reg.intercept_)
    
    #Print the HFA and Alpha values
    print('Homefield Advantage for: '+stat+' (alpha: '+str(alf)+')')
    print('{:.3f}'.format(dfRegResults[dfRegResults['coef_name'] == hfaStr]['ridge_reg_coef'][0]))
    
    # Only keep ratings from 'team/name' perspective
    # Ie drop the inverse (opponent) side of the results
    dfAdjStat = (dfRegResults[dfRegResults['coef_name'].str.slice(0, len(xStr)) == xStr].
      rename(columns = {"ridge_reg_value": 'adj_'+stat}).
      reset_index(drop = True))
    dfAdjStat['coef_name'] = dfAdjStat['coef_name'].str.replace(xStr+'_','')
    dfAdjStat = dfAdjStat.drop(columns=['ridge_reg_coef'])
    # Return a column representing the data of interest
    return(dfAdjStat)


### --------------------------------------------------------------------------
### PART 3 - REGRESS THE RAW DATA
### --------------------------------------------------------------------------

# Loop throug the aray of strings, call the opponent adjustment function,  
# and append to the season df
for i in range(0,len(statStr)):
    df = dfGame[[xStr,hfaStr,oppStr,statStr[i]]]
    df = df.dropna()
    adjResult = adjFunc(df, statStr[i])
    dfSeas = dfSeas.join(adjResult.set_index('coef_name'), on=xStr)


# ### --------------------------------------------------------------------------
# ### PART 4 - OUTPUT
# ### --------------------------------------------------------------------------
print(dfSeas)
dfSeas.to_csv(outFile, index=False)
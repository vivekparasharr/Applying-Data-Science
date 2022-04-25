
y = dependent variable
x = independent variables

y = b0 + b1.x1 # Simple linear regression
y = b0 + b1.x1 + b2.x2 + .. + bn.xn # Multiple Linear Regression

'''
Assumptions of linear regression
Before we test the assumptions, we’ll need to fit our linear regression models
1. linearity
2. homoskedasticity
3. multivariate normality
4. independence of errors
5. lack of multicollinearity
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

# Importing the dataset
dataset = pd.read_csv('Data/Ecommerce_Customers.csv')

# Explore the dataset
dataset.info()
dataset.Address

# Visualize the dataset
sns.pairplot(dataset) # hue=None, palette=None, vars=[None], x_vars=[None], y_vars=[None], dropna=True, kind = 'reg' to find linearity, 'scatter' is default 
# Here the Length of Membership seems to have a linear relationship with Yearlt amount spent


# One thing to note is that I’m assuming outliers have been removed 

# Manipulate the dataset (add new feaures based on data, etc.)
# Extract state from address
def str_split(row):
    return row.Address.split(' ')[-2]
dataset['state'] = dataset.apply(str_split, axis=1) # apply the function to the dataset
dataset.groupby('state').count()[['Address']].index # all are states so the function worked

# defining x and y variables of the linear regression equation
X = dataset.iloc[:, [3,4,5,6]].values
y = dataset.iloc[:, 7].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# 4 because we are building the transformer based on 4th column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the model
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Returning the R^2 for the model
# .score() automatically scores y_train_pred, and compares those with y_train to calculate the R^2
regressor_r2 = regressor.score(X_train, y_train) 
print('R^2:', regressor_r2)
# Instead of using .score() we could also use r2_score()
# But we will first need to manually score y_train_pred values
from sklearn.metrics import r2_score
y_train_pred = regressor.predict(X_train)
r2 = r2_score(y_train, y_train_pred)
print('R^2:', r2)

# Predicting the Test set results
y_test_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_test_pred.reshape(len(y_test_pred),1), y_test.reshape(len(y_test),1)),1))


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    return df_results

'''
LinearityPermalink
- This assumes that there is a linear relationship between the predictors (e.g. independent variables or features) and the response variable (e.g. dependent variable or label). This also assumes that the predictors are additive.
- Why it can happen: There may not just be a linear relationship among the data. Modeling is about trying to estimate a function that explains a process, and linear regression would not be a fitting estimator (pun intended) if there is no linear relationship.
- What it will affect: The predictions will be extremely inaccurate because our model is underfitting. This is a serious violation that should not be ignored.
- How to detect it: If there is only one predictor, this is pretty easy to test with a scatter plot. Most cases aren’t so simple, so we’ll have to modify this by using a scatter plot to see our predicted values versus the actual values (in other words, view the residuals). Ideally, the points should lie on or around a diagonal line on the scatter plot.
- How to fix it: Either adding polynomial terms to some of the predictors or applying nonlinear transformations . If those do not work, try adding additional variables to help capture the relationship between the predictors and the label.
'''
def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
           'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()

linear_assumption(regressor, X_train, y_train)
'''
How to read the output
- The predicted values are equally distributed above and below the diagonal, hence the relationship seems linear 
- If there were a higher concentration of predicted values below the diagonal, then we say the predictions are biased towards lower values 
- If there were a higher concentration of predicted values above the diagonal, then we say the predictions are biased towards higher values 
'''

'''
Normality of the Error TermsPermalink
- More specifically, this assumes that the error terms of the model are normally distributed. Linear regressions other than Ordinary Least Squares (OLS) may also assume normality of the predictors or the label, but that is not the case here.
- Why it can happen: This can actually happen if either the predictors or the label are significantly non-normal. Other potential reasons could include the linearity assumption being violated or outliers affecting our model.
- What it will affect: A violation of this assumption could cause issues with either shrinking or inflating our confidence intervals.
- How to detect it: There are a variety of ways to do so, but we’ll look at both a histogram and the p-value from the Anderson-Darling test for normality.
- How to fix it: It depends on the root cause, but there are a few options. Nonlinear transformations of the variables, excluding specific variables (such as long-tailed variables), or removing outliers may solve this problem.
'''
def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')

normal_errors_assumption(regressor, X_train, y_train)
# Output seems normal so assumption satisfied

'''
No Multicollinearity among PredictorsPermalink
- This assumes that the predictors used in the regression are not correlated with each other. This won’t render our model unusable if violated, but it will cause issues with the interpretability of the model.
- Why it can happen: A lot of data is just naturally correlated. For example, if trying to predict a house price with square footage, the number of bedrooms, and the number of bathrooms, we can expect to see correlation between those three variables because bedrooms and bathrooms make up a portion of square footage.
- What it will affect: Multicollinearity causes issues with the interpretation of the coefficients. Specifically, you can interpret a coefficient as “an increase of 1 in this predictor results in a change of (coefficient) in the response variable, holding all other predictors constant.” This becomes problematic when multicollinearity is present because we can’t hold correlated predictors constant. Additionally, it increases the standard error of the coefficients, which results in them potentially showing as statistically insignificant when they might actually be significant.
- How to detect it: There are a few ways, but we will use a heatmap of the correlation as a visual aid and examine the variance inflation factor (VIF).
- How to fix it: This can be fixed by other removing predictors with a high variance inflation factor (VIF) or performing dimensionality reduction.
'''
def multicollinearity_assumption(model, features, label, feature_names=None):
    """
    Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                       correlation among the predictors, then either remove prepdictors with high
                       Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           
                       This assumption being violated causes issues with interpretability of the 
                       coefficients and the standard errors of the coefficients.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print('Assumption 3: Little to no multicollinearity among predictors')
        
    # Plotting the heatmap
    plt.figure(figsize = (10,8))
    sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
    plt.title('Correlation of Variables')
    plt.show()
        
    print('Variance Inflation Factors (VIF)')
    print('> 10: An indication that multicollinearity may be present')
    print('> 100: Certain multicollinearity among the variables')
    print('-------------------------------------')
       
    # Gathering the VIF for each variable
    VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
    for idx, vif in enumerate(VIF):
        print('{0}: {1}'.format(feature_names[idx], vif))
        
    # Gathering and printing total cases of possible or definite multicollinearity
    possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
    definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
    print()
    print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
    print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
    print()

    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print('Assumption satisfied')
        else:
            print('Assumption possibly satisfied')
            print()
            print('Coefficient interpretability may be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')

    else:
        print('Assumption not satisfied')
        print()
        print('Coefficient interpretability will be problematic')
        print('Consider removing variables with a high Variance Inflation Factor (VIF)')


# Additional variable being created to test multicollinearity
regressor_feature_names = np.array(['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'])

multicollinearity_assumption(regressor, X_train, y_train, regressor_feature_names)


'''
No Autocorrelation of the Error TermsPermalink
- This assumes no autocorrelation of the error terms. Autocorrelation being present typically indicates that we are missing some information that should be captured by the model.
- Why it can happen: In a time series scenario, there could be information about the past that we aren’t capturing. In a non-time series scenario, our model could be systematically biased by either under or over predicting in certain conditions. Lastly, this could be a result of a violation of the linearity assumption.
- What it will affect: This will impact our model estimates.
- How to detect it: We will perform a Durbin-Watson test to determine if either positive or negative correlation is present. Alternatively, you could create plots of residual autocorrelations.
- How to fix it: A simple fix of adding lag variables can fix this problem. Alternatively, interaction terms, additional variables, or additional transformations may fix this.
'''
def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    from statsmodels.stats.stattools import durbin_watson
    print('Assumption 4: No Autocorrelation', '\n')
    
    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    print('-------------------------------------')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')

autocorrelation_assumption(regressor, X_train, y_train)

'''
Homoscedasticity
- This assumes homoscedasticity, which is the same variance within our error terms. Heteroscedasticity, the violation of homoscedasticity, occurs when we don’t have an even variance across the error terms.
- Why it can happen: Our model may be giving too much weight to a subset of the data, particularly where the error variance was the largest.
- What it will affect: Significance tests for coefficients due to the standard errors being biased. Additionally, the confidence intervals will be either too wide or too narrow.
- How to detect it: Plot the residuals and see if the variance appears to be uniform.
- How to fix it: Heteroscedasticity (can you tell I like the scedasticity words?) can be solved either by using weighted least squares regression instead of the standard OLS or transforming either the dependent or highly skewed variables. Performing a log transformation on the dependent variable is not a bad place to start.
'''
def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  

# Plotting the residuals of our ideal dataset:
homoscedasticity_assumption(regressor, X_train, y_train)
# residuals have relative constant variance




#############################################################################
#######################  Code for the Master Function #######################
#############################################################################

def linear_regression_assumptions(features, label, feature_names=None):
    """
    Tests a linear regression on the model to see if assumptions are being met
    """
    from sklearn.linear_model import LinearRegression
    
    # Setting feature names to x1, x2, x3, etc. if they are not defined
    if feature_names is None:
        feature_names = ['X'+str(feature+1) for feature in range(features.shape[1])]
    
    print('Fitting linear regression')
    # Multi-threading if the dataset is a size where doing so is beneficial
    if features.shape[0] < 100000:
        model = LinearRegression(n_jobs=-1)
    else:
        model = LinearRegression()
        
    model.fit(features, label)
    
    # Returning linear regression R^2 and coefficients before performing diagnostics
    r2 = model.score(features, label)
    print()
    print('R^2:', r2, '\n')
    print('Coefficients')
    print('-------------------------------------')
    print('Intercept:', model.intercept_)
    
    for feature in range(len(model.coef_)):
        print('{0}: {1}'.format(feature_names[feature], model.coef_[feature]))

    print('\nPerforming linear regression assumption testing')
    
    # Creating predictions and calculating residuals for assumption tests
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    
    def linear_assumption():
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a polynomial term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)
        
        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        print('If non-linearity is apparent, consider adding a polynomial term')
        
        
    def normal_errors_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
               
        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: The error terms are normally distributed')
        print()
    
        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        p_value = normal_ad(df_results['Residuals'])[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
    
        # Plotting the residuals distribution
        plt.subplots(figsize=(12, 6))
        plt.title('Distribution of Residuals')
        sns.distplot(df_results['Residuals'])
        plt.show()
    
        print()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')
        
        
    def multicollinearity_assumption():
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either remove prepdictors with high
                           Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           
                           This assumption being violated causes issues with interpretability of the 
                           coefficients and the standard errors of the coefficients.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('\n=======================================================================================')
        print('Assumption 3: Little to no multicollinearity among predictors')
        
        # Plotting the heatmap
        plt.figure(figsize = (10,8))
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
        plt.title('Correlation of Variables')
        plt.show()
        
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
       
        # Gathering the VIF for each variable
        VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
        
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
            else:
                print('Assumption possibly satisfied')
                print()
                print('Coefficient interpretability may be problematic')
                print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        else:
            print('Assumption not satisfied')
            print()
            print('Coefficient interpretability will be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        
        
    def autocorrelation_assumption():
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a pattern that is not explained due to
                         the current value being dependent on the previous value.
                         This may be resolved by adding a lag variable of either the dependent
                         variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('\n=======================================================================================')
        print('Assumption 4: No Autocorrelation')
        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        durbinWatson = durbin_watson(df_results['Residuals'])
        print('Durbin-Watson:', durbinWatson)
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        else:
            print('Little to no autocorrelation', '\n')
            print('Assumption satisfied')

            
    def homoscedasticity_assumption():
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('\n=======================================================================================')
        print('Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        
        # Plotting the residuals
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show() 
        print('If heteroscedasticity is apparent, confidence intervals and predictions will be affected')
        
        
    linear_assumption()
    normal_errors_assumption()
    multicollinearity_assumption()
    autocorrelation_assumption()
    homoscedasticity_assumption()


#############################################################################
########################  Code for Model Validation  ########################
#############################################################################
from sklearn.model_selection import RepeatedKFold


#############################################################################
########################  Code for Model Evaluation  ########################
#############################################################################
'''
sklearn package provides various model evaluation metrics. Following are the important ones:
- Max_error
- Mean Absolute Error
- Mean Squared Error
- Median Squared Error
- R Squared
'''

def model_evaluation(original_data,predicted_data):
    # Max_error
    from sklearn.metrics import max_error
    me = max_error(original_data,predicted_data)
    print('1. Max_error: \t\t\t',me)
    # Mean Absolute Error
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(original_data,predicted_data)
    print('2. Mean absolute error: \t',mae)
    # Mean Squared Error
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(original_data,predicted_data)
    print('3. Mean Squared Error: \t\t',mse)
    # Median Squared Error
    from sklearn.metrics import median_absolute_error
    med_ae = median_absolute_error(original_data,predicted_data)
    print('4. Median Squared Error: \t',med_ae)
    # R Squared
    from sklearn.metrics import r2_score
    r2 = r2_score(original_data,predicted_data)
    print('5. R Squared: \t\t\t',r2)

model_evaluation(y_test, y_test_pred)


#############################################################################
#####################  Overfitting: R^2 is too high  ########################
#############################################################################
# One of the most common problems that you’ll encounter when building models is 
# multicollinearity. This occurs when two or more predictor variables in a dataset
# are highly correlated.

import seaborn as sns
sns.get_dataset_names()
tips = sns.load_dataset("tips")
tips.plot(x='total_bill', y='tip', kind='scatter')

'''
Methods for correlation analyses:
1. Parametric Correlation : It measures a linear dependence between two variables 
(x and y) is known as a parametric correlation test because it depends on the 
distribution of the data.
'''
# a. Pearson correlation
from scipy.stats import pearsonr 
corr, _ = pearsonr(tips.total_bill, tips.tip)   # Apply the pearsonr() 
print('Pearsons correlation: %.5f' % corr) 

'''
2. Non-Parametric Correlation: Kendall(tau) and Spearman(rho), which are 
rank-based correlation coefficients, are known as non-parametric correlation.
'''
# a. Kendall
from scipy.stats import kendalltau 
corr, _ = kendalltau(tips.total_bill, tips.tip) 
print('Kendall Rank correlation: %.5f' % corr) 
# b. Spearman
from scipy.stats import spearmanr
corr, _ = spearmanr(tips.total_bill, tips.tip) 
print('Spearman Rank correlation: %.5f' % corr) 


# Calculate the correlation matrix and VIF values for the predictor variables

# correlation matrix (based on Pearson correlation coefficient)
tips.corr()
tips.corr().round(3)
tips.corr().style.background_gradient(cmap='coolwarm') # cmap='RdYlGn', 'bwr', 'PuOr', 


# VIF: high VIF values (some texts define a “high” VIF value as 5 while others use 10) indicate multicollinearity
#create dataset
df = pd.DataFrame({'rating': [90, 85, 82, 88, 94, 90, 76, 75, 87, 86],
                   'points': [25, 20, 14, 16, 27, 20, 12, 15, 14, 19],
                   'assists': [5, 7, 7, 8, 5, 7, 6, 9, 9, 5],
                   'rebounds': [11, 8, 10, 6, 6, 9, 6, 10, 10, 7]})
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
#find design matrix for linear regression model using 'rating' as response variable 
y, X = dmatrices('rating ~ points+assists+rebounds', data=df, return_type='dataframe')
#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns
#view VIF for each explanatory variable 
vif
# How to Interpret VIF Values:
# A value of 1 indicates there is no correlation between a given explanatory variable and any other explanatory variables in the model.
# A value between 1 and 5 indicates moderate correlation between a given explanatory variable and other explanatory variables in the model, but this is often not severe enough to require attention.
# A value greater than 5 indicates potentially severe correlation between a given explanatory variable and other explanatory variables in the model. In this case, the coefficient estimates and p-values in the regression output are likely unreliable.


#############################################################################
###############  Avoid Overfitting using Subset Selection  ##################
#############################################################################

'''
Best subset selection 
Given a set of p total predictor variables, there are 2^p models that we could 
potentially build. One method that we can use to pick the best model is known 
as best subset selection and it works as follows:
1. Let M0 denote the null model, which contains no predictor variables. 
2. For k = 1, 2, … p:
    Fit all pCk models that contain exactly k predictors.
    Pick the best among these pCk models and call it Mk. Define “best” as the model 
    with the highest R2 or equivalently the lowest RSS.
3. Select a single best model from among M0…Mp using cross-validation prediction 
error, Cp, BIC, AIC, or adjusted R2.
Note that for a set of p predictor variables, there are 2p possible models. 

Criteria for Choosing the “Best” Model
1. Cp: (RSS+2dσ̂) / n
2. AIC: (RSS+2dσ̂2) / (nσ̂2)
3. BIC: (RSS+log(n)dσ̂2) / n
4. Adjusted R2: 1 – ( (RSS/(n-d-1)) / (TSS / (n-1)) )
where:
    d: The number of predictors
    n: Total observations
    σ̂: Estimate of the variance of the error associate with each response measurement in a regression model
    RSS: Residual sum of squares of the regression model
    TSS: Total sum of squares of the regression model
'''

'''
Stepwise selection - there are two approaches: forward and backward stepwise
Forward stepwise selection works as follows:
1. Let M0 denote the null model, which contains no predictor variables. 
2. For k = 0, 2, … p-1:
    Fit all p-k models that augment the predictors in Mk with one additional predictor variable.
    Pick the best among these p-k models and call it Mk+1. Define “best” as the model with the highest R2 or equivalently the lowest RSS.
3. Select a single best model from among M0…Mp using cross-validation prediction error, Cp, BIC, AIC, or adjusted R2.

Backward stepwise selection works as follows:
1. Let Mp denote the full model, which contains all p predictor variables. 
2. For k = p, p-1, … 1:
    Fit all k models that contain all but one of the predictors in Mk, for a total of k-1 predictor variables.
    Pick the best among these k models and call it Mk-1. Define “best” as the model with the highest R2 or equivalently the lowest RSS.
3. Select a single best model from among M0…Mp using cross-validation prediction error, Cp, BIC, AIC, or adjusted R2.
'''


#############################################################################
###############  Avoid Overfitting using Regularization  ####################
#############################################################################

# Regularization can be achieved using either of the following two models
# Ridge Regression
# Lasso Regression

# Original model - ordinary least square regression model
# Y = β0 + β1X1 + β2X2 + … + βpXp + ε
# Y: The response variable
# Xj: The jth predictor variable
# βj: The average effect on Y of a one unit increase in Xj, holding all other predictors fixed
# ε: The error term
# The values for β0, β1, B2, … , βp are chosen using the least square method, which minimizes the sum of squared residuals (RSS)
# least squares regression tries to find coefficient estimates that minimize the sum of squared residuals (RSS):
# RSS = Σ(yi – ŷi)^2
# yi: The actual response value for the ith observation
# ŷi: The predicted response value based on the multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Without Regularization -> r2 for training set:',regressor.score(X_train, y_train),'and test set:',regressor.score(X_test, y_test)) 

# Ridge regression - Regularizing the linear model
# The basic idea of ridge regression is to introduce a little bias so that the variance can be substantially reduced, which leads to a lower overall MSE.
# ridge regression, seeks to minimize the following:
# RSS + λΣβj2
# where j ranges from 1 to p and λ ≥ 0
# This second term in the equation is known as a shrinkage penalty.
# When λ = 0, this penalty term has no effect and ridge regression produces the same coefficient estimates as least squares. However, as λ approaches infinity, the shrinkage penalty becomes more influential and the ridge regression coefficient estimates approach zero.
# In general, the predictor variables that are least influential in the model will shrink towards zero the fastest.
# Regularization, significantly reduces the variance of the model, without substantial increase in its bias.
# Using Ridge()
from sklearn.linear_model import Ridge
#alpha =0.5
ridge_reg=Ridge(alpha=0.5,normalize=True)
ridge_reg.fit(X_train,y_train)
print('Ridge Regres alpha=0.5 -> r2 for training set:',ridge_reg.score(X_train, y_train),'and test set:',ridge_reg.score(X_test, y_test)) 
#alpha =1
ridge_reg=Ridge(alpha=1,normalize=True)
ridge_reg.fit(X_train,y_train)
print('Ridge Regressi alpha=1 -> r2 for training set:',ridge_reg.score(X_train, y_train),'and test set:',ridge_reg.score(X_test, y_test)) 
#alpha =2
ridge_reg=Ridge(alpha=2,normalize=True)
ridge_reg.fit(X_train,y_train)
print('Ridge Regressi alpha=2 -> r2 for training set:',ridge_reg.score(X_train, y_train),'and test set:',ridge_reg.score(X_test, y_test)) 

# Using RidgeCV() - https://www.statology.org/ridge-regression-in-python/
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#define model
ridge_reg = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
#fit model
ridge_reg.fit(X_train, y_train)
#display lambda that produced the lowest test MSE
print('Ridge Regression -> best alpha (that produced the lowest test MSE):', ridge_reg.alpha_)
# R^2 of the model 
print('Ridge Regression -> r2 for training set:',ridge_reg.score(X_train, y_train),'and test set:',ridge_reg.score(X_test, y_test)) 


# Lasso Regression - use to fit a regression model when multicollinearity is present in the data
# lasso regression seeks to minimize the following:
# RSS + λΣ|βj|
# where j ranges from 1 to p predictor variables and λ ≥ 0.
# This second term in the equation is known as a shrinkage penalty. 
# We select a value for λ that produces the lowest possible test MSE (mean squared error).
# Note: “alpha” is used instead of “lambda” in Python
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold # use the RepeatedKFold() function to perform k-fold cross-validation to find the optimal alpha value to use for the penalty term
#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#define model
lasso_reg = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)
#fit model
lasso_reg.fit(X_train, y_train)
#display lambda that produced the lowest test MSE
print('Lasso Regression -> best alpha (that produced the lowest test MSE):', lasso_reg.alpha_)
# R^2 of the model 
print('Lasso Regression -> r2 for training set:',lasso_reg.score(X_train, y_train),'and test set:',lasso_reg.score(X_test, y_test)) 



#############################################################################
########  Deal with multicollinearity using dimension reduction  ############
####################  principal components regression  ######################
#############################################################################

# principal components regression, which finds M linear combinations (known as “principal components”) of the original p predictors and then uses least squares to fit a linear regression model using the principal components as predictors.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# We’ll use hp as the response variable and the following variables as the predictors: mpg, disp, drat, wt, qsec
data_full = pd.read_csv('https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv')
data = data_full[["mpg", "disp", "drat", "wt", "qsec", "hp"]]
'''
Fit the PCR Model:
1. pca.fit_transform(scale(X)): This tells Python that each of the predictor variables should be scaled to have a mean of 0 and a standard deviation of 1. This ensures that no predictor variable is overly influential in the model if it happens to be measured in different units.
2. cv = RepeatedKFold(): This tells Python to use k-fold cross-validation to evaluate the performance of the model. For this example we choose k = 10 folds, repeated 3 times.
'''
#define predictor and response variables
X = data[["mpg", "disp", "drat", "wt", "qsec"]]
y = data[["hp"]]
#scale predictor variables
pca = PCA()
X_reduced = pca.fit_transform(scale(X))
#define cross validation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

regr = LinearRegression()
mse = []
# Calculate MSE with only the intercept
score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
mse.append(score)
# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, 6):
    score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
# Plot cross-validation results    
plt.plot(mse) # The plot displays the number of principal components along the x-axis and the test MSE (mean squared error) along the y-axis.
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('hp')
# Interpret the plot: From the plot we can see that the test MSE decreases by 
# adding in two principal components, yet it begins to increase as we add more 
# than two principal components. Thus, the optimal model includes just the first 
# two principal components.

# We can also use the following code to calculate the percentage of variance in the response variable explained by adding in each principal component to the model:
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# Interpreting the result:
# By using just the first principal component, we can explain 69.83% of the variation in the response variable.
# By adding in the second principal component, we can explain 89.35% of the variation in the response variable.

#Use the Final Model to Make Predictions
#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 
#scale the training and testing data
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:1]
#train PCR model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:1], y_train)
#calculate RMSE
pred = regr.predict(X_reduced_test)
np.sqrt(mean_squared_error(y_test, pred))
# RMSE turns out to be 40.2096. This is the average deviation between the 
# predicted value for hp and the observed value for hp for the observations 
# in the testing set.




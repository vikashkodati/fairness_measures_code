'''
Created on Jun 13, 2017

@author: meike.zehlike
'''
from scipy.stats import stats, norm
from scipy.stats.stats import ttest_ind
import statsmodels.api as sm


def t_test_ind(dataset, target_col, protected_col, equal_var=0):
    """
    corresponds to difference of means test

    performs the independent two-sample t-Test, or Welch's test if equality of the variances is not
    given

    @param dataset:
    @param target_col:      name of the column that contains the classifier results
    @param protected_col:   name of the column that contains the protection status
    @param equal_var:       if True, perform a standard independent 2 sample test that
                            assumes equal population variances and sample size. If False (default), perform Welch’s t-test,
                            which does not assume equal population variance

    @return: calculated t-statistic and two-tailed p-value

    """
    protected_targets = dataset.get_all_targets_of_group(target_col, protected_col, 1)
    nonprotected_targets = dataset.get_all_targets_of_group(target_col, protected_col, 0)
    return ttest_ind(protected_targets, nonprotected_targets, equal_var)


def fisher_exact_two_groups(dataset, target_col, protected_col):
    """
    Performs a Fisher exact test on a 2x2 contingency table as in scipy.stats.fisher_exact_two_groups()

    @param dataset:
    @param target_col:      name of the column that contains the classifier results
    @param protected_col:   name of the column that contains the protection status

    @return: odds ratio and related p-value
    """
    positive_protected = dataset.count_classification_and_category(target_col, protected_col, group=1, accepted=1)
    negative_protected = dataset.count_classification_and_category(target_col, protected_col, group=1, accepted=0)
    positive_nonprotected = dataset.count_classification_and_category(target_col, protected_col, group=0, accepted=1)
    negative_nonprotected = dataset.count_classification_and_category(target_col, protected_col, group=0, accepted=0)

    contingency_table = [[positive_protected, negative_protected], [positive_nonprotected, negative_nonprotected]]

    return stats.fisher_exact(contingency_table)


def regression_slope_test(dataset, target_col, protected_col):
    """
    Performs Ordinary Least Squares using statsmodels
    TODO

    @param dataset:
    @param target_col:      name of the column that contains the classifier results
    @param protected_col:   name of the column that contains the protection status

    @return: t-statistic for protected columns
    """

    #X will either be a numpy array or a pandas data frame with shape (n, p) where n is the number of data points
    #and p is the number of predictors (protected variables). y is either a one-dimensional numpy array or
    #a pandas series of length n.
    y = dataset.data[target_col]
    X = dataset.data[protected_col]

    #add a constant term to fit the intercept of the linear model
    X = sm.add_constant(X)

    est = sm.OLS(y, X)
    results = est.fit()
    #print(results.summary())
    return results.tvalues[protected_col]

def two_proportion_z_test(dataset, target_col, protected_col):
    """
    Performs Difference in proportions for two groups, using the two proportions z-test
    TODO

    @param dataset:
    @param target_col:      name of the column that contains the classifier results
    @param protected_col:   name of the column that contains the protection status

    @return: test statistic and p-value for the z-test
    """

    #number of success in nob trials, get the number of successes for each group
    group_one = dataset.data[target_col].loc[dataset.data[protected_col] == 0]
    group_two = dataset.data[target_col].loc[dataset.data[protected_col] != 0]
    count = group_one.loc[group_one != 0].size, group_two.loc[group_two != 0].size

    #number of observations, get number of observations for each of the two groups
    nob = dataset.data[protected_col].loc[dataset.data[protected_col] == 0].size, \
          dataset.data[protected_col].loc[dataset.data[protected_col] != 0].size

    z_score, pval = sm.stats.proportions_ztest(count, nob)

    print(count)
    print(nob)
    print(norm.cdf(z_score))
    return z_score, pval
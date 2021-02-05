# Kowalski

![Kowalski Analysis GIF](https://media1.tenor.com/images/a1486d5a56353d73eda51c72ce2e1bd7/tenor.gif?itemid=12855053)

A prosonal collection of routines.

- Includes custom collection of metrics and scorers (namely kappa and specificity on `kowalski.utils.metrics`)
- A collection of estimators with hyperparam grids to cross validate
- Soon to be included, a collection of feature selection methods with grids to be cross validated (`kowalski.grid_search.estimator_pipelines.estimators_collection`)
- somewhat `mlflow` oriented
- plotting wrappers, such as `kowalski.utils.plotting.plot_confusion_matrix` (much wow here)
- scoring convenience provided by `kowalski.utils.evaluate.evaluate_on_test`
- assessment interfaces that I haven't seen anywhere else - check the [assessments section](#example-usage)



## installation instructions

Git clone this repo and pip install in dev mode to your target virtualenv/condaenv. Example:

```shell
git clone git@github.com:bgalvao/kowalski.git
cd kowalski
pip install -e ./  # fyi, this is run in the directory where setup.py resides
```


## example usage

There are two main features of this pseudo-package.

- Nested Cross Validation assessment (`kowalski.assessment.nested_cv.NestedCVTestAssessment`)
- Target Shuffiling assessment (`kowalski.assessment.target_shuffling.TargetShufflingAssessment`)

(ok I'll write this later)



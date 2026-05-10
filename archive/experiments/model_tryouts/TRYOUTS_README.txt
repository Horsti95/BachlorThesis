MODEL TRYOUTS - Experimental Model Comparison
===============================================
NOT PART OF THE THESIS. This is a standalone experiment
to find which ML model gets the best results on the sleep data.

QUICK START
-----------
1. Make sure the thesis pipeline has been run at least once
   (so that features_cache_global/ exists with cached features)

2. Run quick comparison (train/test split, fastest):
   python model_tryouts/all_models.py --cv train_test

3. Run proper comparison (5-fold CV):
   python model_tryouts/all_models.py --cv stratified_5fold

4. Run gold-standard comparison (LOSO, will take hours):
   python model_tryouts/all_models.py --cv loso

MODELS TESTED
-------------
Classical (on 149 hand-crafted features):
  - Logistic Regression, Ridge Classifier
  - kNN (k=5, k=10)
  - SVM (linear, RBF)
  - Naive Bayes
  - Decision Tree
  - Random Forest, Extra Trees
  - AdaBoost, Gradient Boosting
  - XGBoost, LightGBM, CatBoost

Neural Networks (on 149 features):
  - FNN / MLP (PyTorch)

Deep Learning (on raw EEG signals - needs --dl flag and --data-path):
  - 1D-CNN
  - LSTM, GRU
  - CNN-LSTM hybrid
  - Simple Transformer

EXTRA DEPENDENCIES
------------------
pip install lightgbm catboost

RESULTS
-------
Results are saved to model_tryouts/results/ as CSV and JSON files.

NOTES
-----
- Deep learning models need raw data (--data-path), not just cached features
- LOSO with deep learning is extremely slow, use --cv stratified_5fold for DL
- SVM can be slow on large datasets (>100k samples)
- Models that are not installed (e.g., LightGBM) are automatically skipped

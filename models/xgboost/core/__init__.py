"""
XGBoost model training and optimization pipeline.

This module contains the core training scripts for each step:
- Step 2a: XGBoost baseline
- Step 2b: Hyperparameter optimization  
- Step 2c: Probability calibration
- Step 3: Betting strategies evaluation
"""

__all__ = ['step2a_baseline', 'step2b_optimization', 'step2c_calibration', 'step3_strategies']

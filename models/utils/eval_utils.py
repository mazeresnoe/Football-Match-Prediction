"""
Fonctions d'évaluation et affichage des résultats
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from typing import Dict, List, Tuple
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import models.configs.global_config as cfg

# ========================================
# ROI / ÉVALUATION
# ========================================

def calculate_roi(y_true: np.ndarray, y_pred_proba: np.ndarray, odds_df: pd.DataFrame) -> Tuple[float,float,int]:
    total_stake, total_return, n_bets = 0, 0, 0
    for i, (true_result, probs) in enumerate(zip(y_true, y_pred_proba)):
        pred_class = np.argmax(probs)
        pred_prob = probs[pred_class]
        if pred_prob < cfg.MIN_CONFIDENCE_THRESHOLD:
            continue
        stake = cfg.STAKE_PER_MATCH
        n_bets += 1
        total_stake += stake
        if pred_class == 0:
            odds = odds_df.iloc[i]['odds_home']; win = (true_result==1)
        elif pred_class == 1:
            odds = odds_df.iloc[i]['odds_draw']; win = (true_result==0)
        else:
            odds = odds_df.iloc[i]['odds_away']; win = (true_result==-1)
        if win: total_return += stake*odds
    profit = total_return-total_stake
    roi = (profit/total_stake*100) if total_stake>0 else 0
    return roi, profit, n_bets


def evaluate_predictions(y_true: np.ndarray, y_pred_proba: np.ndarray, odds_df: pd.DataFrame=None, model_name: str="Model") -> Dict:
    y_true_idx = np.array([{1:0,0:1,-1:2}[y] for y in y_true])
    valid_mask = np.isfinite(y_pred_proba).all(axis=1)
    valid_mask &= (y_pred_proba.sum(axis=1)>0)
    y_true_idx_valid = y_true_idx[valid_mask]
    y_pred_proba_valid = y_pred_proba[valid_mask]
    y_pred_valid = np.argmax(y_pred_proba_valid, axis=1)

    acc = accuracy_score(y_true_idx_valid, y_pred_valid)
    ll = log_loss(y_true_idx_valid, y_pred_proba_valid)

    y_true_onehot = np.zeros((len(y_true_idx_valid),3))
    y_true_onehot[np.arange(len(y_true_idx_valid)), y_true_idx_valid]=1
    # Brier par classe (optionnel, mais cohérent)
    brier_scores = np.mean((y_pred_proba_valid - y_true_onehot) ** 2, axis=0)
    # Brier multiclasses officiel (identique au code 2)
    brier_avg = np.mean(np.sum((y_pred_proba_valid - y_true_onehot) ** 2, axis=1))

    roi, profit, n_bets = None, None, 0
    if odds_df is not None:
        combined_mask = valid_mask & odds_df['odds_home'].notna().values
        if combined_mask.sum()>0:
            roi, profit, n_bets = calculate_roi(y_true[combined_mask], y_pred_proba[combined_mask], odds_df.loc[combined_mask])

    return {
        "model": model_name,
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier_avg,
        "brier_home": brier_scores[0],
        "brier_draw": brier_scores[1],
        "brier_away": brier_scores[2],
        "roi": roi,
        "profit": profit,
        "n_bets": n_bets
    }


def print_evaluation_summary(results: Dict, dataset_name: str=""):
    dataset_str = f" - {dataset_name}" if dataset_name else ""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {results['model']:^50}{dataset_str:^10} ║
╚══════════════════════════════════════════════════════════════╝
 Accuracy : {results['accuracy']:.4f}, LogLoss : {results['log_loss']:.4f}, Brier : {results['brier_score']:.4f}
    - Home {results['brier_home']:.4f}, Draw {results['brier_draw']:.4f}, Away {results['brier_away']:.4f}
 ROI : {results['roi']}, Profit : {results['profit']}, N Paris : {results['n_bets']}
""")


def compare_train_cv_results(train_results: Dict, cv_results: Dict):
    print(f"\n{'='*70}\n  COMPARAISON TRAIN vs CV\n{'='*70}")
    acc_diff = train_results['accuracy'] - cv_results['accuracy']
    ll_diff = cv_results['log_loss'] - train_results['log_loss']
    bs_diff = cv_results['brier_score'] - train_results['brier_score']
    roi_diff = (train_results['roi'] or 0) - (cv_results['roi'] or 0)
    print(f"Accuracy Diff: {acc_diff:.4f}, LogLoss Diff: {ll_diff:.4f}, Brier Diff: {bs_diff:.4f}, ROI Diff: {roi_diff:.2f}")


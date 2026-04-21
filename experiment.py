"""
Persistent Anomaly Framework — Pilot Simulation
================================================
Synthetic validation of the PAF screening pipeline.

Generates a cohort of N subjects (T sessions each, K items per session, M options),
injects above-chance signal into a small subset ("true anomalies"), and tests
whether the 3-phase pipeline (screening → Bayesian validation → characterization)
recovers the injected anomalies at the claimed sensitivity/specificity.

Author: Zach Sparks
"""

import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    n_subjects: int = 200
    n_anomalies: int = 8          # true persistent anomalies
    n_sessions: int = 5
    n_items: int = 100
    n_options: int = 4            # M
    effect_sizes: list = field(default_factory=lambda: [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25])
    bayesian_prior: float = 0.05  # pi_0
    pa_threshold: float = 0.95    # posterior threshold
    dbscan_eps: float = 1.5
    dbscan_min_samples: int = 3
    n_seeds: int = 5
    random_seed: int = 42


# ============================================================
# Data Generation
# ============================================================
def generate_answer_key(n_items: int, n_options: int, rng: np.random.Generator) -> np.ndarray:
    """Generate hidden ground truth key."""
    return rng.integers(0, n_options, size=n_items)


def generate_session(
    n_items: int, n_options: int, answer_key: np.ndarray,
    is_anomaly: bool, effect_size: float, rng: np.random.Generator
) -> dict:
    """
    Generate one session of responses for one subject.
    
    For null subjects: uniform random responses.
    For anomalies: elevated probability of matching answer key,
    plus structured error patterns (directional bias, temporal clustering).
    """
    base_p = 1.0 / n_options
    
    if is_anomaly:
        # Anomaly: elevated accuracy toward hidden key
        p_correct = base_p + effect_size
        p_incorrect = (1.0 - p_correct) / (n_options - 1)
        
        responses = np.zeros(n_items, dtype=int)
        initial_selections = np.zeros(n_items, dtype=int)
        response_times = np.zeros(n_items)
        
        for j in range(n_items):
            probs = np.full(n_options, p_incorrect)
            probs[answer_key[j]] = p_correct
            probs = np.clip(probs, 0.01, None)
            probs /= probs.sum()
            
            # Initial selection: even higher accuracy (pre-override signal)
            init_probs = np.full(n_options, p_incorrect * 0.8)
            init_probs[answer_key[j]] = p_correct * 1.3
            init_probs = np.clip(init_probs, 0.01, None)
            init_probs /= init_probs.sum()
            initial_selections[j] = rng.choice(n_options, p=init_probs)
            
            # Override: sometimes change correct initial to incorrect
            if initial_selections[j] == answer_key[j] and rng.random() < 0.3:
                # Conscious override — select wrong answer
                wrong_options = [o for o in range(n_options) if o != answer_key[j]]
                responses[j] = rng.choice(wrong_options)
            else:
                responses[j] = rng.choice(n_options, p=probs)
            
            # Response time: faster on items where initial = correct
            base_rt = rng.exponential(2.0) + 1.0
            if initial_selections[j] == answer_key[j]:
                response_times[j] = base_rt * 0.85  # faster intuitive
            else:
                response_times[j] = base_rt * 1.1
        
        # Temporal clustering: anomaly signal clusters in bursts
        burst_mask = np.zeros(n_items, dtype=bool)
        n_bursts = rng.integers(2, 5)
        for _ in range(n_bursts):
            start = rng.integers(0, n_items - 8)
            burst_mask[start:start+rng.integers(3, 8)] = True
        
    else:
        # Null subject: pure random
        responses = rng.integers(0, n_options, size=n_items)
        initial_selections = rng.integers(0, n_options, size=n_items)
        response_times = rng.exponential(2.0, size=n_items) + 1.0
        burst_mask = np.zeros(n_items, dtype=bool)
    
    return {
        'responses': responses,
        'initial_selections': initial_selections,
        'response_times': response_times,
        'burst_mask': burst_mask,
    }


def generate_cohort(cfg: Config, rng: np.random.Generator) -> Tuple[list, np.ndarray]:
    """Generate full cohort data: subjects × sessions."""
    labels = np.zeros(cfg.n_subjects, dtype=int)
    labels[:cfg.n_anomalies] = 1
    
    cohort = []
    for i in range(cfg.n_subjects):
        subject_sessions = []
        effect = cfg.effect_sizes[i % len(cfg.effect_sizes)] if labels[i] else 0.0
        for t in range(cfg.n_sessions):
            key = generate_answer_key(cfg.n_items, cfg.n_options, rng)
            session = generate_session(
                cfg.n_items, cfg.n_options, key,
                is_anomaly=bool(labels[i]), effect_size=effect, rng=rng
            )
            session['answer_key'] = key
            subject_sessions.append(session)
        cohort.append(subject_sessions)
    
    return cohort, labels


# ============================================================
# Feature Extraction (11 dimensions)
# ============================================================
def extract_features(cohort: list, cfg: Config) -> np.ndarray:
    """Extract 11-dimensional feature vector for each subject."""
    n = len(cohort)
    features = np.zeros((n, 11))
    
    for i, sessions in enumerate(cohort):
        # Accuracy per session
        accuracies = []
        all_errors = []
        all_rts = []
        override_counts = []
        
        for s in sessions:
            resp = s['responses']
            key = s['answer_key']
            init = s['initial_selections']
            
            acc = np.mean(resp == key)
            accuracies.append(acc)
            
            # Error vectors (distance from correct)
            errors = (resp - key) % cfg.n_options
            all_errors.append(errors)
            all_rts.append(s['response_times'])
            
            # Override: initial correct, final wrong
            overrides = np.sum((init == key) & (resp != key))
            override_counts.append(overrides / cfg.n_items)
        
        accuracies = np.array(accuracies)
        
        # --- Signal Features (F1-F6) ---
        
        # F1: Response consistency (cross-session correlation)
        if cfg.n_sessions >= 2:
            corrs = []
            for t1 in range(cfg.n_sessions):
                for t2 in range(t1+1, cfg.n_sessions):
                    r1 = sessions[t1]['responses']
                    r2 = sessions[t2]['responses']
                    corrs.append(np.corrcoef(r1.astype(float), r2.astype(float))[0,1])
            features[i, 0] = np.nanmean(corrs)
        
        # F2: Cross-session entropy
        resp_concat = np.concatenate([s['responses'] for s in sessions])
        _, counts = np.unique(resp_concat, return_counts=True)
        probs = counts / counts.sum()
        features[i, 1] = -np.sum(probs * np.log(probs + 1e-10))
        
        # F3: Cluster convergence (accuracy stability)
        features[i, 2] = 1.0 - np.std(accuracies) if len(accuracies) > 1 else 0.5
        
        # F4: Mahalanobis distance from chance (0.25)
        chance = 1.0 / cfg.n_options
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies) + 1e-10
        features[i, 3] = abs(mean_acc - chance) / std_acc
        
        # F5: Bayesian posterior (cumulative across sessions)
        prior = cfg.bayesian_prior
        posterior = prior
        for acc in accuracies:
            # Likelihood ratio: binomial
            k = int(acc * cfg.n_items)
            n_items = cfg.n_items
            p0 = 1.0 / cfg.n_options
            p1 = min(p0 + 0.15, 0.9)  # assumed anomaly accuracy
            
            lr = (stats.binom.pmf(k, n_items, p1) + 1e-300) / \
                 (stats.binom.pmf(k, n_items, p0) + 1e-300)
            lr = np.clip(lr, 1e-10, 1e10)
            
            posterior = (lr * posterior) / (lr * posterior + (1 - posterior))
            posterior = np.clip(posterior, 1e-10, 1 - 1e-10)
        features[i, 4] = posterior
        
        # F6: Local density (mean accuracy as proxy)
        features[i, 5] = np.mean(accuracies)
        
        # --- Noise Features (N1-N5) ---
        
        # N1: Temporal asymmetry (anticipatory MI proxy)
        # Check if responses predict *next* trial's answer
        ami_scores = []
        for s in sessions:
            resp = s['responses']
            key = s['answer_key']
            # Forward correlation: does response[t] predict key[t+1]?
            if len(resp) > 1:
                forward_hits = np.mean(resp[:-1] == key[1:])
                backward_hits = np.mean(resp[1:] == key[:-1])
                ami_scores.append(forward_hits - backward_hits)
        features[i, 6] = np.mean(ami_scores) if ami_scores else 0.0
        
        # N2: Error directionality (KL divergence of error distribution)
        all_err_flat = np.concatenate(all_errors)
        err_hist = np.bincount(all_err_flat, minlength=cfg.n_options).astype(float)
        err_hist /= err_hist.sum() + 1e-10
        uniform = np.ones(cfg.n_options) / cfg.n_options
        kl = np.sum(err_hist * np.log((err_hist + 1e-10) / uniform))
        features[i, 7] = kl
        
        # N3: Context-modulated variance
        # Split sessions into "high load" (odd) and "low load" (even)
        if cfg.n_sessions >= 2:
            high_acc = [accuracies[t] for t in range(0, cfg.n_sessions, 2)]
            low_acc = [accuracies[t] for t in range(1, cfg.n_sessions, 2)]
            var_high = np.var(high_acc) + 1e-10
            var_low = np.var(low_acc) + 1e-10
            features[i, 8] = var_high / var_low
        
        # N4: Cross-task coherence (correlation of accuracy across session pairs)
        if cfg.n_sessions >= 2:
            per_item_acc = []
            for s in sessions:
                per_item_acc.append((s['responses'] == s['answer_key']).astype(float))
            cross_corrs = []
            for t1 in range(len(per_item_acc)):
                for t2 in range(t1+1, len(per_item_acc)):
                    c = np.corrcoef(per_item_acc[t1], per_item_acc[t2])[0,1]
                    if not np.isnan(c):
                        cross_corrs.append(abs(c))
            features[i, 9] = np.mean(cross_corrs) if cross_corrs else 0.0
        
        # N5: Heavy tails (kurtosis of accuracy distribution as proxy for alpha-stable)
        all_rt_flat = np.concatenate(all_rts)
        features[i, 10] = stats.kurtosis(all_rt_flat, fisher=True)
        
        # Override index (bonus diagnostic, not in 11-dim but tracked)
    
    return features


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(features: np.ndarray, labels: np.ndarray, cfg: Config) -> dict:
    """Run 3-phase PAF pipeline."""
    n = len(labels)
    
    # --- Phase 1: Screening via k-means ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=cfg.random_seed, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Find cluster with highest mean Bayesian posterior (F5)
    cluster_posteriors = {}
    for c in range(3):
        mask = clusters == c
        cluster_posteriors[c] = np.mean(features[mask, 4])
    
    best_cluster = max(cluster_posteriors, key=cluster_posteriors.get)
    phase1_candidates = np.where(clusters == best_cluster)[0]
    
    # --- Phase 2: Bayesian validation ---
    phase2_confirmed = []
    posteriors_all = features[:, 4]
    
    for idx in phase1_candidates:
        if posteriors_all[idx] >= cfg.pa_threshold:
            phase2_confirmed.append(idx)
    
    # Also check all subjects (some anomalies might be in other clusters)
    for idx in range(n):
        if idx not in phase1_candidates and posteriors_all[idx] >= cfg.pa_threshold:
            phase2_confirmed.append(idx)
    
    phase2_confirmed = sorted(set(phase2_confirmed))
    
    # --- Phase 3: DBSCAN + XGBoost characterization ---
    dbscan = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    db_labels = dbscan.fit_predict(X_scaled)
    
    # XGBoost classification (train on full features, evaluate)
    from xgboost import XGBClassifier
    
    # Use posterior threshold as primary classifier
    predictions = (posteriors_all >= cfg.pa_threshold).astype(int)
    
    # Also train XGBoost for AUC
    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=cfg.random_seed, eval_metric='logloss',
        use_label_encoder=False
    )
    xgb.fit(X_scaled, labels)
    xgb_probs = xgb.predict_proba(X_scaled)[:, 1]
    xgb_preds = xgb.predict(X_scaled)
    
    # --- Metrics ---
    # Bayesian threshold classifier
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC using posterior as score
    auc_posterior = roc_auc_score(labels, posteriors_all)
    auc_xgb = roc_auc_score(labels, xgb_probs)
    
    # Feature importance
    importance = dict(zip(
        ['F1_consistency', 'F2_entropy', 'F3_convergence', 'F4_mahalanobis',
         'F5_posterior', 'F6_density', 'N1_temporal_asymmetry', 'N2_error_direction',
         'N3_context_variance', 'N4_cross_task', 'N5_heavy_tails'],
        xgb.feature_importances_.tolist()
    ))
    
    return {
        'phase1_candidates': len(phase1_candidates),
        'phase2_confirmed': len(phase2_confirmed),
        'bayesian_threshold': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'auc': auc_posterior,
        },
        'xgboost': {
            'auc': auc_xgb,
            'sensitivity': float(np.sum((xgb_preds == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)),
            'specificity': float(np.sum((xgb_preds == 0) & (labels == 0)) / max(np.sum(labels == 0), 1)),
        },
        'feature_importance': importance,
        'anomaly_posteriors': posteriors_all[:cfg.n_anomalies].tolist(),
        'null_posterior_max': float(np.max(posteriors_all[cfg.n_anomalies:])),
        'null_posterior_mean': float(np.mean(posteriors_all[cfg.n_anomalies:])),
    }


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config()
    all_results = []
    
    print("=" * 70)
    print("PAF Pilot Simulation: Persistent Anomaly Detection")
    print(f"N={cfg.n_subjects}, T={cfg.n_sessions} sessions, "
          f"K={cfg.n_items} items, M={cfg.n_options} options")
    print(f"True anomalies: {cfg.n_anomalies} "
          f"(effect sizes: {cfg.effect_sizes})")
    print("=" * 70)
    
    for seed in range(cfg.n_seeds):
        cfg_seed = Config(random_seed=cfg.random_seed + seed)
        rng = np.random.default_rng(cfg_seed.random_seed)
        
        cohort, labels = generate_cohort(cfg_seed, rng)
        features = extract_features(cohort, cfg_seed)
        results = run_pipeline(features, labels, cfg_seed)
        results['seed'] = seed
        all_results.append(results)
        
        print(f"\nSeed {seed}:")
        print(f"  Phase 1 candidates: {results['phase1_candidates']}")
        print(f"  Phase 2 confirmed:  {results['phase2_confirmed']}")
        b = results['bayesian_threshold']
        print(f"  Bayesian threshold: Sens={b['sensitivity']:.3f}, "
              f"Spec={b['specificity']:.3f}, AUC={b['auc']:.3f}")
        x = results['xgboost']
        print(f"  XGBoost:           Sens={x['sensitivity']:.3f}, "
              f"Spec={x['specificity']:.3f}, AUC={x['auc']:.3f}")
        print(f"  Anomaly posteriors: {[f'{p:.4f}' for p in results['anomaly_posteriors']]}")
        print(f"  Max null posterior:  {results['null_posterior_max']:.6f}")
    
    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    bayes_sens = [r['bayesian_threshold']['sensitivity'] for r in all_results]
    bayes_spec = [r['bayesian_threshold']['specificity'] for r in all_results]
    bayes_auc = [r['bayesian_threshold']['auc'] for r in all_results]
    xgb_auc = [r['xgboost']['auc'] for r in all_results]
    
    print(f"\nBayesian Threshold (P >= {cfg.pa_threshold}):")
    print(f"  Sensitivity: {np.mean(bayes_sens):.3f} ± {np.std(bayes_sens):.3f}")
    print(f"  Specificity: {np.mean(bayes_spec):.3f} ± {np.std(bayes_spec):.3f}")
    print(f"  AUC:         {np.mean(bayes_auc):.3f} ± {np.std(bayes_auc):.3f}")
    
    print(f"\nXGBoost (11-dimensional feature space):")
    print(f"  AUC:         {np.mean(xgb_auc):.3f} ± {np.std(xgb_auc):.3f}")
    
    # Feature importance aggregate
    print(f"\nFeature Importance (mean across seeds):")
    feat_names = list(all_results[0]['feature_importance'].keys())
    for fname in sorted(feat_names, key=lambda f: -np.mean([r['feature_importance'][f] for r in all_results])):
        vals = [r['feature_importance'][fname] for r in all_results]
        print(f"  {fname:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Save
    summary = {
        'config': asdict(cfg),
        'aggregate': {
            'bayesian_sensitivity': f"{np.mean(bayes_sens):.3f} ± {np.std(bayes_sens):.3f}",
            'bayesian_specificity': f"{np.mean(bayes_spec):.3f} ± {np.std(bayes_spec):.3f}",
            'bayesian_auc': f"{np.mean(bayes_auc):.3f} ± {np.std(bayes_auc):.3f}",
            'xgboost_auc': f"{np.mean(xgb_auc):.3f} ± {np.std(xgb_auc):.3f}",
        },
        'per_seed': all_results,
    }
    
    with open('results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to results.json")
    return summary


if __name__ == '__main__':
    main()

#include "../include/Metrics.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>

double Metrics::accuracy(const VectorXd& y_true, const VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have same size");
    }
    
    int correct = 0;
    for (int i = 0; i < y_true.size(); i++) {
        if (std::abs(y_true(i) - y_pred(i)) < 0.5) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y_true.size();
}

ConfusionMatrix Metrics::confusion_matrix(const VectorXd& y_true, 
                                        const VectorXd& y_pred) {
    ConfusionMatrix cm = {0, 0, 0, 0};
    
    for (int i = 0; i < y_true.size(); i++) {
        bool actual_positive = y_true(i) > 0.5;
        bool predicted_positive = y_pred(i) > 0.5;
        
        if (actual_positive && predicted_positive) {
            cm.true_positive++;
        } else if (!actual_positive && !predicted_positive) {
            cm.true_negative++;
        } else if (!actual_positive && predicted_positive) {
            cm.false_positive++;
        } else {
            cm.false_negative++;
        }
    }
    
    return cm;
}

double Metrics::precision(const VectorXd& y_true, const VectorXd& y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    if (cm.true_positive + cm.false_positive == 0) {
        return 0.0;
    }
    return static_cast<double>(cm.true_positive) / 
           (cm.true_positive + cm.false_positive);
}

double Metrics::recall(const VectorXd& y_true, const VectorXd& y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    if (cm.true_positive + cm.false_negative == 0) {
        return 0.0;
    }
    return static_cast<double>(cm.true_positive) / 
           (cm.true_positive + cm.false_negative);
}

double Metrics::f1_score(const VectorXd& y_true, const VectorXd& y_pred) {
    double prec = precision(y_true, y_pred);
    double rec = recall(y_true, y_pred);
    if (prec + rec == 0.0) {
        return 0.0;
    }
    return 2.0 * prec * rec / (prec + rec);
}

double Metrics::roc_auc(const VectorXd& y_true, const VectorXd& y_pred_proba) {
    if (y_true.size() != y_pred_proba.size()) {
        throw std::invalid_argument("y_true and y_pred_proba must have same size");
    }
    
    // Get indices sorted by predicted probability
    std::vector<int> indices(y_true.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&y_pred_proba](int i, int j) {
            return y_pred_proba(i) > y_pred_proba(j);
        });
    
    double auc = 0.0;
    double fp = 0.0, tp = 0.0;
    double fp_prev = 0.0, tp_prev = 0.0;
    
    // Count total positives and negatives
    double total_positives = (y_true.array() > 0.5).count();
    double total_negatives = y_true.size() - total_positives;
    
    if (total_positives == 0 || total_negatives == 0) {
        return 0.5; // Cannot compute meaningful AUC
    }
    
    // Compute ROC curve points and AUC using trapezoidal rule
    for (int idx : indices) {
        if (y_true(idx) > 0.5) {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        auc += (fp - fp_prev) * (tp + tp_prev) / 2.0;
        fp_prev = fp;
        tp_prev = tp;
    }
    
    // Normalize AUC
    auc /= (total_positives * total_negatives);
    
    return auc;
}

ClassificationReport Metrics::classification_report(const VectorXd& y_true,
                                                  const VectorXd& y_pred,
                                                  const VectorXd& y_pred_proba) {
    ClassificationReport report;
    
    report.accuracy = accuracy(y_true, y_pred);
    report.precision = precision(y_true, y_pred);
    report.recall = recall(y_true, y_pred);
    report.f1_score = f1_score(y_true, y_pred);
    report.auc_roc = roc_auc(y_true, y_pred_proba);
    
    return report;
}

double Metrics::mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred) {
    return (y_true - y_pred).array().square().mean();
}

double Metrics::mean_absolute_error(const VectorXd& y_true, const VectorXd& y_pred) {
    return (y_true - y_pred).array().abs().mean();
}

double Metrics::log_loss(const VectorXd& y_true, const VectorXd& y_pred_proba) {
    double loss = 0.0;
    int n = y_true.size();
    
    for (int i = 0; i < n; i++) {
        double p = std::max(std::min(y_pred_proba(i), 1.0 - 1e-15), 1e-15);
        loss += y_true(i) * std::log(p) + (1.0 - y_true(i)) * std::log(1.0 - p);
    }
    
    return -loss / n;
}

double Metrics::find_best_threshold(const VectorXd& y_true,
                                  const VectorXd& y_pred_proba,
                                  const std::string& metric) {
    int n_steps = 100;
    double best_threshold = 0.5;
    double best_score = -1.0;
    
    for (int i = 0; i <= n_steps; i++) {
        double threshold = static_cast<double>(i) / n_steps;
        VectorXd y_pred = (y_pred_proba.array() >= threshold).cast<double>();
        
        double score;
        if (metric == "accuracy") {
            score = accuracy(y_true, y_pred);
        } else if (metric == "f1") {
            score = f1_score(y_true, y_pred);
        } else if (metric == "precision") {
            score = precision(y_true, y_pred);
        } else if (metric == "recall") {
            score = recall(y_true, y_pred);
        } else {
            throw std::invalid_argument("Unknown metric: " + metric);
        }
        
        if (score > best_score) {
            best_score = score;
            best_threshold = threshold;
        }
    }
    
    return best_threshold;
}

void Metrics::save_roc_curve_data(const VectorXd& y_true,
                                const VectorXd& y_pred_proba,
                                const std::string& filename) {
    // Get indices sorted by predicted probability
    std::vector<int> indices(y_true.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&y_pred_proba](int i, int j) {
            return y_pred_proba(i) > y_pred_proba(j);
        });
    
    double total_positives = (y_true.array() > 0.5).count();
    double total_negatives = y_true.size() - total_positives;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "fpr,tpr\n";
    
    double fp = 0.0, tp = 0.0;
    double fp_prev = 0.0, tp_prev = 0.0;
    
    for (int idx : indices) {
        if (y_true(idx) > 0.5) {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        double fpr = fp / total_negatives;
        double tpr = tp / total_positives;
        
        file << fpr << "," << tpr << "\n";
        
        fp_prev = fp;
        tp_prev = tp;
    }
    
    file.close();
}

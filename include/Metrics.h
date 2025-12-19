#pragma once
#include "Utils.h"
#include <vector>
#include <map>

struct ClassificationReport {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double auc_roc;
    
    // Per-class metrics for multiclass (future extension)
    std::map<int, double> class_precision;
    std::map<int, double> class_recall;
    std::map<int, double> class_f1;
    
    void print() const {
        std::cout << "\n=== Classification Report ===\n";
        std::cout << "Accuracy:  " << accuracy << "\n";
        std::cout << "Precision: " << precision << "\n";
        std::cout << "Recall:    " << recall << "\n";
        std::cout << "F1 Score:  " << f1_score << "\n";
        std::cout << "AUC-ROC:   " << auc_roc << "\n";
    }
};

struct ConfusionMatrix {
    int true_positive;
    int true_negative;
    int false_positive;
    int false_negative;
    
    void print() const {
        std::cout << "\n=== Confusion Matrix ===\n";
        std::cout << "               Predicted\n";
        std::cout << "               +     -\n";
        std::cout << "Actual   +   " << true_positive << "    " << false_negative << "\n";
        std::cout << "         -   " << false_positive << "    " << true_negative << "\n";
    }
};

class Metrics {
public:
    // Basic metrics
    static double accuracy(const VectorXd& y_true, const VectorXd& y_pred);
    static double precision(const VectorXd& y_true, const VectorXd& y_pred);
    static double recall(const VectorXd& y_true, const VectorXd& y_pred);
    static double f1_score(const VectorXd& y_true, const VectorXd& y_pred);
    
    // Confusion matrix
    static ConfusionMatrix confusion_matrix(const VectorXd& y_true, 
                                          const VectorXd& y_pred);
    
    // ROC-AUC
    static double roc_auc(const VectorXd& y_true, const VectorXd& y_pred_proba);
    
    // Complete classification report
    static ClassificationReport classification_report(const VectorXd& y_true, 
                                                    const VectorXd& y_pred,
                                                    const VectorXd& y_pred_proba);
    
    // Additional metrics
    static double mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred);
    static double mean_absolute_error(const VectorXd& y_true, const VectorXd& y_pred);
    static double log_loss(const VectorXd& y_true, const VectorXd& y_pred_proba);
    
    // For multiclass (future extension)
    static std::vector<double> precision_per_class(const VectorXi& y_true, 
                                                 const VectorXi& y_pred);
    static std::vector<double> recall_per_class(const VectorXi& y_true, 
                                              const VectorXi& y_pred);
    
    // Threshold optimization
    static double find_best_threshold(const VectorXd& y_true, 
                                    const VectorXd& y_pred_proba,
                                    const std::string& metric = "f1");
    
    // Plotting helper (output data for external plotting)
    static void save_roc_curve_data(const VectorXd& y_true, 
                                  const VectorXd& y_pred_proba,
                                  const std::string& filename);
};

#pragma once
#include "Utils.h"
#include <vector>
#include <string>
#include <memory>

class LogisticRegression {
private:
    VectorXd weights;
    double learning_rate;
    double regularization_lambda;
    std::string regularization_type; // "l1", "l2", or "none"
    int max_iterations;
    double tolerance;
    bool verbose;
    
    std::vector<double> loss_history;
    std::vector<double> grad_norm_history;
    
    // Internal methods
    double computeLoss(const VectorXd& y_true, const VectorXd& y_pred) const;
    VectorXd computeGradients(const MatrixXd& X, const VectorXd& y_true, 
                             const VectorXd& y_pred) const;
    void applyRegularization(VectorXd& gradients, const VectorXd& weights) const;
    
public:
    // Constructors
    LogisticRegression(double lr = 0.01, int max_iter = 1000, 
                      double tol = 1e-4, bool verbose = false);
    
    // Core methods
    void fit(const MatrixXd& X, const VectorXd& y);
    VectorXd predict_proba(const MatrixXd& X) const;
    VectorXd predict(const MatrixXd& X, double threshold = 0.5) const;
    
    // Static helper methods
    static double sigmoid(double z);
    static VectorXd sigmoid(const VectorXd& z);
    
    // Regularization
    void setRegularization(double lambda, const std::string& type = "l2");
    
    // Model evaluation
    double score(const MatrixXd& X, const VectorXd& y, double threshold = 0.5) const;
    
    // Model persistence
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);
    
    // Getters
    VectorXd getWeights() const { return weights; }
    std::vector<double> getLossHistory() const { return loss_history; }
    std::vector<double> getGradNormHistory() const { return grad_norm_history; }
    
    // Setters
    void setLearningRate(double lr) { learning_rate = lr; }
    void setMaxIterations(int max_iter) { max_iterations = max_iter; }
    void setVerbose(bool v) { verbose = v; }
    
    // Feature importance
    VectorXd getFeatureImportance() const;
    
    // Early stopping callback (for extension)
    virtual bool earlyStoppingCallback(int iteration, double loss, 
                                      double grad_norm) {
        return false; // Override in derived classes
    }
};

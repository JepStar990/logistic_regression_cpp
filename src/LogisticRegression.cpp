#include "../include/LogisticRegression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

LogisticRegression::LogisticRegression(double lr, int max_iter, 
                                     double tol, bool verbose)
    : learning_rate(lr), max_iterations(max_iter), 
      tolerance(tol), verbose(verbose),
      regularization_lambda(0.0), regularization_type("none") {}

double LogisticRegression::sigmoid(double z) {
    if (z > 0) {
        return 1.0 / (1.0 + std::exp(-z));
    } else {
        double exp_z = std::exp(z);
        return exp_z / (1.0 + exp_z);
    }
}

VectorXd LogisticRegression::sigmoid(const VectorXd& z) {
    VectorXd result(z.size());
    for (int i = 0; i < z.size(); i++) {
        result(i) = sigmoid(z(i));
    }
    return result;
}

void LogisticRegression::setRegularization(double lambda, const std::string& type) {
    regularization_lambda = lambda;
    regularization_type = type;
    if (type != "l1" && type != "l2" && type != "none") {
        std::cerr << "Warning: Unknown regularization type. Using 'none'." << std::endl;
        regularization_type = "none";
    }
}

double LogisticRegression::computeLoss(const VectorXd& y_true, 
                                     const VectorXd& y_pred) const {
    int m = y_true.size();
    double loss = 0.0;
    
    // Binary cross-entropy loss
    for (int i = 0; i < m; i++) {
        double y = y_true(i);
        double p = std::max(std::min(y_pred(i), 1.0 - 1e-15), 1e-15);
        loss += y * std::log(p) + (1.0 - y) * std::log(1.0 - p);
    }
    loss = -loss / m;
    
    // Add regularization
    if (regularization_lambda > 0) {
        if (regularization_type == "l2") {
            loss += regularization_lambda * 0.5 * weights.squaredNorm();
        } else if (regularization_type == "l1") {
            loss += regularization_lambda * weights.array().abs().sum();
        }
    }
    
    return loss;
}

VectorXd LogisticRegression::computeGradients(const MatrixXd& X, 
                                            const VectorXd& y_true,
                                            const VectorXd& y_pred) const {
    int m = y_true.size();
    VectorXd error = y_pred - y_true;
    VectorXd gradients = X.transpose() * error / m;
    
    return gradients;
}

void LogisticRegression::applyRegularization(VectorXd& gradients, 
                                           const VectorXd& weights) const {
    if (regularization_lambda > 0) {
        if (regularization_type == "l2") {
            gradients += regularization_lambda * weights;
        } else if (regularization_type == "l1") {
            for (int i = 0; i < gradients.size(); i++) {
                gradients(i) += regularization_lambda * (weights(i) > 0 ? 1.0 : -1.0);
            }
        }
    }
}

void LogisticRegression::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    int m = X.rows();
    int n = X.cols();
    
    // Initialize weights
    weights = VectorXd::Random(n) * 0.01;
    
    // Clear history
    loss_history.clear();
    grad_norm_history.clear();
    
    Utils::Timer timer;
    
    // Gradient descent
    for (int iter = 0; iter < max_iterations; iter++) {
        // Forward pass
        VectorXd z = X * weights;
        VectorXd y_pred = sigmoid(z);
        
        // Compute loss
        double loss = computeLoss(y, y_pred);
        loss_history.push_back(loss);
        
        // Compute gradients
        VectorXd gradients = computeGradients(X, y, y_pred);
        applyRegularization(gradients, weights);
        
        double grad_norm = gradients.norm();
        grad_norm_history.push_back(grad_norm);
        
        // Update weights
        weights -= learning_rate * gradients;
        
        // Check convergence
        if (grad_norm < tolerance) {
            if (verbose) {
                std::cout << "Converged after " << iter + 1 
                         << " iterations" << std::endl;
            }
            break;
        }
        
        // Early stopping callback
        if (earlyStoppingCallback(iter, loss, grad_norm)) {
            if (verbose) {
                std::cout << "Early stopping at iteration " << iter + 1 << std::endl;
            }
            break;
        }
        
        // Print progress
        if (verbose && iter % 100 == 0) {
            std::cout << "Iteration " << iter << ": Loss = " << loss 
                     << ", Grad norm = " << grad_norm << std::endl;
        }
    }
    
    if (verbose) {
        double training_time = timer.elapsed();
        std::cout << "\nTraining completed in " << training_time << " seconds\n";
        std::cout << "Final loss: " << loss_history.back() << std::endl;
        std::cout << "Final gradient norm: " << grad_norm_history.back() << std::endl;
    }
}

VectorXd LogisticRegression::predict_proba(const MatrixXd& X) const {
    if (X.cols() != weights.size()) {
        throw std::invalid_argument("X must have the same number of features as training data");
    }
    
    VectorXd z = X * weights;
    return sigmoid(z);
}

VectorXd LogisticRegression::predict(const MatrixXd& X, double threshold) const {
    VectorXd probabilities = predict_proba(X);
    VectorXd predictions = (probabilities.array() >= threshold).cast<double>();
    return predictions;
}

double LogisticRegression::score(const MatrixXd& X, const VectorXd& y, 
                               double threshold) const {
    VectorXd predictions = predict(X, threshold);
    int correct = 0;
    for (int i = 0; i < y.size(); i++) {
        if (std::abs(predictions(i) - y(i)) < 0.5) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y.size();
}

bool LogisticRegression::saveModel(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    file << "LogisticRegression Model\n";
    file << "Version: 1.0\n";
    file << "Weights: " << weights.size() << "\n";
    for (int i = 0; i < weights.size(); i++) {
        file << weights(i) << " ";
    }
    file << "\n";
    file << "LearningRate: " << learning_rate << "\n";
    file << "RegularizationLambda: " << regularization_lambda << "\n";
    file << "RegularizationType: " << regularization_type << "\n";
    
    file.close();
    return true;
}

bool LogisticRegression::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    std::getline(file, line); // Skip version
    
    std::getline(file, line);
    if (line.find("Weights:") == std::string::npos) {
        std::cerr << "Error: Invalid model file format" << std::endl;
        return false;
    }
    
    int weight_size = std::stoi(line.substr(line.find(":") + 2));
    std::getline(file, line);
    
    std::istringstream weight_stream(line);
    weights.resize(weight_size);
    for (int i = 0; i < weight_size; i++) {
        weight_stream >> weights(i);
    }
    
    // Read other parameters
    while (std::getline(file, line)) {
        if (line.find("LearningRate:") != std::string::npos) {
            learning_rate = std::stod(line.substr(line.find(":") + 2));
        } else if (line.find("RegularizationLambda:") != std::string::npos) {
            regularization_lambda = std::stod(line.substr(line.find(":") + 2));
        } else if (line.find("RegularizationType:") != std::string::npos) {
            regularization_type = line.substr(line.find(":") + 2);
        }
    }
    
    file.close();
    return true;
}

VectorXd LogisticRegression::getFeatureImportance() const {
    return weights.array().abs();
}

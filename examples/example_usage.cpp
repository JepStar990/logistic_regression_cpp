#include "../include/DataLoader.h"
#include "../include/LogisticRegression.h"
#include "../include/Metrics.h"
#include <iostream>

void example_simple() {
    std::cout << "=== Simple Example ===\n";
    
    // Create simple dataset
    MatrixXd X(6, 2);
    VectorXd y(6);
    
    // Features: [study_hours, sleep_hours]
    X << 1, 5,
         2, 4,
         3, 3,
         4, 2,
         5, 1,
         6, 0;
    
    // Labels: 1 = pass, 0 = fail
    y << 0, 0, 0, 1, 1, 1;
    
    // Add bias term
    MatrixXd X_with_bias(X.rows(), X.cols() + 1);
    X_with_bias.col(0) = VectorXd::Ones(X.rows());
    X_with_bias.block(0, 1, X.rows(), X.cols()) = X;
    
    // Create model
    LogisticRegression model(0.1, 100, 1e-4, true);
    
    // Train
    model.fit(X_with_bias, y);
    
    // Predict
    VectorXd y_pred = model.predict(X_with_bias);
    
    std::cout << "\nPredictions vs Actual:\n";
    for (int i = 0; i < y.size(); i++) {
        std::cout << "Sample " << i << ": Predicted = " << y_pred(i) 
                  << ", Actual = " << y(i) << std::endl;
    }
    
    double acc = model.score(X_with_bias, y);
    std::cout << "\nAccuracy: " << acc << std::endl;
}

void example_with_regularization() {
    std::cout << "\n=== Example with Regularization ===\n";
    
    // Generate synthetic data with some noise
    Dataset dataset = DataLoader::generateSyntheticData(500, 20, 0.3, 123);
    
    // Add bias term
    MatrixXd X_train_with_bias(dataset.X_train.rows(), dataset.X_train.cols() + 1);
    X_train_with_bias.col(0) = VectorXd::Ones(dataset.X_train.rows());
    X_train_with_bias.block(0, 1, dataset.X_train.rows(), dataset.X_train.cols()) = dataset.X_train;
    
    MatrixXd X_test_with_bias(dataset.X_test.rows(), dataset.X_test.cols() + 1);
    X_test_with_bias.col(0) = VectorXd::Ones(dataset.X_test.rows());
    X_test_with_bias.block(0, 1, dataset.X_test.rows(), dataset.X_test.cols()) = dataset.X_test;
    
    // Train without regularization
    LogisticRegression model_no_reg(0.1, 500, 1e-4, false);
    model_no_reg.fit(X_train_with_bias, dataset.y_train);
    
    // Train with L2 regularization
    LogisticRegression model_l2(0.1, 500, 1e-4, false);
    model_l2.setRegularization(0.1, "l2");
    model_l2.fit(X_train_with_bias, dataset.y_train);
    
    // Compare performance
    double acc_no_reg = model_no_reg.score(X_test_with_bias, dataset.y_test);
    double acc_l2 = model_l2.score(X_test_with_bias, dataset.y_test);
    
    std::cout << "Accuracy without regularization: " << acc_no_reg << std::endl;
    std::cout << "Accuracy with L2 regularization: " << acc_l2 << std::endl;
    
    // Compare weight magnitudes
    VectorXd weights_no_reg = model_no_reg.getWeights();
    VectorXd weights_l2 = model_l2.getWeights();
    
    std::cout << "\nWeight L2 norm without regularization: " << weights_no_reg.norm() << std::endl;
    std::cout << "Weight L2 norm with regularization: " << weights_l2.norm() << std::endl;
}

void example_feature_importance() {
    std::cout << "\n=== Feature Importance Example ===\n";
    
    // Generate data where only first 2 features are important
    Dataset dataset = DataLoader::generateSyntheticData(1000, 10, 0.2, 456);
    
    // Add bias term
    MatrixXd X_train_with_bias(dataset.X_train.rows(), dataset.X_train.cols() + 1);
    X_train_with_bias.col(0) = VectorXd::Ones(dataset.X_train.rows());
    X_train_with_bias.block(0, 1, dataset.X_train.rows(), dataset.X_train.cols()) = dataset.X_train;
    
    // Train model
    LogisticRegression model(0.05, 1000, 1e-4, false);
    model.fit(X_train_with_bias, dataset.y_train);
    
    // Get feature importance
    VectorXd importance = model.getFeatureImportance();
    
    std::cout << "\nFeature Importance (absolute weight values):\n";
    for (int i = 1; i < importance.size(); i++) { // Skip bias term
        std::cout << "Feature " << i-1 << ": " << importance(i) << std::endl;
    }
}

int main() {
    std::cout << "Logistic Regression Examples\n";
    std::cout << "===========================\n";
    
    example_simple();
    example_with_regularization();
    example_feature_importance();
    
    return 0;
}

#include "../include/DataLoader.h"
#include "../include/LogisticRegression.h"
#include "../include/Metrics.h"
#include <cassert>
#include <iostream>

void test_sigmoid() {
    std::cout << "Testing sigmoid function...\n";
    
    // Test extreme values
    assert(std::abs(LogisticRegression::sigmoid(0.0) - 0.5) < 1e-10);
    assert(LogisticRegression::sigmoid(10.0) > 0.999);
    assert(LogisticRegression::sigmoid(-10.0) < 0.001);
    
    // Test vector sigmoid
    VectorXd z(3);
    z << -1.0, 0.0, 1.0;
    VectorXd result = LogisticRegression::sigmoid(z);
    assert(std::abs(result(0) - 0.268941) < 1e-5);
    assert(std::abs(result(1) - 0.5) < 1e-5);
    assert(std::abs(result(2) - 0.731059) < 1e-5);
    
    std::cout << "✓ Sigmoid tests passed\n";
}

void test_metrics() {
    std::cout << "\nTesting metrics...\n";
    
    VectorXd y_true(5);
    VectorXd y_pred(5);
    
    y_true << 1, 0, 1, 0, 1;
    y_pred << 1, 0, 0, 0, 1;
    
    double acc = Metrics::accuracy(y_true, y_pred);
    assert(std::abs(acc - 0.8) < 1e-10);
    
    double prec = Metrics::precision(y_true, y_pred);
    assert(std::abs(prec - 1.0) < 1e-10);
    
    double rec = Metrics::recall(y_true, y_pred);
    assert(std::abs(rec - 0.666666) < 1e-5);
    
    double f1 = Metrics::f1_score(y_true, y_pred);
    assert(std::abs(f1 - 0.8) < 1e-5);
    
    std::cout << "✓ Metrics tests passed\n";
}

void test_logistic_regression() {
    std::cout << "\nTesting logistic regression...\n";
    
    // Generate simple synthetic data
    MatrixXd X(100, 2);
    VectorXd y(100);
    
    // Create linearly separable data
    for (int i = 0; i < 50; i++) {
        X(i, 0) = i * 0.1;
        X(i, 1) = i * 0.1 + 1.0;
        y(i) = 0.0;
    }
    for (int i = 50; i < 100; i++) {
        X(i, 0) = i * 0.1 + 5.0;
        X(i, 1) = i * 0.1 + 6.0;
        y(i) = 1.0;
    }
    
    // Add bias term
    MatrixXd X_with_bias(X.rows(), X.cols() + 1);
    X_with_bias.col(0) = VectorXd::Ones(X.rows());
    X_with_bias.block(0, 1, X.rows(), X.cols()) = X;
    
    // Create and train model
    LogisticRegression model(0.1, 100, 1e-4, false);
    model.fit(X_with_bias, y);
    
    // Test predictions
    VectorXd y_pred = model.predict(X_with_bias);
    double acc = model.score(X_with_bias, y);
    
    assert(acc > 0.9); // Should achieve high accuracy on separable data
    
    std::cout << "✓ Logistic regression tests passed (accuracy: " << acc << ")\n";
}

void test_dataloader() {
    std::cout << "\nTesting data loader...\n";
    
    // Test synthetic data generation
    Dataset dataset = DataLoader::generateSyntheticData(200, 5, 0.1, 42);
    
    assert(dataset.X_train.rows() == 160); // 80% of 200
    assert(dataset.X_test.rows() == 40);   // 20% of 200
    assert(dataset.X_train.cols() == 5);
    assert(dataset.y_train.size() == 160);
    assert(dataset.y_test.size() == 40);
    
    std::cout << "✓ Data loader tests passed\n";
}

int main() {
    std::cout << "=== Running Unit Tests ===\n";
    
    try {
        test_sigmoid();
        test_metrics();
        test_dataloader();
        test_logistic_regression();
        
        std::cout << "\n✅ All tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}

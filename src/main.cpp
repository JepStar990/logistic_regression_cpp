#include "../include/DataLoader.h"
#include "../include/LogisticRegression.h"
#include "../include/Metrics.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "=== Logistic Regression from Scratch in C++ ===\n";
    
    try {
        // Option 1: Load data from CSV
        DataLoader loader;
        if (loader.loadCSV("../datasets/titanic.csv", true, ',', true)) {
            loader.printInfo();
            
            // Split data
            Dataset dataset = loader.split(0.8, true, 42);
            
            // Normalize features
            loader.normalize();
            
            // Add bias term
            loader.addBiasTerm();
            
            // Create and train model
            LogisticRegression model(0.1, 1000, 1e-4, true);
            model.setRegularization(0.01, "l2");
            
            std::cout << "\nTraining model...\n";
            model.fit(dataset.X_train, dataset.y_train);
            
            // Make predictions
            VectorXd y_pred = model.predict(dataset.X_test);
            VectorXd y_pred_proba = model.predict_proba(dataset.X_test);
            
            // Evaluate
            ClassificationReport report = Metrics::classification_report(
                dataset.y_test, y_pred, y_pred_proba);
            report.print();
            
            ConfusionMatrix cm = Metrics::confusion_matrix(dataset.y_test, y_pred);
            cm.print();
            
            // Save model
            model.saveModel("titanic_model.txt");
            
            std::cout << "\nModel saved to 'titanic_model.txt'\n";
        } else {
            std::cout << "\nCSV file not found. Generating synthetic data...\n";
            
            // Option 2: Use synthetic data
            Dataset dataset = DataLoader::generateSyntheticData(1000, 10, 0.1, 42);
            
            // Create and train model
            LogisticRegression model(0.1, 500, 1e-4, true);
            
            std::cout << "\nTraining on synthetic data...\n";
            model.fit(dataset.X_train, dataset.y_train);
            
            // Make predictions
            VectorXd y_pred = model.predict(dataset.X_test);
            VectorXd y_pred_proba = model.predict_proba(dataset.X_test);
            
            // Evaluate
            double acc = model.score(dataset.X_test, dataset.y_test);
            std::cout << "\nAccuracy on test set: " << acc << std::endl;
            
            ClassificationReport report = Metrics::classification_report(
                dataset.y_test, y_pred, y_pred_proba);
            report.print();
            
            // Find best threshold
            double best_threshold = Metrics::find_best_threshold(
                dataset.y_test, y_pred_proba, "f1");
            std::cout << "\nBest threshold for F1 score: " << best_threshold << std::endl;
            
            // Save ROC curve data for plotting
            Metrics::save_roc_curve_data(dataset.y_test, y_pred_proba, "roc_curve.csv");
            std::cout << "ROC curve data saved to 'roc_curve.csv'\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

#pragma once
#include "Utils.h"
#include <vector>
#include <string>
#include <memory>

struct Dataset {
    MatrixXd X_train;
    VectorXd y_train;
    MatrixXd X_test;
    VectorXd y_test;
    
    void printSummary() const {
        std::cout << "Training set: " << X_train.rows() << " samples, " 
                  << X_train.cols() << " features\n";
        std::cout << "Test set: " << X_test.rows() << " samples, "
                  << X_test.cols() << " features\n";
    }
};

class DataLoader {
private:
    MatrixXd data;
    VectorXd labels;
    VectorXd feature_means;
    VectorXd feature_stds;
    bool is_normalized;
    bool has_labels;
    std::vector<std::string> feature_names;
    
    void removeRowsWithMissingValues();
    void convertCategoricalToNumerical();
    
public:
    DataLoader();
    
    // Load data from CSV file
    bool loadCSV(const std::string& filename, bool has_labels = true, 
                 char delimiter = ',', bool header = true);
    
    // Load data from Eigen matrices
    void loadData(const MatrixXd& X, const VectorXd& y);
    
    // Split data into train and test sets
    Dataset split(double train_ratio = 0.8, bool shuffle = true, int seed = 42);
    
    // Preprocessing methods
    void normalize();
    void standardize();
    void addBiasTerm();
    void removeOutliers(double threshold = 3.0);
    
    // For test data normalization (using training stats)
    void normalizeTestData(MatrixXd& X_test) const;
    void standardizeTestData(MatrixXd& X_test) const;
    
    // Generate synthetic data for testing
    static Dataset generateSyntheticData(int n_samples = 1000, int n_features = 10, 
                                         double noise = 0.1, int seed = 42);
    
    // Getters
    MatrixXd getData() const { return data; }
    VectorXd getLabels() const { return labels; }
    VectorXd getFeatureMeans() const { return feature_means; }
    VectorXd getFeatureStds() const { return feature_stds; }
    std::vector<std::string> getFeatureNames() const { return feature_names; }
    
    // Info
    void printInfo() const;
};

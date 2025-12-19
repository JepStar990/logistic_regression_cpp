#include "../include/DataLoader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

DataLoader::DataLoader() : is_normalized(false), has_labels(true) {}

bool DataLoader::loadCSV(const std::string& filename, bool has_labels, 
                        char delimiter, bool header) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    this->has_labels = has_labels;
    feature_names.clear();
    
    // Read file
    std::vector<std::vector<double>> rows;
    std::string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) continue;
        
        auto tokens = Utils::split(line, delimiter);
        
        if (line_num == 1 && header) {
            feature_names = tokens;
            if (has_labels && !tokens.empty()) {
                feature_names.pop_back(); // Last column is label
            }
            continue;
        }
        
        std::vector<double> row;
        try {
            for (size_t i = 0; i < tokens.size(); i++) {
                double value;
                if (tokens[i].empty() || tokens[i] == "NA" || tokens[i] == "null") {
                    value = std::numeric_limits<double>::quiet_NaN();
                } else {
                    value = std::stod(tokens[i]);
                }
                row.push_back(value);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse line " << line_num << ": " << e.what() << std::endl;
            continue;
        }
        
        rows.push_back(row);
    }
    
    file.close();
    
    if (rows.empty()) {
        std::cerr << "Error: No data loaded from file" << std::endl;
        return false;
    }
    
    // Convert to Eigen matrices
    int n_samples = rows.size();
    int n_features = rows[0].size() - (has_labels ? 1 : 0);
    
    data.resize(n_samples, n_features);
    if (has_labels) {
        labels.resize(n_samples);
    }
    
    // Handle missing values with mean imputation
    std::vector<double> col_sums(n_features, 0.0);
    std::vector<int> col_counts(n_features, 0);
    
    // First pass: compute column means for non-NaN values
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            if (!std::isnan(rows[i][j])) {
                col_sums[j] += rows[i][j];
                col_counts[j]++;
            }
        }
    }
    
    std::vector<double> col_means(n_features);
    for (int j = 0; j < n_features; j++) {
        col_means[j] = col_counts[j] > 0 ? col_sums[j] / col_counts[j] : 0.0;
    }
    
    // Second pass: fill data matrix and impute missing values
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            if (std::isnan(rows[i][j])) {
                data(i, j) = col_means[j];
            } else {
                data(i, j) = rows[i][j];
            }
        }
        if (has_labels) {
            labels(i) = rows[i][n_features]; // Last column is label
        }
    }
    
    std::cout << "Loaded " << n_samples << " samples with " << n_features 
              << " features from " << filename << std::endl;
    
    return true;
}

void DataLoader::loadData(const MatrixXd& X, const VectorXd& y) {
    data = X;
    labels = y;
    has_labels = true;
}

Dataset DataLoader::split(double train_ratio, bool shuffle, int seed) {
    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("train_ratio must be between 0 and 1");
    }
    
    int n_samples = data.rows();
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        std::mt19937 g(seed);
        std::shuffle(indices.begin(), indices.end(), g);
    }
    
    int train_size = static_cast<int>(n_samples * train_ratio);
    
    Dataset dataset;
    dataset.X_train.resize(train_size, data.cols());
    dataset.y_train.resize(train_size);
    dataset.X_test.resize(n_samples - train_size, data.cols());
    dataset.y_test.resize(n_samples - train_size);
    
    for (int i = 0; i < train_size; i++) {
        dataset.X_train.row(i) = data.row(indices[i]);
        if (has_labels) {
            dataset.y_train(i) = labels(indices[i]);
        }
    }
    
    for (int i = train_size; i < n_samples; i++) {
        dataset.X_test.row(i - train_size) = data.row(indices[i]);
        if (has_labels) {
            dataset.y_test(i - train_size) = labels(indices[i]);
        }
    }
    
    return dataset;
}

void DataLoader::normalize() {
    if (data.rows() == 0) return;
    
    feature_means = data.colwise().mean();
    feature_stds.resize(data.cols());
    
    for (int j = 0; j < data.cols(); j++) {
        double mean = feature_means(j);
        double sq_sum = 0.0;
        
        for (int i = 0; i < data.rows(); i++) {
            sq_sum += (data(i, j) - mean) * (data(i, j) - mean);
        }
        
        feature_stds(j) = std::sqrt(sq_sum / data.rows());
        
        if (feature_stds(j) > 1e-10) {
            for (int i = 0; i < data.rows(); i++) {
                data(i, j) = (data(i, j) - mean) / feature_stds(j);
            }
        }
    }
    
    is_normalized = true;
}

void DataLoader::standardize() {
    if (data.rows() == 0) return;
    
    feature_means = data.colwise().mean();
    
    for (int j = 0; j < data.cols(); j++) {
        double min_val = data.col(j).minCoeff();
        double max_val = data.col(j).maxCoeff();
        double range = max_val - min_val;
        
        if (range > 1e-10) {
            for (int i = 0; i < data.rows(); i++) {
                data(i, j) = (data(i, j) - min_val) / range;
            }
        }
    }
}

void DataLoader::addBiasTerm() {
    MatrixXd new_data(data.rows(), data.cols() + 1);
    new_data.col(0) = VectorXd::Ones(data.rows());
    new_data.block(0, 1, data.rows(), data.cols()) = data;
    data = new_data;
}

void DataLoader::normalizeTestData(MatrixXd& X_test) const {
    if (!is_normalized || X_test.rows() == 0) return;
    
    for (int j = 0; j < X_test.cols(); j++) {
        if (feature_stds(j) > 1e-10) {
            for (int i = 0; i < X_test.rows(); i++) {
                X_test(i, j) = (X_test(i, j) - feature_means(j)) / feature_stds(j);
            }
        }
    }
}

void DataLoader::standardizeTestData(MatrixXd& X_test) const {
    if (X_test.rows() == 0) return;
    
    // Note: This requires precomputed min/max from training
    // For simplicity, we'll use current implementation
    // In production, store training min/max
    for (int j = 0; j < X_test.cols(); j++) {
        double min_val = X_test.col(j).minCoeff();
        double max_val = X_test.col(j).maxCoeff();
        double range = max_val - min_val;
        
        if (range > 1e-10) {
            for (int i = 0; i < X_test.rows(); i++) {
                X_test(i, j) = (X_test(i, j) - min_val) / range;
            }
        }
    }
}

Dataset DataLoader::generateSyntheticData(int n_samples, int n_features, 
                                         double noise, int seed) {
    Utils::Random rng(seed);
    
    // Generate random weights for the true model
    VectorXd true_weights = VectorXd::Random(n_features) * 2.0;
    double true_bias = rng.uniform() * 2.0 - 1.0;
    
    // Generate random features
    MatrixXd X(n_samples, n_features);
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X(i, j) = rng.normal();
        }
    }
    
    // Generate labels using logistic model
    VectorXd y(n_samples);
    for (int i = 0; i < n_samples; i++) {
        double z = X.row(i).dot(true_weights) + true_bias;
        double prob = 1.0 / (1.0 + std::exp(-z));
        y(i) = (rng.uniform() < prob) ? 1.0 : 0.0;
    }
    
    // Add noise
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X(i, j) += rng.normal() * noise;
        }
    }
    
    Dataset dataset;
    int train_size = n_samples * 0.8;
    dataset.X_train = X.topRows(train_size);
    dataset.y_train = y.head(train_size);
    dataset.X_test = X.bottomRows(n_samples - train_size);
    dataset.y_test = y.tail(n_samples - train_size);
    
    return dataset;
}

void DataLoader::printInfo() const {
    std::cout << "\n=== Dataset Info ===\n";
    std::cout << "Samples: " << data.rows() << std::endl;
    std::cout << "Features: " << data.cols() << std::endl;
    
    if (has_labels) {
        std::cout << "Labels: " << labels.size() << std::endl;
        if (labels.size() > 0) {
            int positive = (labels.array() > 0.5).count();
            int negative = labels.size() - positive;
            std::cout << "Class distribution: " << positive << " positive, " 
                      << negative << " negative" << std::endl;
            std::cout << "Positive ratio: " << (double)positive/labels.size() << std::endl;
        }
    }
    
    if (data.rows() > 0) {
        std::cout << "\nFeature statistics:\n";
        for (int j = 0; j < std::min(5, (int)data.cols()); j++) {
            std::cout << "  Feature " << j << ": ";
            std::cout << "mean=" << data.col(j).mean() << ", ";
            std::cout << "std=" << std::sqrt((data.col(j).array() - data.col(j).mean()).square().sum() / data.rows()) << std::endl;
        }
        if (data.cols() > 5) {
            std::cout << "  ... and " << data.cols() - 5 << " more features\n";
        }
    }
}

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
    std::vector<std::vector<std::string>> rows;
    std::string line;
    int line_num = 0;
    
    // Read all lines
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) continue;
        
        // Parse CSV line with quoted fields
        std::vector<std::string> tokens;
        std::string token;
        bool in_quotes = false;
        
        for (char c : line) {
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == delimiter && !in_quotes) {
                tokens.push_back(token);
                token.clear();
            } else {
                token += c;
            }
        }
        tokens.push_back(token);  // Add last token
        
        if (line_num == 1 && header) {
            feature_names = tokens;
            if (has_labels && !tokens.empty()) {
                feature_names.pop_back(); // Last column is label
            }
            continue;
        }
        
        // Skip rows that don't have enough columns
        if (tokens.size() < 2) {
            std::cerr << "Warning: Skipping line " << line_num 
                      << " - insufficient columns" << std::endl;
            continue;
        }
        
        rows.push_back(tokens);
    }
    
    file.close();
    
    if (rows.empty()) {
        std::cerr << "Error: No data loaded from file" << std::endl;
        return false;
    }
    
    // Determine dimensions
    int n_samples = rows.size();
    int n_features = rows[0].size() - (has_labels ? 1 : 0);
    
    data.resize(n_samples, n_features);
    if (has_labels) {
        labels.resize(n_samples);
    }
    
    // Track which features are numeric
    std::vector<bool> is_numeric_feature(n_features, true);
    std::vector<std::vector<double>> numeric_data(n_features);
    
    // First pass: try to convert each column to numeric
    for (int j = 0; j < n_features; j++) {
        bool all_numeric = true;
        for (int i = 0; i < n_samples; i++) {
            std::string val = rows[i][j];
            
            // Remove quotes if present
            if (!val.empty() && val.front() == '"' && val.back() == '"') {
                val = val.substr(1, val.size() - 2);
            }
            
            // Skip empty values
            if (val.empty() || val == "NA" || val == "null" || val == "nan") {
                numeric_data[j].push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            
            // Try to convert to double
            try {
                double num_val = std::stod(val);
                numeric_data[j].push_back(num_val);
            } catch (...) {
                // Conversion failed - mark as non-numeric
                all_numeric = false;
                break;
            }
        }
        
        is_numeric_feature[j] = all_numeric;
    }
    
    // Count how many numeric features we have
    int numeric_feature_count = 0;
    for (bool is_numeric : is_numeric_feature) {
        if (is_numeric) numeric_feature_count++;
    }
    
    std::cout << "Found " << numeric_feature_count << " numeric features out of " 
              << n_features << " total features\n";
    
    // Resize data for only numeric features
    data.resize(n_samples, numeric_feature_count);
    
    // Second pass: fill numeric features
    int current_feature = 0;
    for (int j = 0; j < n_features; j++) {
        if (is_numeric_feature[j]) {
            // Handle missing values with mean imputation
            double sum = 0.0;
            int count = 0;
            
            // Compute mean of non-NaN values
            for (double val : numeric_data[j]) {
                if (!std::isnan(val)) {
                    sum += val;
                    count++;
                }
            }
            double mean = count > 0 ? sum / count : 0.0;
            
            // Fill column with imputed values
            for (int i = 0; i < n_samples; i++) {
                double val = numeric_data[j][i];
                if (std::isnan(val)) {
                    data(i, current_feature) = mean;
                } else {
                    data(i, current_feature) = val;
                }
            }
            current_feature++;
        }
    }
    
    // Load labels if present
    if (has_labels) {
        int label_col = n_features; // Labels are in the last column
        

        label_col = 1;

        for (int i = 0; i < n_samples; i++) {
            std::string label_str = rows[i][label_col];
            
            // Simple label encoding: assume "0", "1" or similar
            try {
                labels(i) = std::stod(label_str);
            } catch (...) {
                // Try to parse common label formats
                if (label_str == "0" || label_str == "false" || label_str == "False" || 
                    label_str == "FALSE" || label_str == "no" || label_str == "No") {
                    labels(i) = 0.0;
                } else if (label_str == "1" || label_str == "true" || label_str == "True" ||
                          label_str == "TRUE" || label_str == "yes" || label_str == "Yes") {
                    labels(i) = 1.0;
                } else {
                    // Default to 0 if can't parse
                    labels(i) = 0.0;
                }
            }
        }
    }
    
    std::cout << "Successfully loaded " << n_samples << " samples with " 
              << numeric_feature_count << " numeric features from " << filename << std::endl;
    
    // Print summary of first few samples
    if (n_samples > 0) {
        std::cout << "\nFirst 3 samples:\n";
        for (int i = 0; i < std::min(3, n_samples); i++) {
            std::cout << "Sample " << i << ": ";
            for (int j = 0; j < std::min(5, numeric_feature_count); j++) {
                std::cout << data(i, j) << " ";
            }
            if (has_labels) {
                std::cout << " -> Label: " << labels(i);
            }
            std::cout << std::endl;
        }
    }
    
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

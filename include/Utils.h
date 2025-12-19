#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXi;

namespace Utils {
    
    class Timer {
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        
    public:
        Timer() { start(); }
        
        void start() {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        double elapsed() {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            return elapsed.count();
        }
    };
    
    class Random {
    private:
        std::mt19937 generator;
        std::uniform_real_distribution<double> uniform_dist;
        std::normal_distribution<double> normal_dist;
        
    public:
        Random(int seed = 42) : generator(seed),
                                uniform_dist(0.0, 1.0),
                                normal_dist(0.0, 1.0) {}
        
        double uniform() { return uniform_dist(generator); }
        double normal() { return normal_dist(generator); }
        int randint(int min, int max) {
            std::uniform_int_distribution<int> dist(min, max);
            return dist(generator);
        }
    };

    static std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        bool in_quotes = false;
    
        for (char c : s) {
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == delimiter && !in_quotes) {
                tokens.push_back(token);
                token.clear();
            } else {
                token += c;
            }
        }
        tokens.push_back(token);
    
        return tokens;
    }
    
    static bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
}

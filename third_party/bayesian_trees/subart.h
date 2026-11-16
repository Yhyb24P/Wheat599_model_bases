// Adapted from suBART deddf4ea (MIT); see LICENSE.subart.
#pragma once
#include<RcppArmadillo.h>
#include<vector>
struct Node;
struct modelParam;
struct modelParam {
        arma::mat x_train;
        arma::mat y_mat;
        arma::mat x_test;
        arma::mat xcut;
        int n_tree;
        int d_var;
        double alpha;
        double beta;
        arma::vec sigma_mu;
        arma::mat Sigma;
        arma::mat S_0_wish;
        arma::vec a_j_vec;
        arma::vec A_j_vec;
        arma::mat W;
        arma::mat R;
        arma::mat D;
        bool sv_bool;
        arma::mat sv_matrix;
        double nu;
        int node_min_size;
        int n_mcmc;
        int n_burn;
        arma::vec move_proposal;
        arma::vec move_acceptance;
        arma::vec categorical_indicators;
        bool categorical_indicators_bool;
        double v_j;
        double sigma_mu_j;
        modelParam(arma::mat x_train_,
                   arma::mat y_mat_,
                   arma::mat x_test_,
                   arma::mat x_cut_,
                   int n_tree_,
                   int node_min_size_,
                   double alpha_,
                   double beta_,
                   double nu_,
                   arma::vec sigma_mu_,
                   arma::mat Sigma_,
                   arma::mat S_0_wish_,
                   arma::vec A_j_vec_,
                   double n_mcmc_,
                   double n_burn_,
                   bool sv_bool_,
                   arma::mat sv_matrix_,
                   arma::vec categorical_indicators_);
};
class Forest {
public:
        std::vector<Node*> trees;
        Forest(modelParam &data);
};
struct Node {
     bool isRoot;
     bool isLeaf;
     Node* left;
     Node* right;
     Node* parent;
     arma::vec train_index;
     arma::vec test_index;
     int var_split;
     double var_split_rule;
     double lower;
     double upper;
     double curr_weight;
     int depth = 0;
     double mu;
     arma::uvec split_variables;
     double log_likelihood = 0.0;
     double r_sum = 0.0;
     double u_sum = 0.0;
     double Gamma_j;
     double sigma_mu_j_sq;
     double S_j;
     int n_leaf = 0;
     int n_leaf_test = 0;
     void addingLeaves(modelParam& data);
     void deletingLeaves();
     void Stump(modelParam& data);
     void updateWeight(const arma::mat X, int i);
     void getLimits();
     bool isLeft();
     bool isRight();
     void grow(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec &curr_u);
     void prune(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u);
     void change(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u);
     void nodeLogLike(modelParam &data);
     void updateResiduals(modelParam& data, arma::vec &curr_res, arma::vec &curr_u);
     void updateResiduals_uni(modelParam& data, arma::vec &curr_res);
     void displayCurrNode();
     Node(modelParam &data);
     ~Node();
};
void leaves(Node* x, std::vector<Node*>& leaves);
std::vector<Node*> leaves(Node*x);

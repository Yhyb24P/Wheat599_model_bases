// Adapted from MateusMaiaDS/subart deddf4ea (MIT); see LICENSE.subart.
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
#include "subart.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
modelParam::modelParam(arma::mat x_train_,
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
                        arma:: mat sv_matrix_,
                        arma::vec categorical_indicators_){
        x_train = x_train_;
        y_mat = y_mat_;
        x_test = x_test_;
        xcut = x_cut_;
        n_tree = n_tree_;
        node_min_size = node_min_size_;
        alpha = alpha_;
        beta = beta_;
        nu = nu_;
        sigma_mu = sigma_mu_;
        Sigma = Sigma_;
        S_0_wish = S_0_wish_;
        A_j_vec = A_j_vec_;
        a_j_vec = arma::vec(y_mat_.n_cols,arma::fill::zeros);
        n_mcmc = n_mcmc_;
        n_burn = n_burn_;
        sv_bool = sv_bool_;
        sv_matrix = sv_matrix_;
        R = Sigma_;
        D = arma::eye(y_mat_.n_cols,y_mat_.n_cols);
        move_proposal = arma::vec(3,arma::fill::zeros);
        move_acceptance = arma::vec(3,arma::fill::zeros);
        categorical_indicators = categorical_indicators_;
        categorical_indicators_bool = sum(categorical_indicators_)==0 ? false: true;
}
Node::Node(modelParam &data){
        isLeaf = true;
        isRoot = true;
        left = NULL;
        right = NULL;
        parent = NULL;
        train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
        test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1) ;
        var_split = -1;
        var_split_rule = -1.0;
        lower = 0.0;
        upper = 1.0;
        mu = 0.0;
        n_leaf = 0.0;
        n_leaf_test = 0;
        log_likelihood = 0.0;
        depth = 0;
}
Node::~Node() {
        if(!isLeaf) {
                delete left;
                delete right;
        }
}
void Node::Stump(modelParam& data){
        left = this;
        right = this;
        parent = this;
        for(int i=0; i<data.x_train.n_rows;i++){
                train_index[i] = i;
        }
        for(int i=0; i<data.x_test.n_rows;i++){
                test_index[i] = i;
        }
}
void Node::addingLeaves(modelParam& data){
     left = new Node(data);
     right = new Node(data);
     isLeaf = false;
     left -> isRoot = false;
     left -> isLeaf = true;
     left -> left = left;
     left -> right = left;
     left -> parent = this;
     left -> var_split = 0;
     left -> var_split_rule = -1.0;
     left -> lower = 0.0;
     left -> upper = 1.0;
     left -> mu = 0.0;
     left -> log_likelihood = 0.0;
     left -> n_leaf = 0.0;
     left -> depth = depth+1;
     left -> train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
     left -> test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1);
     right -> isRoot = false;
     right -> isLeaf = true;
     right -> left = right;
     right -> right = right;
     right -> parent = this;
     right -> var_split = 0;
     right -> var_split_rule = -1.0;
     right -> lower = 0.0;
     right -> upper = 1.0;
     right -> mu = 0.0;
     right -> log_likelihood = 0.0;
     right -> n_leaf = 0.0;
     right -> depth = depth+1;
     right -> train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
     right -> test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1);
     return;
}
bool Node::isLeft(){
        return (this == this->parent->left);
}
bool Node::isRight(){
        return (this == this->parent->right);
}
void Node::getLimits(){
        Node* x = this;
        lower = 0.0;
        upper = 1.0;
        bool tree_iter = x->isRoot ? false: true;
        while(tree_iter){
                bool is_left = x->isLeft();
                x = x->parent;
                tree_iter = x->isRoot ? false : true;
                if(x->var_split == var_split){
                        tree_iter = false ;
                        if(is_left){
                                upper = x->var_split_rule;
                                lower = x->lower;
                        } else {
                                upper = x->upper;
                                lower = x->var_split_rule;
                        }
                }
        }
}
void Node::deletingLeaves(){
     delete left;
     delete right;
     left = this;
     right = this;
     isLeaf = true;
     return;
}
void get_leaves(Node* x,  std::vector<Node*> &leaves_vec) {
        if(x->isLeaf){
                leaves_vec.push_back(x);
        } else {
                get_leaves(x->left, leaves_vec);
                get_leaves(x->right,leaves_vec);
        }
        return;
}
std::vector<Node*> leaves(Node* x) {
        std::vector<Node*> leaves_init(0);
        get_leaves(x,leaves_init);
        return(leaves_init);
}
void get_nogs(std::vector<Node*>& nogs, Node* node){
        if(!node->isLeaf){
                bool bool_left_is_leaf = node->left->isLeaf;
                bool bool_right_is_leaf = node->right->isLeaf;
                if(bool_left_is_leaf && bool_right_is_leaf){
                        nogs.push_back(node);
                } else {
                        get_nogs(nogs, node->left);
                        get_nogs(nogs, node->right);
                }
        }
}
void collect_split_vars(arma::vec& var_count, Node* tree){
        if(!tree->isLeaf){
                var_count[tree->var_split] = var_count[tree->var_split] + 1;
                collect_split_vars(var_count, tree->left);
                collect_split_vars(var_count,tree->right);
        }
}
std::vector<Node*> nogs(Node* tree){
        std::vector<Node*> nogs_init(0);
        get_nogs(nogs_init,tree);
        return nogs_init;
}
Forest::Forest(modelParam& data){
        trees.resize(data.n_tree*data.y_mat.n_cols);
        for(int  i=0;i<(data.n_tree*data.y_mat.n_cols);i++){
                trees[i] = new Node(data);
                trees[i]->Stump(data);
        }
}
void Node::updateResiduals(modelParam& data,
                           arma::vec &curr_res,
                           arma::vec &curr_u){
        if(isRoot){
                n_leaf = data.x_train.n_rows;
                n_leaf_test = data.x_test.n_rows;
        }
        if(train_index[0]==-1){
                n_leaf = 0;
                r_sum = 0;
                log_likelihood = -2000000;
                return;
        }
        if(n_leaf < data.node_min_size){
                return;
        }
        r_sum = 0.0;
        u_sum = 0.0;
        for(int i = 0; i < n_leaf;i++){
                r_sum = r_sum + curr_res(train_index[i]);
                u_sum = u_sum + curr_u(train_index[i]);
        }
        sigma_mu_j_sq = data.sigma_mu_j*data.sigma_mu_j;
        Gamma_j  = n_leaf+data.v_j/sigma_mu_j_sq;
        S_j = r_sum-u_sum;
        return;
}
void Node::nodeLogLike(modelParam& data){
        log_likelihood = 0.5*log(data.v_j/(sigma_mu_j_sq*Gamma_j)) +0.5*(S_j*S_j)/(data.v_j*Gamma_j);
        return;
}
void updateMuPredictions(Node* tree, modelParam &data,
                         arma::vec &current_prediction_train,
                         arma::vec &current_prediction_test){
        std::vector<Node*> t_nodes = leaves(tree);
        for(int i = 0; i < t_nodes.size();i++){
                t_nodes[i]->mu = R::rnorm((t_nodes[i]->S_j)/(t_nodes[i]->Gamma_j),sqrt(data.v_j/(t_nodes[i]->Gamma_j))) ;
                bool update_train_ = true;
                bool update_test_ = true;
                int for_size_;
                if(data.x_train.n_rows > data.x_test.n_rows){
                        for_size_ = data.x_train.n_rows;
                } else {
                        for_size_ = data.x_test.n_rows;
                }
                for(int ii = 0; ii < for_size_ ; ii++){
                        if(ii >= data.x_train.n_rows){
                                update_train_ = false;
                        } else {
                                if(t_nodes[i]->train_index[ii] == -1){
                                        update_train_ = false;
                                }
                        }
                        if(ii >= data.x_test.n_rows){
                                update_test_ = false;
                        } else {
                                if(t_nodes[i]->test_index[ii] == -1){
                                        update_test_ = false;
                                }
                        }
                        if(update_train_){
                                current_prediction_train[t_nodes[i]->train_index[ii]] = t_nodes[i] -> mu;
                        }
                        if(!(update_train_) & !(update_test_)){
                                break;
                        }
                        if(t_nodes[i]->n_leaf_test == 0 ){
                                continue;
                        }
                        if(update_test_){
                                current_prediction_test[t_nodes[i]->test_index[ii]] = t_nodes[i] -> mu;
                        }
                }
        }
}
void updateSigma(arma::mat &y_mat_hat,
                 modelParam &data){
        arma::mat S(data.y_mat.n_cols,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat residuals_mat = y_mat_hat-data.y_mat;
        S = residuals_mat.t()*residuals_mat;
        data.Sigma = arma::iwishrnd((data.S_0_wish+S),data.nu + data.y_mat.n_cols - 1 + data.y_mat.n_rows);
}
void update_a_j(modelParam &data){
        double shape_j = 0.5*(data.y_mat.n_cols+data.nu);
        arma::mat Precision = arma::inv(data.Sigma);
        for(int j = 0; j < data.y_mat.n_cols; j++){
                double scale_j = 1/(data.A_j_vec(j)*data.A_j_vec(j))+data.nu*Precision(j,j);
                double a_j_vec_double_aux = R::rgamma(shape_j,1/scale_j);
                data.a_j_vec(j) = 1/a_j_vec_double_aux;
                data.S_0_wish(j,j) = (2*data.nu)/data.a_j_vec(j);
        }
        return;
}

// Binary split support depends only on fitting markers, never on query values.
void split_support(Node* node, const modelParam& d) {
    std::vector<arma::uword> valid;
    for (arma::uword j=0;j<d.x_train.n_cols;++j) {
        int ones=0;
        for (int k=0;k<node->n_leaf;++k) ones+=d.x_train(node->train_index(k),j)>.5;
        if (ones>=d.node_min_size && node->n_leaf-ones>=d.node_min_size) valid.push_back(j);
    }
    node->split_variables=arma::uvec(valid);
}

double lse(const arma::vec& a) {
    double m=a.max();
    return m+std::log(arma::accu(arma::exp(a-m)));
}
int categorical(const arma::vec& logp) {
    arma::vec probs=arma::exp(logp-logp.max());
    double u=R::runif(0,arma::accu(probs)),total=0;
    for (arma::uword j=0;j<probs.n_elem;++j) {total+=probs(j);if (u<total) return j;}
    return probs.n_elem-1;
}
double split_probability(Node* n,const modelParam& d) {
    return n->split_variables.empty()?0:d.alpha/std::pow(1+n->depth,d.beta);
}
void allocate_children(Node* n,modelParam& d,int variable) {
    n->addingLeaves(d);n->var_split=variable;n->var_split_rule=.5;
    for (int k=0;k<n->n_leaf;++k) {
        int i=n->train_index(k);Node* child=d.x_train(i,variable)<=.5?n->left:n->right;
        child->train_index(child->n_leaf++)=i;
    }
    for (int k=0;k<n->n_leaf_test;++k) {
        int i=n->test_index(k);Node* child=d.x_test(i,variable)<=.5?n->left:n->right;
        child->test_index(child->n_leaf_test++)=i;
    }
    split_support(n->left,d);split_support(n->right,d);
}
double integrated(Node* n,modelParam& d,arma::vec& r,arma::vec& u) {
    n->updateResiduals(d,r,u);n->nodeLogLike(d);return n->log_likelihood;
}

// Equal birth/death probabilities include rejected death proposals at a stump.
// Split-variable proposal and prior coincide, so their normalized factors cancel.
void tree_step(Node* root,modelParam& d,arma::vec& r,arma::vec& u,const arma::vec& logp) {
    auto terminal=leaves(root);
    if (R::runif(0,1)<.5) {
        d.move_proposal(0)++;
        Node* n=terminal[static_cast<int>(R::runif(0,terminal.size()))];
        if (n->split_variables.empty()) return;
        double old=integrated(n,d,r,u),p=split_probability(n,d);
        int j=n->split_variables(categorical(logp.elem(n->split_variables)));
        allocate_children(n,d,j);
        double logratio=integrated(n->left,d,r,u)+integrated(n->right,d,r,u)-old;
        logratio+=std::log(p)+std::log1p(-split_probability(n->left,d))+
            std::log1p(-split_probability(n->right,d))-std::log1p(-p);
        logratio+=std::log(static_cast<double>(terminal.size())/nogs(root).size());
        if (std::log(R::runif(0,1))<logratio) d.move_acceptance(0)++;
        else n->deletingLeaves();
    } else {
        d.move_proposal(1)++;
        auto prunable=nogs(root);
        if (prunable.empty()) return;
        Node* n=prunable[static_cast<int>(R::runif(0,prunable.size()))];
        double p=split_probability(n,d);
        double logratio=integrated(n,d,r,u)-integrated(n->left,d,r,u)-integrated(n->right,d,r,u);
        logratio+=std::log1p(-p)-std::log(p)-std::log1p(-split_probability(n->left,d))-
            std::log1p(-split_probability(n->right,d));
        logratio+=std::log(static_cast<double>(prunable.size())/(terminal.size()-1));
        if (std::log(R::runif(0,1))<logratio) {n->deletingLeaves();d.move_acceptance(1)++;}
    }
}
void internal_nodes(Node* n,std::vector<Node*>& nodes) {
    if (!n->isLeaf) {nodes.push_back(n);internal_nodes(n->left,nodes);internal_nodes(n->right,nodes);}
}
arma::vec log_dirichlet(const arma::vec& shape) {
    arma::vec v(shape.n_elem);
    for (arma::uword j=0;j<v.n_elem;++j) {
        double s=shape(j);
        v(j)=s<1?std::log(R::rgamma(s+1,1))+std::log(R::runif(0,1))/s:std::log(R::rgamma(s,1));
    }
    return v-lse(v);
}
double legal_normalizers(const arma::vec& logp,const std::vector<Node*>& nodes) {
    double v=0;for (Node* n:nodes) v+=lse(logp.elem(n->split_variables));return v;
}
double concentration_prior(double a,int p) {
    double u=a/(a+p);
    return .5*std::log(u)+std::log1p(-u); // Beta(1/2,1), including the log-a Jacobian.
}
double split_count_marginal(double a,const arma::vec& counts) {
    double v=std::lgamma(a)-std::lgamma(a+arma::accu(counts));
    for (double count:counts) if(count>0) v+=std::lgamma(a/counts.n_elem+count)-std::lgamma(a/counts.n_elem);
    return v;
}
int probability_step(arma::vec& logp,double concentration,const std::vector<Node*>& nodes) {
    arma::vec count(logp.n_elem,arma::fill::zeros);for (Node* n:nodes) count(n->var_split)++;
    double correction=legal_normalizers(logp,nodes);int accepted=0;
    for (int k=0;k<5;++k) {
        arma::vec proposal=log_dirichlet(count+concentration/logp.n_elem);
        double next=legal_normalizers(proposal,nodes);
        if (std::log(R::runif(0,1))<correction-next) {logp=proposal;correction=next;accepted++;}
    }
    return accepted;
}
int update_dart(arma::vec& logp,double& concentration,const std::vector<Node*>& nodes) {
    int accepted=probability_step(logp,concentration,nodes);
    arma::vec count(logp.n_elem,arma::fill::zeros);for (Node* n:nodes) count(n->var_split)++;
    double next=std::exp(R::rnorm(std::log(concentration),.5));
    if (!std::isfinite(next) || next<=0) return accepted;
    // Joint concentration/probability move avoids high-dimensional Gibbs locking.
    arma::vec proposal=log_dirichlet(count+next/logp.n_elem);
    double ratio=split_count_marginal(next,count)-split_count_marginal(concentration,count)+
        concentration_prior(next,logp.n_elem)-concentration_prior(concentration,logp.n_elem)+
        legal_normalizers(logp,nodes)-legal_normalizers(proposal,nodes);
    if (std::log(R::runif(0,1))<ratio) {logp=proposal;concentration=next;}
    return accepted;
}

// Bhattacharya-style observation-space Gaussian draw; B prior is independent of Sigma.
arma::mat gaussian_background(const arma::mat& z,const arma::mat& residual,const arma::mat& sigma,
                              double variance,const arma::mat& eigenvectors,const arma::vec& eigenvalues) {
    arma::vec se;arma::mat rotation;arma::eig_sym(se,rotation,sigma);
    arma::mat out(z.n_cols,residual.n_cols),y=residual*rotation;
    for (arma::uword e=0;e<se.n_elem;++e) {
        arma::vec prior=std::sqrt(variance)*arma::randn<arma::vec>(z.n_cols);
        arma::vec noise=std::sqrt(se(e))*arma::randn<arma::vec>(z.n_rows);
        arma::vec v=eigenvectors.t()*(y.col(e)-z*prior-noise);
        v/=(se(e)+variance*eigenvalues);
        out.col(e)=prior+variance*z.t()*(eigenvectors*v);
    }
    return out*rotation.t();
}

// [[Rcpp::export]]
Rcpp::List sparse_polygenic_bart(arma::mat x,arma::mat y,arma::mat query,int trees,double kappa,
                               int warmup,int draws,int seed) {
    if (x.n_rows!=y.n_rows || query.n_cols!=x.n_cols || y.n_cols!=4 || trees<1 || kappa<=0 || warmup<0 || draws<1)
        Rcpp::stop("Invalid sampler dimensions or settings");
    if (!x.is_finite() || !y.is_finite() || !query.is_finite() || arma::any(arma::vectorise((x!=0)%(x!=1))) ||
        arma::any(arma::vectorise((query!=0)%(query!=1)))) Rcpp::stop("Finite binary markers required");
    Rcpp::Function set_seed("set.seed");set_seed(seed);
    int n=x.n_rows,p=x.n_cols,E=y.n_cols,total=warmup+draws;
    arma::rowvec xm=arma::mean(x,0),xs=arma::stddev(x,1,0),ym=arma::mean(y,0),ys=arma::stddev(y,1,0);
    xs.transform([](double v){return v>0?v:1.;});
    if (ys.min()<=0) Rcpp::stop("Constant fit phenotype");
    arma::mat z=(x.each_row()-xm).each_row()/xs,zq=(query.each_row()-xm).each_row()/xs;
    y=(y.each_row()-ym).each_row()/ys;
    arma::vec sd=(arma::max(y,0)-arma::min(y,0)).t()/(2*kappa*std::sqrt(trees));
    arma::vec A(E);for(int e=0;e<E;++e) A(e)=arma::stddev(y.col(e))/R::qt(.95,3,1,0);
    modelParam data(x,y,query,arma::mat(),trees,5,.95,2,3,sd,arma::eye(E,E),arma::eye(E,E),A,
                    total,warmup,false,arma::ones(E,p),arma::zeros(p));
    Forest forest(data);
    // Own the mature raw-pointer nodes even when an interrupt unwinds the sampler.
    std::vector<std::unique_ptr<Node>> owners;
    for (Node* root:forest.trees) {owners.emplace_back(root);root->n_leaf=n;root->n_leaf_test=query.n_rows;split_support(root,data);}
    arma::vec eig;arma::mat U;arma::eig_sym(eig,U,z*z.t());eig=arma::clamp(eig,0,arma::datum::inf);
    arma::mat coef(p,E,arma::fill::zeros),linear(n,E,arma::fill::zeros),linear_q(query.n_rows,E,arma::fill::zeros);
    arma::rowvec mu(E,arma::fill::zeros);
    arma::mat fit(n,E,arma::fill::zeros),qfit(query.n_rows,E,arma::fill::zeros);
    arma::cube treefit(n,trees,E,arma::fill::zeros),qtreefit(query.n_rows,trees,E,arma::fill::zeros);
    arma::mat logp(p,E);logp.fill(-std::log(p));arma::vec theta(E,arma::fill::ones);
    arma::cube posterior(query.n_rows,E,draws),training(n,E,draws),sigma(E,E,draws);
    // Four concentration parameters, split counts, DART acceptance, and component RMS.
    arma::mat trace(draws,20);arma::mat mean_coefficient(p,E,arma::fill::zeros);
    arma::rowvec mean_intercept(E,arma::fill::zeros);
    for (int iteration=0;iteration<total;++iteration) {
        if (iteration%32==0) Rcpp::checkUserInterrupt();
        arma::mat background_residual=y-fit;background_residual.each_row()-=mu;
        coef=gaussian_background(z,background_residual,data.Sigma,1./p,U,eig);
        linear=z*coef;linear_q=zq*coef;
        arma::vec sv;arma::mat sr;arma::eig_sym(sv,sr,data.Sigma);
        arma::vec mv=sr.t()*arma::mean(y-fit-linear,0).t();
        for(int e=0;e<E;++e) mv(e)=mv(e)*n/(n+sv(e)/100)+R::rnorm(0,std::sqrt(sv(e)/(n+sv(e)/100)));
        mu=(sr*mv).t();
        data.y_mat=y-linear;data.y_mat.each_row()-=mu;
        for (int e=0;e<E;++e) {
            arma::mat precision=arma::inv_sympd(data.Sigma);
            arma::vec u(n,arma::fill::zeros);
            for(int j=0;j<E;++j) if(j!=e) u-=(precision(e,j)/precision(e,e))*(data.y_mat.col(j)-fit.col(j));
            data.v_j=1/precision(e,e);data.sigma_mu_j=sd(e);
            arma::vec probabilities=logp.col(e);
            for (int t=0;t<trees;++t) {
                Node* root=forest.trees[t+e*trees];
                arma::vec residual=data.y_mat.col(e)-fit.col(e)+treefit.slice(e).col(t);
                tree_step(root,data,residual,u,probabilities);
                for (Node* leaf:leaves(root)) integrated(leaf,data,residual,u);
                arma::vec a(n,arma::fill::zeros),b(query.n_rows,arma::fill::zeros);
                updateMuPredictions(root,data,a,b);
                fit.col(e)+=a-treefit.slice(e).col(t);qfit.col(e)+=b-qtreefit.slice(e).col(t);
                treefit.slice(e).col(t)=a;qtreefit.slice(e).col(t)=b;
            }
            std::vector<Node*> nodes;for(int t=0;t<trees;++t) internal_nodes(forest.trees[t+e*trees],nodes);
            int accepted=update_dart(probabilities,theta(e),nodes);logp.col(e)=probabilities;
            if (iteration>=warmup) {
                int s=iteration-warmup;trace(s,e)=theta(e);trace(s,4+e)=nodes.size();trace(s,8+e)=accepted/5.;
                trace(s,12+e)=std::sqrt(arma::mean(arma::square(linear.col(e))));
                trace(s,16+e)=std::sqrt(arma::mean(arma::square(fit.col(e))));
            }
        }
        update_a_j(data);updateSigma(fit,data);
        if (!data.Sigma.is_finite() || !fit.is_finite() || !coef.is_finite()) Rcpp::stop("Nonfinite MCMC state");
        if (iteration>=warmup) {
            int s=iteration-warmup;
            training.slice(s)=fit+linear;training.slice(s).each_row()+=mu;
            training.slice(s).each_row()%=ys;training.slice(s).each_row()+=ym;
            posterior.slice(s)=qfit+linear_q;posterior.slice(s).each_row()+=mu;
            posterior.slice(s).each_row()%=ys;posterior.slice(s).each_row()+=ym;
            sigma.slice(s)=data.Sigma;mean_coefficient+=coef/draws;mean_intercept+=mu/draws;
        }
    }
    return Rcpp::List::create(Rcpp::Named("query")=posterior,Rcpp::Named("training")=training,
        Rcpp::Named("sigma")=sigma,Rcpp::Named("trace")=trace,Rcpp::Named("mean_coefficient")=mean_coefficient,
        Rcpp::Named("mean_intercept")=mean_intercept,Rcpp::Named("moves_proposed")=data.move_proposal,
        Rcpp::Named("moves_accepted")=data.move_acceptance);
}

// [[Rcpp::export]]
Rcpp::List gaussian_block_check(arma::mat z,arma::mat y,arma::mat sigma,double variance,int draws) {
    arma::vec eig;arma::mat U;arma::eig_sym(eig,U,z*z.t());
    arma::mat samples(z.n_cols*y.n_cols,draws);
    for(int k=0;k<draws;++k) samples.col(k)=arma::vectorise(gaussian_background(z,y,sigma,variance,U,eig));
    return Rcpp::List::create(Rcpp::Named("mean")=arma::mean(samples,1),Rcpp::Named("covariance")=arma::cov(samples.t()));
}

// [[Rcpp::export]]
Rcpp::List sampler_kernel_checks(int steps) {
    arma::mat x(20,1,arma::fill::zeros);x.rows(10,19).ones();
    arma::mat y(20,4,arma::fill::zeros),q=x.rows(0,1);
    modelParam d(x,y,q,arma::mat(),1,5,.95,2,3,arma::ones(4)*.5,arma::eye(4,4),
                 arma::eye(4,4),arma::ones(4),steps,0,false,arma::ones(4,1),arma::zeros(1));
    Node root(d);root.Stump(d);root.n_leaf=20;root.n_leaf_test=2;split_support(&root,d);
    d.v_j=1;d.sigma_mu_j=.5;
    arma::vec r=arma::linspace<arma::vec>(-.4,.4,20),u(20,arma::fill::zeros),logp(1,arma::fill::zeros);
    double old=integrated(&root,d,r,u);allocate_children(&root,d,0);
    double ratio=std::exp(integrated(root.left,d,r,u)+integrated(root.right,d,r,u)-old)*.95/.05;
    root.deletingLeaves();double grown=0;
    for (int k=0;k<steps;++k) {tree_step(&root,d,r,u,logp);grown+=!root.isLeaf;}
    // With only variable zero legal, its pi factor cancels exactly: posterior pi is its prior.
    Node fixed(d);fixed.var_split=0;fixed.split_variables=arma::uvec({0});
    std::vector<Node*> nodes={&fixed};arma::vec lp(2);lp.fill(-std::log(2.));
    double first=0,second=0;
    for (int k=0;k<steps;++k) {probability_step(lp,1,nodes);double v=std::exp(lp(0));first+=v;second+=v*v;}
    double concentration=1,prior_u=0;std::vector<Node*> empty;
    arma::mat covariance_mean(4,4,arma::fill::zeros),fitted(20,4,arma::fill::zeros);
    fitted.col(0)=r;
    for (int k=0;k<steps;++k) {
        update_dart(lp,concentration,empty);prior_u+=concentration/(concentration+2);
        updateSigma(fitted,d);covariance_mean+=d.Sigma/steps;
    }
    arma::mat exact=(d.S_0_wish+fitted.t()*fitted)/(d.nu+d.y_mat.n_rows-2);
    return Rcpp::List::create(Rcpp::Named("split_frequency")=grown/steps,
        Rcpp::Named("split_exact")=ratio/(1+ratio),Rcpp::Named("dirichlet_mean")=first/steps,
        Rcpp::Named("dirichlet_second_moment")=second/steps,
        Rcpp::Named("concentration_prior_mean_u")=prior_u/steps,
        Rcpp::Named("covariance_max_error")=arma::abs(covariance_mean-exact).max());
}

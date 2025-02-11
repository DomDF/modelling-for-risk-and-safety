data {
  int <lower = 0> n_trials;                     // number of observations
  vector [n_trials] depth;                      // damage depth
  vector [n_trials] length;                     // damage length
  array [n_trials] int <lower = 0, upper = 1> indication;   // did the inspection find the damage?
}

parameters {
  real alpha;       // log-odds intercept
  real beta_depth;  // log-odds coefficient for depth
  real beta_length; // log-odds coefficient for length
}

model {
  // priors
  alpha ~ normal(-6, 2);
  beta_depth ~ normal(1.5, 0.5);
  beta_length ~ normal(1.5, 0.5);
  
  // likelihood
  indication ~ bernoulli_logit(alpha + beta_depth * depth + beta_length * length);
}

generated quantities {
  vector[n_trials] log_lik;
  vector[n_trials] p;
  
  for (i in 1:n_trials) {
    p[i] = inv_logit(alpha + beta_depth * depth[i] + beta_length * length[i]);
    log_lik[i] = bernoulli_logit_lpmf(indication[i] | alpha + beta_depth * depth[i] + beta_length * length[i]);
  }
}
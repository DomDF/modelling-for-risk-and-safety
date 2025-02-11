data {
  int <lower = 0> n_anomalies;  // number of anomalies
  int <lower = 0> n_inspections;  // number of inspections
  vector[n_anomalies * n_inspections] cgr;  // growth rate observations (2 per anomaly)
}

parameters {
  real<lower = 0> mu;  // mean growth rate
  real<lower = 0> sigma;  // standard deviation
}

model {  
  // model
  for(i in 1:(n_anomalies * n_inspections)){
    cgr[i] ~ normal(mu, sigma);
  }
  /*
  //alternative (vectorised) implementation:
  delta_C ~ normal(mu, sigma);

  //some suggested priors
  mu ~ normal(1/4, 3);
  sigma ~ exponential(1);
  */
  
}

generated quantities {
  vector[n_anomalies * n_inspections] cgr_pred;
  //vector[n_anomalies * n_inspections] log_lik;

  for (i in 1:(n_anomalies * n_inspections)) {
    cgr_pred[i] = normal_rng(mu, sigma);
    //log_lik[i] = normal_lpdf(delta_C[i] | mu, sigma);
  }
}

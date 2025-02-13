data {
  int <lower = 0> n_meas;                   // number of observations
  vector <lower = 0> [n_meas] obs_time;     // time of observation
  vector <lower = 0> [n_meas] fail_lb;      // lower bound of failure time
  vector <lower = 0> [n_meas] fail_ub;      // status of observation

  array [n_meas] int<lower = 0, upper = 1> fail_status; // if a failure has occured, we have interval-censored data

  int <lower = 0> n_pred;                   // number of predictions
  vector <lower = 0> [n_pred] pred_time;    // time of prediction
}

parameters {
  real <lower = 0> scale; // scale parameter
  real <lower = 0> shape; // shape parameter
}

model{
    //priors
    scale ~ normal(8, 3);
    shape ~ normal(6, 3);

    //likelihood
    for(n in 1:n_meas){
        if(fail_status[n] == 0){
            target += log1m(loglogistic_cdf(obs_time[n] | scale, shape));
        } else {
            target += log(
                          loglogistic_cdf(fail_ub[n] | scale, shape) - 
                          loglogistic_cdf(fail_lb[n] | scale, shape)
                        );
        }
    }
}

generated quantities {
  vector [n_meas] log_lik;
  vector [n_pred] p_fail_pred;

  for(n in 1:n_meas){
    if(fail_status[n] == 1){
      log_lik[n] = log1m(loglogistic_cdf(obs_time[n] | scale, shape));
    } else {
      log_lik[n] = log(
                        loglogistic_cdf(fail_ub[n] | scale, shape) - 
                        loglogistic_cdf(fail_lb[n] | scale, shape)
                      );
    }
  }

  for(n in 1:n_pred){
    p_fail_pred[n] = loglogistic_cdf(pred_time[n] | scale, shape);
  }
  
}

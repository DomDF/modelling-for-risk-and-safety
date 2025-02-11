data {
  int <lower = 0> n_meas;                   // number of observations
  vector <lower = 0> [n_meas] obs_time;     // time of observation
  vector <lower = 0> [n_meas] fail_lb;      // lower bound of failure time
  vector <lower = 0> [n_meas] fail_ub;      // status of observation
}

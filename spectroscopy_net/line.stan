
data {
    int<lower=1> N;
    real y[N];
    real y_err[N];
    real x[N];
    real bg_lower;
    real bg_upper;
}

parameters {
    real<lower=0.5, upper=1> theta;
    real m;
    real<lower=0> c;
    real<lower=0> bg_variance;
}

model {
    for (n in 1:N) {
    target += log_mix(theta,
                normal_lpdf(y[n] | m * x[n] + c, y_err[n]),
                normal_lpdf(y[n] | m * x[n] + c, pow(pow(y_err[n], 2) + bg_variance, 0.5)));
    }
}

module OptimConstants
    export N_mnbrak, golden_ratio, N_para, brent_ib, brent_eps, Cr, N_optim
    N_mnbrak::Int64 = 1e4;                     # number of maximum iterations for mnbrak
    golden_ratio = 0.5 * (1.0 + sqrt(5.0));
    N_para::Int64 = 1e4;                       # number of maximum iterations for linmin
    brent_ib = 1.0e-1;                          # the size of initial step in line search, if nothing specified
    brent_eps = 1e-14;
    Cr = 1. - 1/golden_ratio;
    N_optim::Int64 = 100;                          # number of maximum iterations for conjugate-gradient
end
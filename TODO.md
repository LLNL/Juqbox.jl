1. ~~Rename parameters() -> objparams()~~

2. Replace import objfunc -> using objfunc in all setup files in examples/\*.jl. Add missing functions to the - [ ] list.

3. ~~Rename module objfunc -> Juqbox (?)~~

4. Document all exported functions (add more?):


- [ ] splineparams 
- [ ] bspline2 
- [ ] gradbspline2
- [ ] bcparams
- [ ] bcarrier2
- [ ] gradbcarrier2
- [ ] objparams
- [ ] traceobjgrad
- [ ] identify_guard_levels
- [ ] identify_forbidden_levels
- [ ] plotunitary
- [ ] plotspecified
- [ ] evalctrl
- [ ] setup_ipopt_problem
- [ ] Working_Arrays
- [ ] estimate_Neumann!
- [ ] assign_thresholds
- [ ] setup_rotmatrices
- [ ] run_optimizer
- [ ] plot_conv_hist
- [ ] setup_prior!
- [ ] wmatsetup
- [ ] assign_thresholds_old
- [ ] assign_thresholds_freq 
- [ ] init_adjoint!
- [ ] tracefidabs2
- [ ] tracefidreal,tracefidcomplex
- [ ] trace_operator
- [ ] adjoint_trace_operator!
- [ ] penalf2a
- [ ] penalf2aTrap
- [ ] penalf2adj
- [ ] penalf2adj!
- [ ] penalf2grad
- [ ] tikhonov_pen
- [ ] tikhonov_grad!
- [ ] KS!
- [ ] accumulate_matrix!
- [ ] controlfunc
- [ ] controlfuncgrad!
- [ ] rotmatrices!
- [ ] fgradforce!
- [ ] adjoint_grad_calc!
- [ ] eval_forward
- [ ] estimate_Neumann
- [ ] calculate_timestep
- [ ] KS_alloc
- [ ] time_step_alloc
- [ ] grad_alloc
- [ ] eval_f_par
- [ ] eval_g
- [ ] eval_grad_f_par
- [ ] eval_jac_g
- [ ] intermediate_par
- [ ] plot_forward
- [ ] specify_level3
- [ ] marginalize3
- [ ] adjoint_tableau
- [ ] step
- [ ] step!
- [ ] explicit_step
- [ ] step_fwdGrad!
- [ ] stepseparable
- [ ] getgamma
- [ ] magnus
- [ ] neumann!
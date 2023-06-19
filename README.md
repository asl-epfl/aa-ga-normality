# aa-ga-normality

This repository contains the code pertaining to the simulations in [1]. In this version, the combination matrix and the correlation matrix for the Gaussian case are fixed and equal to the ones in the manuscript. One needs to change them manually if necessary.

The usage is as follows:

```

simulation_gh.py [-h] --file FILE [--N_user N_USER] [--T T] [--paths PATHS] [--theta THETA THETA]
                        [--cov COV] [--normalize {y,n}] [--distribution {g,e}]
                        {s,t}

positional arguments:
  {s,t}                 s: simulation mode, t: performs test from the pickle data

options:
  -h, --help            show this help message and exit
  --file FILE           file name
  --N_user N_USER       Number of users (nodes) in the setting
  --T T                 The horizon (i in the manuscript)
  --paths PATHS         Number of paths for Monte Carlo simulations
  --theta THETA THETA   Parameters of distributions under the null and alternative hypotheses, respectively (in the manuscript equals to 1 2)
  --cov COV             0 for no correlation across nodes, 1 for the correlation matrix in the manuscript
                        (only valid for the Gaussian case)
  --normalize {y,n}     Normalization for plotting
  --distribution {g,e}  The distribution at the nodes (g for Gaussian and e for Exponential)
```

[1] M. Kayaalp, Y. İnan, V. Koivunen, E. Telatar, and A. H. Sayed, “On the Fusion Strategies for Federated Decision Mak- ing,” 2023. [Accepted to IEEE Statistical Signal Processing Workshop, 2023]. Preprint: https://arxiv.org/abs/2303.06109.

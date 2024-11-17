# dice2016
python implementation of DICE2016 using pyomo and ipopt

## installation
1. clone from GitHub
2. install required dependencies (requires `conda`, see [Installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) with `conda env create -f environment.yml`

## run
1. activate environment with `conda activate dice2016`
2. to edit parameters that control the model behaviour, edit the `dice2016_parameters.csv` file (note that some combinations of parameters may cause infeasibilities or weird results)
3. then `python dice2016.py`.

## notes
for my template I used the DICE2016R3-112018-v8 implementation. This was adjusted for 2018 prices. There are several versions of DICE2016 knocking around. Colours may vary if you compare to a different version. However I think all the versions of DICE2016 are structurally the same.

this implementation doesn't *quite* reproduce the GAMS version of Nordhaus (you can do your own comparison with the data at https://github.com/chrisroadmap/loaded-dice/blob/main/dice2016/DICER3-opt.csv), but it gets pretty damn close. I don't know if this is down to using a different solver (ipopt versus CONOPT), differences between GAMS and Pyomo, or me mucking up the translation of the GAMS code. In the third case do raise an issue if you find a clear and obvious error (i.e. one that VAR would correct). Results are sensitive to assumptions and parameter values anyway so I claim that they are close enough to be useful.

## acknowledgements
this code was inspired by the DICE2013 implementation by Optimized Financial Systems available from https://github.com/moptimization/pythondice2013implementation. Without their skeleton I would not know where to start. They produced this model several years ago and it no longer runs out of the box, so I also brought their DICE2013 implementation code up to date to work with python 3 and newer versions of pyomo, at https://github.com/chrisroadmap/pythondice2013implementation. Also thanks to Bill Nordhaus for making the original DICE2016 GAMS code available.

mkdir ../results/plots

python 01-baseline-pairwise.py
python 02-baseline-onevall.py
python 03-debias-pairwise.py
python 04-debias-onevall.py
python 05-debias-pairwise-eval-within-site.py
python 06-neg-control-debias-pairwise.py
python 06.5-neg-control-baseline-pairwise.py
python 07-min-sample-read-counts-debias-pairwise.py
python 08-make-plots.py
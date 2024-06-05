import openturn_sampling as ot
import matplotlib.pyplot as plt 
import numpy as np

import time


distr1 = ot.TruncatedDistribution(ot.Normal(1), -1.5, ot.TruncatedDistribution.UPPER)
distr2 = ot.TruncatedDistribution(ot.Normal(1), 1.5, ot.TruncatedDistribution.LOWER)
failure_distr = ot.Mixture([distr1,distr2])

xx = np.linspace(-7,7,101).reshape((-1,1))
yy = failure_distr.computePDF(xx)
yyze = ot.Normal(1).computePDF(xx)

full = ot.ComposedDistribution([failure_distr] + (20-1)*[ot.Normal(1)])

plt.hist(np.array(full.getSample(1000)[:, 0]).reshape(-1,), bins = 'auto')



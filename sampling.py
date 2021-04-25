import pymc3 as mc
import numpy as np
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
from theano.ifelse import ifelse
from scipy.stats import gaussian_kde
from utils import matrix

class Sampler(object):
	def __init__(self, D):
		self.D = D
		self.Avar = matrix(0, self.D)
		self.yvar = matrix(0, 1)
		x = tt.vector()
		self.f = th.function([x], -tt.sum(tn.relu(tt.dot(-tt.tile(self.yvar,[1,D])*self.Avar, x))))
	@property
	def A(self):
		return self.Avar.get_value()
	@A.setter
	def A(self, value):
		if len(value)==0:
			self.Avar.set_value(np.zeros((0, self.D)))
		else:
			self.Avar.set_value(np.asarray(value))
    
	@property
	def y(self):
		return self.yvar.get_value()
	@y.setter
	def y(self, value):
		if len(value)==0:
			self.yvar.set_value(np.zeros((0, 1)))
		else:
			self.yvar.set_value(np.asarray(value))

	def sample(self, N, T=50, burn=1000):
		with mc.Model() as model:
			x = mc.Uniform(name='x', lower=-np.ones(self.D), upper=np.ones(self.D), shape=self.D, testval=np.zeros(self.D))
			
			def sphere(x):
				if(mc.math.ge(((x**2).sum()),1.)):
					#Can't do -inf because it'll make sampling throw an error of "bad initial energy"
					return np.iinfo(np.int64).min
				else:
					return self.f(x)

			#need to convert sphere to a theano tensor
			p1 = mc.Potential(name="sphere", var=tt.constant(sphere(x)))

			#No adaptive metropolis function in pymc3?
			steps = mc.Metropolis(vars=x, delay=burn, cov=np.eye(self.D)/10000)
		
			#There aren't key word arguments for thin or burn? --> what to do?
			#burn can be used to get the trace values? is that good enough?
			#samples = pm.sample(N*T+burn, thin=T, burn=burn, verbose=-1)
			#need cores definition otherwise i get a broken pipe?
			#...no clude what tune is
			trace = mc.sample(N*T+burn, tune=T, cores=1)

			samples = trace.get_values('x', burn=burn)
			samples = np.array([x/np.linalg.norm(x) for x in samples])
		
			return samples

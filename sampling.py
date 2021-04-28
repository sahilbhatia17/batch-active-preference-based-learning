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
#		self.f = th.function([x], -tt.sum(tn.relu(tt.dot(-tt.tile(self.yvar,[1,D])*self.Avar, x))))

	#cannot use theano function because x is not an array but a tensor variable
	def f(self, x):
		return -tt.sum(tn.relu(tt.dot(-tt.tile(self.yvar,[1,self.D])*self.Avar, x)))
		
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
			x = mc.Uniform(name='x', lower=-np.ones(self.D), upper=np.ones(self.D), shape=self.D, testval=np.zeros(self.D), transform=None)
			
#			def sphere(x):
#				if(mc.math.ge(((x**2).sum()),1.)):
#					return -np.iif
#				else:
#					return self.f(x)
			
			#potential examples: https://docs.pymc.io/notebooks/gaussian_mixture_model.html
			#PYMC3 and Theano summary: https://docs.pymc.io/PyMC3_and_Theano.html
			#sphere code - the previous method is completely different
			test = (x**2).sum()
			p1 = mc.Potential("sphere", tt.switch((test>1.).all(), -np.inf, self.f(x)))

			#mc.Metropolis in PYMC3 implements adaptive Metropolis-Hastings
			steps = mc.Metropolis([x], delay=burn, cov=np.eye(self.D)/10000)
		
			#sampling just requires, #samples, steps, and cores (I can't run more than 1 core otherwise I get a broken pipe issue)
			#tune also seems to be needed with PYMC3 default value is too small
			trace = mc.sample(N*T+burn, tune=(N*T+burn)/2, step=steps, cores=1, return_inferencedata=False)
			
			#we get selected samples from the trace with thin and burn parameters
			samples = trace.get_values('x', thin=T, burn=burn)
			
			#normalize the samples
			samples = np.array([x/np.linalg.norm(x) for x in samples])
		
			return samples

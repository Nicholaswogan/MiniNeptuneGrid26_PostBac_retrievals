"""
Analyzer and Parser for MultiNest output files
===============================================

Use like so::
	
	# create analyzer object
	a = Analyzer(n_params, outputfiles_basename = "chains/1-")
	
	# get a dictionary containing information about
	#   the logZ and its errors
	#   the individual modes and their parameters
	#   quantiles of the parameter posteriors
	stats = a.get_stats()
	
	# get the best fit (highest likelihood) point
	bestfit_params = a.get_best_fit()
	
	# iterate through the "posterior chain"
	for params in a.get_equal_weighted_posterior():
		print params


"""


from __future__ import absolute_import, unicode_literals, print_function
import numpy
from io import StringIO
import re

_FORTRAN_EXP_RE = re.compile(
	r'(?P<mantissa>(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:[DEde][+-]?\d+)?)'
	r'(?P<exp>[+-]\d+)\b'
)

def _fix_fortran_exponents(text):
	def repl(match):
		mantissa = match.group('mantissa')
		exp = match.group('exp')
		if re.search(r'[DEde][+-]?\d+$', mantissa):
			return match.group(0)
		return f'{mantissa}E{exp}'
	return _FORTRAN_EXP_RE.sub(repl, text)

def loadtxt2d(intext):
	def _load_from_string(text):
		try:
			return numpy.loadtxt(StringIO(text), ndmin=2)
		except:
			return numpy.loadtxt(StringIO(text))

	try:
		return numpy.loadtxt(intext, ndmin=2)
	except:
		pass

	if hasattr(intext, 'read'):
		text = intext.read()
		try:
			intext.seek(0)
		except Exception:
			pass
	else:
		with open(intext) as f:
			text = f.read()

	text = _fix_fortran_exponents(text)
	return _load_from_string(text)

class Analyzer(object):
	"""
		Class for accessing the output of MultiNest.
		
		After the run, this class provides a mean of reading the result
		in a structured way.
		
	"""
	def __init__(self, n_params, outputfiles_basename = "chains/1-", verbose=True):
		self.outputfiles_basename = outputfiles_basename
		self.n_params = n_params
		"""
		[root].txt
		Compatable with getdist with 2+nPar columns. Columns have 
		sample probability, -2*loglikehood, samples. Sample
		probability is the sample prior mass multiplied by its 
		likelihood & normalized by the evidence.
		"""
		self.data_file = "%s.txt" % self.outputfiles_basename
		if verbose:
			print(('  analysing data from %s' % self.data_file))

		"""[root]post_separate.dat
		This file is only created if mmodal is set to T. Posterior 
		samples for modes with local log-evidence value
		greater than nullZ, separated by 2 blank lines. Format is the 
		same as [root].txt file."""
		self.post_file = "%spost_separate.dat" % self.outputfiles_basename

		"""[root]stats.dat
		Contains the global log-evidence, its error & local log-evidence 
		with error & parameter means & standard
		deviations as well as the  best fit & MAP parameters of each of 
		the mode found with local log-evidence > nullZ.
		"""
		self.stats_file = "%sstats.dat" % self.outputfiles_basename

		"""
		[root]post_equal_weights.dat
		Contains the equally weighted posterior samples
		"""
		self.equal_weighted_file = "%spost_equal_weights.dat" % self.outputfiles_basename
	
	def get_data(self):
		"""
			fetches self.data_file
		"""
		if not hasattr(self, 'data'):
			self.data = loadtxt2d(self.data_file)
		return self.data
	def get_equal_weighted_posterior(self):
		"""
			fetches self.data_file
		"""
		if not hasattr(self, 'equal_weighted_posterior'):
			self.equal_weighted_posterior = loadtxt2d(self.equal_weighted_file)
		return self.equal_weighted_posterior

	def get_equal_weighted_posterior_from_data(self, n_samples=None, seed=12345,
		include_loglike=False, include_weights=False):
		"""
			Draw an equal-weight posterior sample directly from [root].txt.

			The first column in [root].txt is treated as the posterior weight,
			the second column is -2*loglike, and the remaining columns are the
			physical parameters.

			@param n_samples:
				Number of equal-weight draws to return. Defaults to the MultiNest
				estimate npost = nint(exp(-sum(w log w))), matching posterior.F90.
			@param seed:
				Random seed used for the stochastic rounding step.
			@param include_loglike:
				If True, include the -2*loglike column in the returned samples.
			@param include_weights:
				If True, include the original posterior weight column in the
				returned samples.
		"""
		data = self.get_data()
		if data.ndim != 2 or data.shape[1] < 3:
			raise ValueError("MultiNest data must be a 2D array with at least 3 columns")

		weights = numpy.asarray(data[:, 0], dtype=float)
		weights = numpy.clip(weights, 0.0, numpy.inf)
		weight_sum = float(weights.sum())
		if not numpy.isfinite(weight_sum) or weight_sum <= 0.0:
			raise ValueError("Cannot resample posterior: weights are non-positive or non-finite")
		weights = weights / weight_sum

		if n_samples is None:
			mask = weights > 0.0
			entropy = -numpy.sum(weights[mask] * numpy.log(weights[mask]))
			n_samples = int(numpy.rint(numpy.exp(entropy)))
		n_samples = int(n_samples)
		if n_samples <= 0:
			raise ValueError("n_samples must be positive")

		rng = numpy.random.default_rng(seed)
		expected = weights * n_samples
		multiplicity = numpy.floor(expected).astype(int)
		remainder = expected - multiplicity
		multiplicity += (rng.random(data.shape[0]) <= remainder).astype(int)
		indices = numpy.repeat(numpy.arange(data.shape[0]), multiplicity)
		if indices.size == 0:
			raise ValueError("Resampling produced no posterior draws")
		if indices.size != n_samples:
			# Match MultiNest's rounded-multiplicity behavior as closely as possible,
			# but allow the stochastic rounding to shift the final count slightly.
			n_samples = indices.size

		columns = []
		if include_weights:
			columns.append(data[indices, 0:1])
		if include_loglike:
			columns.append(data[indices, 1:2])
		columns.append(data[indices, 2:])

		return numpy.hstack(columns)
	
	def _read_error_line(self, l):
		#print('_read_error_line -> line>', l)
		name, values = l.split('   ', 1)
		#print('_read_error_line -> name>', name)
		#print('_read_error_line -> values>', values)
		name = name.strip(': ').strip()
		values = values.strip(': ').strip()
		v, error = values.split(" +/- ")
		return name, float(v), float(error)
	def _read_error_into_dict(self, l, d):
		name, v, error = self._read_error_line(l)
		d[name.lower()] = v
		d['%s error' % name.lower()] = error
	def _read_table(self, txt, d = None, title = None):
		if title is None:
			title, table = txt.split("\n", 1)
		else:
			table = txt
		header, table = table.split("\n", 1)
		data = loadtxt2d(StringIO(table))
		if d is not None:
			d[title.strip().lower()] = data
		if len(data.shape) == 1:
			data = numpy.reshape(data, (1,-1))
		return data
	
	def get_stats(self):
		posterior = self.get_data()
		
		stats = []
		
		for i in range(2, posterior.shape[1]):
			b = list(zip(posterior[:,0], posterior[:,i]))
			b.sort(key=lambda x: x[1])
			b = numpy.array(b)
			b[:,0] = b[:,0].cumsum()
			sig5 = 0.5 + 0.9999994 / 2.
			sig3 = 0.5 + 0.9973 / 2.
			sig2 = 0.5 + 0.95 / 2.
			sig1 = 0.5 + 0.6826 / 2.
			bi = lambda x: numpy.interp(x, b[:,0], b[:,1], left=b[0,1], right=b[-1,1])
			
			low1 = bi(1 - sig1)
			high1 = bi(sig1)
			low2 = bi(1 - sig2)
			high2 = bi(sig2)
			low3 = bi(1 - sig3)
			high3 = bi(sig3)
			low5 = bi(1 - sig5)
			high5 = bi(sig5)
			median = bi(0.5)
			q1 = bi(0.75)
			q3 = bi(0.25)
			q99 = bi(0.99)
			q01 = bi(0.01)
			q90 = bi(0.9)
			q10 = bi(0.1)
			
			stats.append({
				'median': median,
				'sigma': (high1 - low1) / 2.,
				'1sigma': [low1, high1],
				'2sigma': [low2, high2],
				'3sigma': [low3, high3],
				'5sigma': [low5, high5],
				'q75%': q1,
				'q25%': q3,
				'q99%': q99,
				'q01%': q01,
				'q90%': q90,
				'q10%': q10,
			})
		
		mode_stats = self.get_mode_stats()
		mode_stats['marginals'] = stats 
		return mode_stats
	
	def get_mode_stats(self):
		"""
			information about the modes found:
			mean, sigma, maximum a posterior in each dimension
		"""
		with open(self.stats_file) as file:
			lines = file.readlines()
		# Global Evidence
		stats = {'modes':[]}
		self._read_error_into_dict(lines[0], stats)

		if 'Nested Sampling Global Log-Evidence'.lower() in stats:
			stats['global evidence'] = stats['Nested Sampling Global Log-Evidence'.lower()]
			stats['global evidence error'] = stats['Nested Sampling Global Log-Evidence error'.lower()]
		
		if 'Nested Importance Sampling Global Log-Evidence' in lines[1]:
			# INS global evidence
			self._read_error_into_dict(lines[1], stats)
			Z = stats['Nested Importance Sampling Global Log-Evidence'.lower()]
			Zerr = stats['Nested Importance Sampling Global Log-Evidence error'.lower()]
			# use INS results in default name
			stats['global evidence'] = Z
			stats['global evidence error'] = Zerr
			
			text = ''.join(lines[3:])
			
			i = 0
			modelines = text.split("\n\n")
			mode = {
				'index':i
			}
			# no preamble; construct it
			# Strictly local evidence
			mode['Strictly Local Log-Evidence'.lower()] = Z
			mode['Strictly Local Log-Evidence error'.lower()] = Zerr
			mode['Local Log-Evidence'.lower()] = Z
			mode['Local Log-Evidence error'.lower()] = Zerr
			t = self._read_table(modelines[0], title = "Parameters")
			mode['mean'] = t[:,1].tolist()
			mode['sigma'] = t[:,2].tolist()
			mode['maximum'] = self._read_table(modelines[1])[:,1].tolist()
			mode['maximum a posterior'] = self._read_table(modelines[2])[:,1].tolist()
			stats['modes'].append(mode)
		else:
			text = ''.join(lines)
			# without INS:
			parts = text.split("\n\n\n")
			del parts[0]
		
			i = 0
			for p in parts:
				modelines = p.split("\n\n")
				mode = {
					'index':i
				}
				i = i + 1
				modelines1 = modelines[0].split("\n")
				# Strictly local evidence
				self._read_error_into_dict(modelines1[1], mode)
				self._read_error_into_dict(modelines1[2], mode)
				t = self._read_table(modelines[1], title = "Parameters")
				mode['mean'] = t[:,1].tolist()
				mode['sigma'] = t[:,2].tolist()
				mode['maximum'] = self._read_table(modelines[2])[:,1].tolist()
				mode['maximum a posterior'] = self._read_table(modelines[3])[:,1].tolist()
				stats['modes'].append(mode)
		return stats

	def get_best_fit(self):
		data = self.get_data()
		i = (-0.5 * data[:,1]).argmax()
		lastrow = data[i]
		return {'log_likelihood': float(-0.5 * lastrow[1]),
			'parameters': list(lastrow[2:])}

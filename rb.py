# Filename: rb.py
# 
# Randomized benchmarking library
#
# usage: python rb.py flag args*
# for more info, use python rb.py -help
#
# Programmer: Peter Karalekas (2015)

import sys
import lmfit
import random
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import qutipsim.simulator.qutipsim as qutipsim

from qutip import *
from pulseseq.pulselib import *
from pulseseq.sequencer import *
from qrlab.lib.math import fitter

class QubitSim(object):
	'''
	Qubit simulation class (times in ns, frequencies in Hz)

	Properties:
	- qdrive: maximum qubit drive amplitude (Hz)
	- qanharm: qubit anharmonicity amplitude (Hz)
	- resample_factor: how many samples to generate per waveform point
	- filter_bw: bandwidth of filter to apply to waveforms (Hz)
	- decay_rate: rate of decay for collapse operators (Hz)
	- qsim: allows for a pre-created qubit simulation object to be used
	'''

	def __init__(self, 
		qdrive=2*np.pi*40.2e6,
		qanharm=-2*np.pi*250e6,
		resample_factor=4,
		filter_bw=250e6,
		decay_rate=0,
		qsim=None):
		'''
		Default constructor
		'''

		# if no simulator object is passed as an argument
		if qsim == None:
			
			# initialize simulation object
			qsim = qutipsim.Simulator()
			qsim.resample_factor = resample_factor
			qsim.filter_bw = filter_bw

			# add two-level system to the Hilbert space
			qsim.add_dimension('qubit', 2)

			# define destroy operators
			qa = qsim.get_destroy('qubit')
			qs = qsim.get_qdestroy('qubit')
			
			# define Hamiltonian
			qsim.H0 = 0.5 * qanharm * qa.dag() * qa.dag() * qa * qa
			qsim.add_time_term(qdrive/np.sqrt(2)*(qa + qa.dag()), 'qI')
			qsim.add_time_term(qdrive/np.sqrt(2)*1j*(qa - qa.dag()), 'qQ')

			# add collapse operators (T1)
			qsim.add_collapse(np.sqrt(decay_rate) * qa)
			
			# add expectation values to pre-calculate
			qsim.add_expect(1 * (qs + qs.dag()), 'sx')
			qsim.add_expect(1j * (qs - qs.dag()), 'sy')
			qsim.add_expect(qs.dag() * qs, 'sz')

		self.qsim = qsim

class RBSim(object):
	'''
	Randomized Benchmarking simulation class
	'''

	# default pulse definition
	r_def = AmplitudeRotation(base=Gaussian, 
		w=4, 
		pi_amp=1, 
		chans=('qI', 'qQ'), 
		drag=0, 
		chop=6)

	def __init__(self):
		'''
		Default constructor
		'''

		# clifford group matrices
		self.matrices = {"cxp" : (identity(2)+1j*sigmax())/sqrt(2),
						"cxm" : (identity(2)-1j*sigmax())/sqrt(2),
						"cyp" : (identity(2)+1j*sigmay())/sqrt(2),
						"cym" : (identity(2)-1j*sigmay())/sqrt(2),
						"px" : 1j*sigmax(),
						"py" : 1j*sigmay(),
						"p1" : identity(2)}

	def build_train(self, tlen, pulse=None, n=None):
		'''
		Builds the random pulse train

		Properties:
		- tlen: train length (determined by number of random pulses)
		- pulse: pulse to interleave (optional)
		- n: number of times to repeat the interleaved pulse (optional)
		'''

		train = []

		# interleaved train
		if n != None and pulse != None:
			for i in range(0, tlen):
				for j in range(0, n): train.append(pulse)
				train.append(random.choice(self.matrices.keys()))

		# interleaved
		elif pulse != None:
			for i in range(0, tlen):
				train.append(pulse)
				train.append(random.choice(self.matrices.keys()))

		# standard
		else:
			for i in range(0, tlen):
				train.append(random.choice(self.matrices.keys()))

		return train

	def sim_ideal(self, train, psi0=basis(2,0)):
		'''
		Finds the ideal final state after pulse train is applied

		Properties:
		- train: array of pulse names
		- psi0: initial state vector (optional)
		'''

		# initial state
		psis = []
		psis.append(psi0)

		# apply matrix / pulse train
		for i in range(0, len(train)):
			psis.append(self.matrices[train[i]]*psi0)
			psi0 = psis[i+1]

		# first element is psi0, last element is psiF
		return psis

	def final_pulse(self, train):
		'''
		Calculates the final pulse to return to an eigenstate of the z-basis

		Properties:
		- train: array of pulse names
		'''

		# initialize variables
		full = identity(2)
		final = None

		# calculate fully composed train matrix
		for i in range(0, len(train)): full = self.matrices[train[i]]*full

		def check_matrix(mat):
			'''
			Determines if a matrix is diagonalized or anti-diagonalized

			Properties:
			- mat: matrix in question
			'''

			# anti-diagonalized
			if mat[0,0] == 0 and mat[1,1] == 0: return True

			# diagonalized
			elif mat[0,1] == 0 and mat[1,0] == 0: return True

			return False

		# randomly shuffle the list of matricies
		idcs = range(len(self.matrices))
		random.shuffle(idcs)
		idcs = np.array(idcs)
		ordered = np.array(self.matrices.values())
		ordered = ordered[idcs]
		keys = np.array(self.matrices.keys())
		keys = keys[idcs]

		# determine final pulse and flip
		# 0 = no flip, 1 = flip
		for idx, matrix in enumerate(ordered):

			trial = matrix*full

			if check_matrix(trial):
				if trial[0,0] == 0: flip = True
				else: flip = False

				return full, keys[idx], flip

		raise RuntimeError('final_pulse failed to find a pulse!')

	def build_sequence(self, train, r):
		'''
		Builds the pulse sequence from a train and pulse definition

		Properties:
		- train: array of pulse names
		- r: pulse definition object
		'''

		# clifford group pulses
		pulses = 	{"cxp" : r(np.pi/2, 0),
					"cxm" : r(-np.pi/2, 0),
					"cyp" : r(np.pi/2, np.pi/2),
					"cym" : r(-np.pi/2, np.pi/2),
					"px" : r(np.pi, 0),
					"py" : r(np.pi, np.pi/2),
					"p1" : r(0, 0)}

		# create sequence
		seq = Sequence()
		for i in range(0, len(train)): seq.append(pulses[train[i]])
	
		# run sequencer
		sqr = Sequencer(seq, minlen=1)
		sqr.add_required_channel('qI')
		sqr.add_required_channel('qQ')

		return seq, sqr

	def plot_sequence(self, sqr):
		'''
		Plot pulse sequence

		Properties:
		- sqr: sequencer object
		'''

		seqs = sqr.render()
		sqr.plot_seqs(seqs)

	def sim_actual(self, sqr, qbase):
		'''
		Returns the qubit after applying the pulse train

		Properties:
		- sqr: sequencer object
		- qbase: qubit simulation object
		'''

		# simulate system
		seqs = sqr.render()
		psi0 = qbase.qsim.get_ground()
		qbase.qsim.simulate(seqs, psi0)

		return qbase

	def plot_bloch(self, qubit):
		'''
		Plots the evolution of the qubit on the bloch sphere

		Properties:
		- qubit: qubit simulation object
		'''

		b = qutip.Bloch()
		b.add_points([qubit.qsim.get_exp('sx'), qubit.qsim.get_exp('sy'), -2*qubit.qsim.get_exp('sz')+1], meth='l')
		b.make_sphere()
		plt.legend()
		b.show()

	def fidelity(self, psis, qubit, train):
		'''
		Determines the fidelity over the course of a pulse train

		Properties:
		- psis: array of state vectors
		- qubit: qubit simulation object
		- train: array of pulse names
		'''

		# initialize array
		fid = []
		fid.append(abs(psis[0].dag()*qubit.qsim.psi_out[0]).data**2)

		# fill in remaining fidelities
		# the 96 corresponds to resample_factor (4) * chop (6) * 2nd argument in rotation (4)
		for i in range(0, len(train)):
			temp = abs(psis[i+1].dag()*qubit.qsim.psi_out[96*(i+1)].conj()*psis[i+1]).data
			fid.append(temp)

		# the final element is the fidelity of the final states
		return fid

	def build_fit(self, avg, train):
		'''
		Fits a zeroth-order exponential to the data

		Properties:
		- avg: average fidelity values
		- train: array of pulse names
		'''

		# exponential fit
		def f(x,b):
			return 10**(-b * x)

		popt, pcov = opt.curve_fit(f,range(0, len(train)+1),2*np.array(avg)[:,0]-1)
		fit = f(np.arange(0, len(train)+1),popt[0])
		return popt, pcov, fit

	def plot_avgfid(self, avg, train, sds=None, scale=None, fit=None):
		'''
		Plot average fidelity line

		Properties:
		- avg: average fidelity values
		- train: array of pulse names
		- sds: standard deviation values (optional)
		- scale: scale for plotting values (ex. 'log', optional)
		- fit: best fit values (optional)
		'''
		
		# plot log of average fidelity line with error bars and fit
		if sds and scale and fit != None:
			plt.figure()
			plt.errorbar(range(0, len(train)+1), 2*np.array(avg)[:,0]-1, yerr=sds, label='Avg')
			plt.plot(range(0, len(train)+1), fit, label='Fit')
			plt.suptitle('Average Fidelity vs. Pulse Train Length Log (with Fit)')
			plt.xlabel('Pulse Train Length')
			plt.ylabel('2*(Average Fidelity)-1')
			plt.yscale(scale, nonposy='clip')
			plt.ylim(0,1)
			plt.legend(loc=3)

		# plot log of average fidelity line with error bars
		elif sds and scale:
			plt.figure()
			plt.errorbar(range(0, len(train)+1), np.array(avg)[:,0], yerr=sds, label='Avg')
			plt.suptitle('Average Fidelity vs. Pulse Train Length Log')
			plt.yscale(scale)
			plt.ylim(0.5,1.1)
			plt.legend(loc=3)

		# plot average fidelity line with error bars
		elif sds:
			plt.figure()
			plt.errorbar(range(0, len(train)+1), np.array(avg)[:,0], yerr=sds, label='Avg')
			plt.suptitle('Average Fidelity vs. Pulse Train Length')
			plt.legend(loc=3)

		else:
			plt.plot(range(0, len(train)+1), avg, label='Avg', color='black', lw=2, ls = '--')
			plt.suptitle('Fidelity vs. Pulse Train Length')
			plt.legend(loc=3)

	def prep_experiment(self, tlen, pulse=None, n=None, r=r_def):
		'''
		Prepare trains for physical qubit experiment or virtual qubit simulation

		Properties:
		- tlen: train length (determined by number of random pulses)
		- pulse: pulse to interleave (optional)
		- n: number of times to repeat the interleaved pulse (optional)
		- r: pulse definition (optional)
		'''
		
		# build random train
		train = self.build_train(tlen, pulse, n)

		# determine final correction pulse
		full, final, flip = self.final_pulse(train)
		train.append(final)

		# create pulse sequence
		seq, sqr = self.build_sequence(train, r)

		return train, seq, flip

	def sim_experiment(self, train, qbase=QubitSim(decay_rate=2*np.pi*1e5), r=r_def):
		'''
		Perform virtual qubit randomized benchmarking simulation

		Properties:
		- train: array of pulse names
		- qbase: qubit simulation object (optional)
		- r: pulse definition object (optional)
		'''

		# create pulse sequence
		seq, sqr = self.build_sequence(train, r)

		# perform simulation
		qubit = self.sim_actual(sqr, qbase)

		return qubit.qsim.psi_out[-1]

	def analyze_fit(self, xdata, pdata):
		'''
		Given data, perform an exponential decay fit

		Properties:
		- xdata: train length values
		- pdata: excited state probability values
		'''

		# determine variables
		train_lengths = []
		for i in xdata:
			if i not in train_lengths: train_lengths.append(i)
		trains_per_length = len(xdata)/len(train_lengths)

		# plot data
		plt.figure()
		plt.plot(xdata, pdata, 'ro')

		# plot mean and std
		ndata = np.array(pdata).reshape((-1, trains_per_length))
		nmean = np.mean(ndata, axis=1)
		nstd = np.std(ndata, axis=1)
		plt.errorbar(train_lengths, nmean, fmt='ko', yerr=nstd)

		# build fitter
		f = fitter.Fitter('exp_decay')
		p = f.get_lmfit_parameters(np.array(xdata), np.array(pdata))
		p['ofs'].value = 0.5
		p['ofs'].vary = False
		result = f.perform_lmfit(np.array(xdata), np.array(pdata), p=p)

		# extract error
		tau1 = result.params['tau'].value
		tau1_err = result.params['tau'].stderr

		# evaluate fit
		xs_fit = np.linspace(min(train_lengths), max(train_lengths), 101)
		ys_fit = f.eval_func(xs_fit)

		# make label
		label = 'Fit error = %0.5f $\pm$ %0.5f\n' % (1.0/tau1, tau1_err/tau1**2) \
		+ 'Gate fidelity = %0.3f%% $\pm$ %0.3f%%' % ((1.0 - 1.0/tau1)*100.0, (tau1_err/tau1**2)*100.0)
		print '\n' + label

		# plot fit
		plt.plot(xs_fit, ys_fit, 'k-', label=label)
		plt.legend(loc=3)
		plt.suptitle('Zeroth-Order Sequence Fidelity Fit')
		plt.xlabel('Train Length')
		plt.ylabel('Excited State Probability')
		plt.show()

	def analyze_dist(self, xdata, pdata):
		'''
		Given data, determine the distribution of error

		Properties:
		- xdata: train length values
		- pdata: excited state probability values
		'''

		# determine variables
		train_lengths = []
		for i in xdata:
			if i not in train_lengths: train_lengths.append(i)
		trains_per_length = len(xdata)/len(train_lengths)

		# break up data
		ndata = np.array(pdata).reshape((-1, trains_per_length))
		nmean = np.mean(ndata, axis=1)
		nstd = np.std(ndata, axis=1)

		# iterate through lengths
		for i in range(len(train_lengths)):

			# perform normal test
			(chi, p) = stats.normaltest(ndata[i])

			# plot hist and determine parameters
			plt.figure()
			n, bins, patches = plt.hist(ndata[i], 50, normed=1)
			mu = nmean[i]
			sigma = nstd[i]

			# make label
			label = 'Gaussian with $\mu$ = %0.5f and $\sigma$ = %0.5f\n' % (mu, sigma) \
			+ 'Normaltest gives $\chi^2$ = %0.5f and $p$-value = %0.5f' % (chi, p)
			
			# plot Gaussian
			plt.plot(bins, mlab.normpdf(bins, mu, sigma), label=label)
			plt.legend(loc='best')
			title = 'Distribution of Error for Train Length %d' % train_lengths[i]
			plt.suptitle(title)
			plt.xlabel('Excited State Probability')
			plt.ylabel('Number of Sequences')

			# print results
			print '\n' + title + ':\n' + label
		
		plt.show()

	def simulate(self, runs, tlen, pulse=None, n=None, r=r_def, qbase=QubitSim(decay_rate=2*np.pi*1e5)):
		'''
		Perform ideal two-level system vs. virtual qubit fidelity simulation

		Properties:
		- runs: number of simulations to perform
		- tlen: train length (determined by number of random pulses)
		- pulse: pulse to interleave (optional)
		- n: number of times to repeat the interleaved pulse (optional)
		- r: pulse definition object (optional)
		- qbase: qubit simulation object (optional)
		'''

		# initialze variables
		i = 0
		fids = []
		sqr = None
		qubit = None

		# runs loop
		while i < runs:

			# perform simulation
			train = self.build_train(tlen, pulse, n)
			psis = self.sim_ideal(train)
			seq, sqr = self.build_sequence(train, r)
			qubit = self.sim_actual(sqr, qbase)
			fid = self.fidelity(psis, qubit, train)

			# plot of changing fidelity
			if i < 5: plt.plot(range(len(train)+1), fid, label=i)

			# update total arrays
			fids.append(fid)

			i += 1

		# arrays for plotting
		avg = []
		sds = []

		# determine average fidelities
		for j in range(0, len(train)+1):
			s = 0
			for k in range(0, runs): s += fids[k][j]
			avg.append(s/runs)
			sds.append(np.std(np.array(fids)[:,j]))

		# curve fitting
		popt, pcov, fit = self.build_fit(avg, train)

		# average fidelity plotting
		self.plot_avgfid(avg, train)
		self.plot_avgfid(avg, train, sds)
		self.plot_avgfid(avg, train, sds, 'log')
		self.plot_avgfid(avg, train, sds, 'log', fit)

		# plot sequence
		self.plot_sequence(sqr)

		# plot bloch sphere
		self.plot_bloch(qubit)

		# return the evolution of the avg fidelity, last element final state
		return avg, sds, popt, pcov

def print_info(nums):
	'''
	Print helpful information for the user

	Properties:
	- nums: specifies which messages to print
	'''

	# base string to print
	help = ''

	if 0 in nums:
		help += '\nusage: python rb.py flag args*\n\n'
	if 1 in nums:
		help += 'Prepare physical qubit experiment or vitual qubit simulation:\n' \
		+ 'python rb.py -p tlen runs [pulse] [-o output_file]\n' \
		+ 'tlen = length of the randomized pulse train\n' \
		+ 'runs = number of trains to prepare\n' \
		+ 'pulse = pulse to interleave (ex. \"cxp\" = pi/2 +x pulse)\n' \
		+ 'output_file = file to print train info to\n\n'
	if 2 in nums:
		help += 'Perform virtual qubit randomized benchmarking simulation:\n' \
		+ 'python rb.py -e input_file [xdata_file pdata_file]\n' \
		+ 'input_file = file to get train info from\n' \
		+ 'xdata_file = output file for train length values\n' \
		+ 'pdata_file = output file for excited state probability values\n\n'
	if 3 in nums:
		help += 'Analyze randomized benchmarking experimental data:\n' \
		+ 'python rb.py flag xdata_file pdata_file\n' \
		+ 'flag = -d (distribution) or -f (fitting)\n' \
		+ 'xdata_file = text file for train length values\n' \
		+ 'pdata_file = text file for excited state probability values\n\n'
	if 4 in nums:
		help += 'Perform ideal two-level system vs. virtual qubit fidelity simulation:\n' \
		+ 'python rb.py -s runs tlen [pulse] [n]\n' \
		+ 'runs = number of simulations to run\n' \
		+ 'tlen = length of the randomized pulse train\n' \
		+ 'pulse = pulse to interleave (ex. \"cxp\" = pi/2 +x pulse)\n' \
		+ 'n = length of pulse train to interleave\n\n'
	if 5 in nums:
		help += 'for pulse library, use python rb.py -pulses\n'
	if 6 in nums:
		help += 'for more info, use python rb.py -help\n'
	if 7 in nums:
		help += '\nPulse library:\n' \
		+ 'cxp = +x pi/2\n' \
		+ 'cxm = -x pi/2\n' \
		+ 'cyp = +y pi/2\n' \
		+ 'cym = -y pi/2\n' \
		+ 'px = x pi\n' \
		+ 'py = y pi\n' \
		+ 'p1 = wait\n'

	print help
	sys.exit()

if __name__ == '__main__':
	'''
	Executed if this script is run
	'''

	# too few arguments
	if len(sys.argv) == 1: print_info([0,6])

	# pulses names
	names = {"cxp" : '+x pi/2',
			"cxm" : '-x pi/2',
			"cyp" : '+y pi/2',
			"cym" : '-y pi/2',
			"px" : 'x pi',
			"py" : 'y pi',
			"p1" : 'wait'}

	# print full helper info
	if sys.argv[1] == '-help': print_info([0,1,2,3,4,5])

	# print pulse info
	elif sys.argv[1] == '-pulses': print_info([7])

	elif sys.argv[1] == '-p':
		'''
		Prepare physical qubit experiment or vitual qubit simulation
		'''

		# -p helper info
		if len(sys.argv) < 3 or len(sys.argv) > 7: print_info([0,1,6])

		# read command line arguments
		tlen = int(sys.argv[2])
		runs = int(sys.argv[3])
		pulse = None
		prep = None
		
		# read in pulse
		if len(sys.argv) == 5:
			pulse = sys.argv[4]
			if pulse not in names: raise RuntimeError('invalid pulse')

		# output file
		elif len(sys.argv) == 6 and sys.argv[4] == '-o':
			prep = open(sys.argv[6], 'a')

		# pulse and output file
		elif len(sys.argv) == 7 and sys.argv[5] == '-o':
			pulse = sys.argv[4]
			if pulse not in names: raise RuntimeError('invalid pulse')
			prep = open(sys.argv[6], 'a')

		# write runs to file
		if prep != None: prep.write(str(runs) + '\n')

		# experiment preparation loop
		for i in range(runs):

			rb = RBSim()
			train, seq, flip = rb.prep_experiment(tlen, pulse)
	
			# write train and flip values to file
			if prep != None:
				prep.write(str(tlen) + '\n')
				prep.write(str(len(train)) + '\n')
				for t in train: prep.write(t + '\n')
				prep.write(str(int(flip)) + '\n')

			# or print them if no file is given
			else:
				print len(train)
				for t in train: print t
				print 'flip = ' + str(int(flip))

		# close files
		if prep != None: prep.close()

		sys.exit()

	elif sys.argv[1] == '-e':
		'''
		Perform virtual qubit randomized benchmarking simulation
		'''

		# -e helper info
		if len(sys.argv) != 3 and len(sys.argv) != 5: print_info([0,2,6])

		# input file
		inpt = open(sys.argv[2])
		xout = None
		pout = None

		# output files
		if len(sys.argv) == 5:
			xout = open(sys.argv[3], 'a')
			pout = open(sys.argv[4], 'a')

		# get runs and loop
		runs = int(inpt.readline())
		for i in range(runs):
			
			# get train stats
			tlen = int(inpt.readline())
			total = int(inpt.readline())
			
			# build train
			train = []
			for j in range(total): train.append(inpt.readline().strip('\n'))

			# get flip
			flip = int(inpt.readline().strip('\n'))

			# simulate experiment
			rb = RBSim()
			ans = rb.sim_experiment(train)

			# print or write the output
			if xout == None:
				print "train length = " + str(tlen)
				print "excited state probability = " + str(ans[flip, flip].real) + '\n'
			else:
				xout.write(str(tlen) + '\n')
				pout.write(str(ans[flip, flip].real) + '\n')

		# close files
		if xout != None:
			xout.close()
			pout.close()

	elif sys.argv[1] == '-d' or sys.argv[1] == '-f':
		'''
		Analyze randomized benchmarking experimental data
		-d = distribution of error
		-f = exponential decay fit
		'''

		# -d / -f helper info
		if len(sys.argv) != 4: print_info([0,3,6])

		# read command line arguments
		xfile = open(sys.argv[2])
		pfile = open(sys.argv[3])

		# build array of tlen values
		xdata = []
		for line in xfile: xdata.append(float(line.strip()))

		# build array of excited state probability values
		pdata = []
		for line in pfile: pdata.append(float(line.strip()))

		# close files
		xfile.close()
		pfile.close()

		rb = RBSim()
		if sys.argv[1] == '-f': rb.analyze_fit(xdata, pdata)
		elif sys.argv[1] == '-d': rb.analyze_dist(xdata, pdata)
		
		sys.exit()

	elif sys.argv[1] == '-s':
		'''
		Perform ideal two-level system vs. virtual qubit fidelity simulation
		'''

		# -s helper info
		if len(sys.argv) < 4 or len(sys.argv) > 6: print_info([0,4,6])

		# read command line arguments
		runs = int(sys.argv[2])
		tlen = int(sys.argv[3])

		if len(sys.argv) == 4:
			'''
			Standard randomized benchmarking
			'''

			print '\nThis is an example of the functioning of this randomized benchmarking library.\n' \
			+ 'For this simulation, we will perform %d randomized benchmarking runs.\n' % runs \
			+ 'Each run in the simulation will have a train length of %d random pulses.\n' % tlen \
			+ 'It will take a few moments to complete, and will produce six figures.\n' \
			+ 'The program will only exit once all six figures have been closed.\n'

			# build and simulate an example of randomized benchmarking
			rb = RBSim()
			avg, sds, popt, pcov = rb.simulate(runs, tlen)

			print '\nFor this simulation, the final state average fidelity was %f +/- %f.\n' % (avg[-1], sds[-1]) \
			+ 'In addition, the average error per gate was %f +/- %f\n' % (popt[0], np.sqrt(pcov[0]))

		elif len(sys.argv) == 5:
			'''
			Interleaved benchmarking / Decoherence benchmarking
			'''

			# set pulse
			pulse = sys.argv[4]
			if pulse not in names: raise RuntimeError('invalid pulse')

			print '\nThis is an example of the functioning of this randomized benchmarking library.\n' \
			+ 'For this simulation, we will perform %d interleaved benchmarking runs.\n' % runs \
			+ 'Each run will interleave the %s pulse between random pulses.\n' % names[pulse] \
			+ 'This results in a total train length of %d.\n' % (2*tlen) \
			+ 'It will take a few moments to complete, and will produce six figures.\n' \
			+ 'The program will only exit once all six figures have been closed.\n'

			# build and simulate an example of interleaved benchmarking
			rb = RBSim()
			avg, sds, popt, pcov = rb.simulate(runs, tlen, pulse)

			print '\nFor this simulation, the final state average fidelity was %f +/- %f.\n' % (avg[-1], sds[-1]) \
			+ 'In addition, the average error per gate was %f +/- %f\n' % (popt[0], np.sqrt(pcov[0]))

		elif len(sys.argv) == 6:
			'''
			Interleaved train benchmarking
			'''

			# set pulse and interleaved train length
			pulse = sys.argv[4]
			if pulse not in names: raise RuntimeError('invalid pulse')
			n = int(sys.argv[5])

			print '\nThis is an example of the functioning of this randomized benchmarking library.\n' \
			+ 'For this simulation, we will perform %d interleaved train benchmarking runs.\n' % runs \
			+ 'Each run will interleave a train of %d %s pulses between random pulses.\n' % (n, names[pulse]) \
			+ 'This results in a total train length of %d.\n' % ((n+1)*tlen) \
			+ 'It will take a few moments to complete, and will produce six figures.\n' \
			+ 'The program will only exit once all six figures have been closed.\n'

			# build and simulate an example of interleaved train benchmarking
			rb = RBSim()
			avg, sds, popt, pcov = rb.simulate(runs, tlen, pulse, n)

			print '\nFor this simulation, the final state average fidelity was %f +/- %f.\n' % (avg[-1], sds[-1]) \
			+ 'In addition, the average error per gate was %f +/- %f\n' % (popt[0], np.sqrt(pcov[0]))

	else: print_info([0,6])
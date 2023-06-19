import random
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
from scipy.special import logsumexp
import pickle
import seaborn as sns
import sys
import argparse


if __name__ == '__main__':


	def simulate(T = 8000, N_user = 10, paths = 600, lambda0 = 1, lambda1 = 2, outfile_name = 'final_vals.pkl', transform = None, mode = 'Gaussian'):

		# this is the combination vector in the paper.

		combining_vector = np.array([0.13,0.2,0.09,0.15,0.08,0.05,0.1,0.05,0.1,0.05])

		#if N_user > 10, we will append equal values.

		if N_user > len(combining_vector):
			combining_vector = np.append(combining_vector*len(combining_vector)/N_user, np.array([1/N_user]*(N_user - len(combining_vector))))

		# if N_user < 10, truncates and normalizes the existing vector
		elif N_user < len(combining_vector):
			combining_vector = combining_vector[:N_user]
			combining_vector /= np.sum(combining_vector)


		logcombining_vector = np.expand_dims(np.log(combining_vector), axis=1)

		#initial variables for estimating the mean and variance for AA
		mean_est = 0
		nu_est = 0

		#to store the log-belief ratios at the end of horizon
		final_vals_aa = []
		final_vals_ga = []

		#simulate p paths

		for p in range(paths):

			if p % 20 ==0:
				print('path', p+1)


			#initial beliefs
			logbelief_aa = np.log(0.5)
			logbelief_ga = 0


			for t in range(T):

				#loglikelihood ratios from new data
				logratio, anti_logratio = generate_logratio(lambda0, lambda1, N_user, transform, mode = mode)

				#intermediate beliefs are formed
				logintermediate_beliefs_aa = logbelief_aa + logratio - np.log((1-np.exp(logbelief_aa))+np.exp(logbelief_aa)*np.exp(logratio))


				#beliefs are combined with AA
				logbelief_aa = logsumexp(logcombining_vector + logintermediate_beliefs_aa)


				#beliefs are combined with GA
				logbelief_ga += combining_vector @ logratio


				#mean and variance estimations for AA, antithetic sampling used for performance improvement
				mean_est += (logsumexp(logcombining_vector + logratio)+logsumexp(logcombining_vector + anti_logratio))/2
				nu_est += (logsumexp(logcombining_vector + logratio)**2 + logsumexp(logcombining_vector + anti_logratio)**2)/2

			
			#stores the log-belief ratio values at the end of the horizon
			final_vals_aa.append((-logbelief_aa)+np.log(1-np.exp(logbelief_aa)))
			final_vals_ga.append((-logbelief_ga[0]))


		#unbiased estimates of first and second moments for AA
		mean_est /= T*paths
		nu_est /= T*paths

			
		#write to pickle file
		with open(outfile_name, 'wb') as out_file:
			pickle.dump((final_vals_aa, final_vals_ga, mean_est, nu_est, T*paths), out_file)


	def load_data(filename):

		#load values from pickle file and return

		with open(filename, 'rb') as in_file:
			final_vals_aa,final_vals_ga,mean_est,nu_est,data_count = pickle.load(in_file)


		return final_vals_aa,final_vals_ga,mean_est,nu_est,data_count


	def generate_logratio(lambda0, lambda1, N, transform, mode = 'Gaussian'):

		if mode == 'Gaussian':

			#generate Gaussian log-likelihood ratios (According to example 1)

			gaussian_vector = np.random.normal(size = (N,1))

			#if the data is correlated across agents, generates correlated Gaussian rvs.

			if transform is not None:
				# w,v = np.linalg.eigh(cov)
				# transform = (v @ np.diag(np.sqrt(w)) @ v.T)
				gaussian_vector = transform @ gaussian_vector
			

			log_ratio = (lambda1-lambda0)*(gaussian_vector + lambda0) - (lambda1**2 - lambda0**2)/2
			anti_log_ratio = (lambda1-lambda0)*(-gaussian_vector + lambda0) - (lambda1**2 - lambda0**2)/2

			#anti-log ratio is for antithetic sampling
			return log_ratio,anti_log_ratio


		else:

			#generate Exponential log-likelihood ratios (According to example 2)

			uniform_vector = np.random.rand(N,1)
			exponential_vector = -np.log(uniform_vector)/lambda0
			anti_exponential_vector = -np.log(1-uniform_vector)/lambda0

			#anti-log ratio is for antithetic sampling
			return -(lambda1-lambda0)*exponential_vector+np.log(lambda1)-np.log(lambda0), -(lambda1-lambda0)*anti_exponential_vector+np.log(lambda1)-np.log(lambda0)


	def perform_tests(aa_data, ga_data, mean_est, nu_est, data_count, normalize = False):

		#After reading the data from the pickle file, performs KS and SW tests. Then plots the histograms for visual evidence of the asymptotic normality

		#estimated mean and variance for AA
		print('-------------------------------------------')
		print('estimated mean: ', -mean_est)
		print('estimated std: ', data_count/(data_count-1)*(nu_est - mean_est**2))

		final_vals_aa = aa_data
		final_vals_ga = ga_data
		
		# normalizes the data before visualization
		if normalize:
			final_vals_aa -= np.mean(final_vals_aa)
			final_vals_aa /= np.std(final_vals_aa)

			final_vals_ga -= np.mean(final_vals_ga)
			final_vals_ga /= np.std(final_vals_ga)

		#perform the Kolmogorov-Smirnov and Shapiro-Wilk tests and print the results

		print("\nTest results for AA:")

		kstest_result_aa = stats.kstest(final_vals_aa,stats.norm.cdf)
		print(kstest_result_aa)

		shapiro_result_aa = stats.shapiro(final_vals_aa)
		print(shapiro_result_aa)

		print("\nTest results for GA:")

		kstest_result_ga = stats.kstest(final_vals_ga,stats.norm.cdf)
		print(kstest_result_ga)

		shapiro_result_ga = stats.shapiro(final_vals_ga)
		print(shapiro_result_ga)


		#draw the plots

		rv = stats.norm()
		x = np.linspace(-5,5,100)

		fig = plt.figure()
		ax1 = plt.axes()

		plt.rcParams['text.usetex'] = True

		fig.set_size_inches(6,4)

		ax1.set_xlabel(r"$t$")
		ax1.grid()
		
		sns.histplot(ax = ax1, data=final_vals_aa, stat='density', alpha = 0.3, color = 'blue', edgecolor = 'blue')
		sns.histplot(ax = ax1, data=final_vals_ga, stat='density', alpha = 0.3, color = 'red', edgecolor = 'red')

		legend_description = [r'$f^{(A)}(t)$', r'$f^{G}(t)$']

		#visual comparison with standard normal density

		if normalize:

			norm_density_x = np.linspace(-5,5,100)
			norm_density = rv.pdf(norm_density_x)

			sns.lineplot(ax= ax1, x= norm_density_x, y = norm_density, color = 'black')

			legend_description = [r'$\mathcal{G}(t)$'] + legend_description
		
		ax1.set_ylabel('')
	

		ax1.legend(legend_description)

		plt.show()

	#argument parsing takes place here

	parser = argparse.ArgumentParser(description='Simulate and test the AA and GA federated fusion rules.')
	parser.add_argument('mode', type=str, choices=['s', 't'], help = 's: simulation mode, t: performs test from the pickle data')
	parser.add_argument('--file', type=str, required = True, help = 'file name')
	parser.add_argument('--N_user', type=int, default = 10, help = 'Number of users (nodes) in the setting')
	parser.add_argument('--T', type=int, default = 5000, help = 'The horizon (i in the manuscript)')
	parser.add_argument('--paths', type=int, default = 500, help = 'Number of paths for Monte Carlo simulations')
	parser.add_argument('--theta', type=int, nargs = 2, default = [1, 2], help = 'Parameters of distributions under the null and alternative hypotheses, respectively')
	parser.add_argument('--cov', type=int, default = 0, help = '0 for no correlation across nodes, 1 for the correlation matrix in the manuscript (only valid for the Gaussian case)')
	parser.add_argument('--normalize', type=str, choices = ['y','n'], default = 'n', help = 'Normalization for plotting')
	parser.add_argument('--distribution', type=str, default = 'g', choices = ['g','e'], help = 'The distribution at the nodes (g for Gaussian and e for Exponential)')

	args = parser.parse_args()

	if args.mode == 's':

		# simulation mode

		N_user = args.N_user

		if N_user <= 0:
			raise argparse.ArgumentTypeError("N_user > 0")

		if args.cov != 0:
			cov = np.ones((N_user,N_user))*0.95
			cov += np.eye(N_user)*0.05
			transform = np.linalg.cholesky(cov)

		else:
			transform = None

		T = args.T
		if T <= 0:
			raise argparse.ArgumentTypeError("T > 0")

		paths = args.paths
		if paths <= 0:
			raise argparse.ArgumentTypeError("paths > 0")

		distribution = 'Gaussian' if args.distribution == 'g' else 'Exponential'
		
		simulate(T = T, N_user = N_user, paths = paths, lambda0 = args.theta[0], lambda1 = args.theta[1], outfile_name = args.file, mode = distribution, transform = transform)

	elif args.mode == 't':

		#testing mode

		aa_data,ga_data,mean_est,nu_est,data_count = load_data(args.file)
		aa_data = np.asarray(aa_data)
		ga_data = np.asarray(ga_data)

		perform_tests(aa_data, ga_data, mean_est,nu_est, data_count, normalize = (args.normalize != 'n'))

	else:
		print('invalid argument')

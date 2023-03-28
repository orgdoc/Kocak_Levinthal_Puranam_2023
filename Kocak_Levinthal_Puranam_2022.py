# Code for Kocak, Levinthal, and Puranam (2022)

import numpy as np
import time
import pickle
import random

# Import the following package to speed up the simulation
# If not imported, comment out all lines with "nb."
# Requires installation of Numba package
# https://numba.pydata.org/numba-doc/latest/user/installing.html
import numba as nb


# S denotes number of actions
S = 7
# N denotes number of agents
N = 7

# tau denotes the exploration parameter
# phi denotes the learning rate
# d1 denotes the lower bound for payoffs
# d2 denotes the upper bound for second highest payoff
# c denotes the interdependence level of chi
# IS denotes Influence Structure

# Softmax function used by agents
@nb.njit
def softmax(attraction,agent,tau): # Softmax action selection with attraction vector at time t and agent as paramaters
	S = len(attraction[agent])

	denom = 0
	for i in range(S):
		denom += np.exp((attraction[agent,i])/tau)
	roulette = random.random()

	p = 0
	for i in range(S):
		p += np.exp(attraction[agent,i]/tau)/denom
		if p > roulette:
			return i
	return S-1

# Generate payoffs
@nb.njit
def environment(s,d1,d2): 
	
	if d1 == d2:
		env = np.ones(s)*d1
	else:
		env = 0.01*np.random.choice(np.arange(100*d1,100*d2),s,replace=True)

	env[0] = 1  
	return env

# Generate a mesh influence structure
@nb.njit
def generateMesh(N,asym = 0):
	influence = np.ones((N,N))
	for k in range(N):
		influence[k,k] = 0

	# If asymmetric ties are to be introduced randomly into the mesh network
	if asym>0:
		asymties = np.sort(np.random.choice(np.arange(int(N*(N-1)/2)),asym,replace=False))
		m = 0
		m2 = 0
		for k in range(0,N-1):
			for l in range(k+1,N):
				if m < len(asymties) and m2 == asymties[m]:
					influence[k,l] = 0
					m += 1
				m2 += 1
	return influence

# Calculate resolution of beliefs
@nb.njit
def resolution(belief): 
	return np.max(belief)-np.min(belief)


# Calculate similarity of beliefs
@nb.njit
def similarity(matrix):# Matrix is the attraction matrix with dimension N x S
	N,S = matrix.shape
	diff = np.zeros((N,N))
	for i1 in range(N):
		for j1 in range(i1):
			for s1 in range(S):            
				diff[i1,j1] += np.abs(matrix[i1,s1]-matrix[j1,s1])
	return (1.-2.*np.sum(diff)/(N*(N+1)*S))

# Start simulation
@nb.njit
def simulate(T,c,tau,delt,IS,N = 7,S = 7,phi = None,E = 1000,I = 1,w = 0.5):
	
	PAYOFF = np.zeros((I,E,T)) # Organizational payoff
	SIMILARITY = np.zeros((I,E,T)) # Average similarity of attractions across agents
	RESOLUTION = np.zeros((I,E,T)) # Average resolution of attractions: max(attraction)-min(attraction)
	OPTIMAL = np.zeros((I,E,T)) # Percentage agents reached maximum payoff
	CONVERGENCE = np.zeros((I,E,T)) # Flatness of the distribution of agents over action space
	PAYOFF_CHANGE = np.zeros((I,E,T)) # Change of organizational payoff over time
	ACTION_CHANGE = np.zeros((I,E,T)) # Total number of agents changed actions over time

	En = np.zeros((I,T)) # Averaging PAYOFF over environments
	Es = np.zeros((I,T)) # Averaging SIMILARITY over environments
	Er = np.zeros((I,T)) # Averaging RESOLUTION over environments
	Eo = np.zeros((I,T)) # Averaging OPTIMAL over environments
	Ec = np.zeros((I,T)) # Averaging CONVERGENCE over environments
	Est = np.zeros((I,T)) # Averaging PAYOFF_CHANGE over environments
	Est2 = np.zeros((I,T)) # Averaging ACTION_CHANGE over environments
	
	In = np.zeros((T)) # Averaging PAYOFF over environments and influence structures (in case of stochastically generated)
	Inv = np.zeros((T)) # Standard Dev of average PAYOFF over environments over influence structures (in case of stochastically generated, 0 otherwise)
	Is = np.zeros((T)) # Averaging SIMILARITY over environments and influence structures (in case of stochastically generated)
	Isv = np.zeros((T)) # Standard Dev of average SIMILARITY over environments over influence structures (in case of stochastically generated, 0 otherwise)
	Ir = np.zeros((T)) # Averaging RESOLUTION over environments and influence structures (in case of stochastically generated)
	Irv = np.zeros((T)) # Standard Dev of average RESOLUTION over environments over influence structures (in case of stochastically generated, 0 otherwise)
	Io = np.zeros((T)) # Averaging OPTIMAL over environments and influence structures (in case of stochastically generated)
	Iov = np.zeros((T)) # Standard Dev of average OPTIMAL over environments over influence structures (in case of stochastically generated, 0 otherwise)
	Ic = np.zeros((T)) # Averaging CONVERGENCE over environments and influence structures (in case of stochastically generated)
	Icv = np.zeros((T)) # Standard Dev of average CONVERGENCE over environments over influence structures (in case of stochastically generated, 0 otherwise)
	Ist = np.zeros((T)) # Averaging PAYOFF_CHANGE over environments and influence structures (in case of stochastically generated)
	Ist2 = np.zeros((T)) # Averaging ACTION_CHANGE over environments and influence structures (in case of stochastically generated)

	output = np.zeros((T,13)) # Collecting all outputs in one array

	#DEFINING MODEL'S INTERNAL OBJECTS
	agent_state = np.zeros((T,N))
	Payoff = np.zeros((T,N))
	Similarity = np.zeros((T))
	Resolution = np.zeros((T,N))
	Optimal = np.zeros((T,N))
	Convergence = np.zeros((T,S))
	

	# When a learning rate is not defined, then the agents learn by averaging
	averaging = phi is None

	# If phi is defined, it is assigned as the learning rate to each agent
	# Further work can vary phi across agents
	if not(averaging):
		aphi = np.zeros((N))
		for n in range(N):
			aphi[n] = phi

	# Payoff boundaries
	delta1,delta2 = delt
	
	for i in range(I):
		influence = IS.copy()
		
		for j in range(E):

			Environment = environment(S,delta1,delta2) # Generate environment
			att = 0.1*(np.random.choice(int(11),(N,S),replace=True)) # Initial attractions
			choice_count = np.ones((N,S)) # Initiate count of choices by one in case of averaging

			for t in range(T):

				atte = np.copy(att) # Make a copy of attractions, necessary for influence later
				Similarity[t] = similarity(att) # Calculate similarity

				# Choice and payoff step
				for a in range(N):
					Resolution[t,a] = resolution(att[a]) # Calculate resolution
					choice = int(softmax(att,a,tau))
					agent_state[t,a] = choice # Agent makes a choice
					choice_count[a,choice] += 1.0 # Increment the choice by one
					Payoff[t,a] = Environment[choice] # Retrieve the baseline payoff associated with the choice

				# Calcuate the distribution of agents over agent space
				distshare = np.zeros((S))
				for n in range(N):
					distshare[int(agent_state[t,n])] += 1/N

				# Discounting the payoff by interdependence and learning step
				for a in range(N):
					choice = int(agent_state[t,a])
					Payoff[t,a] = Payoff[t,a]*(1-c*(1-distshare[choice]))
					Optimal[t,a] = int(Payoff[t,a]==1) # Whether agent achieved maximum possible payoff

					if averaging:
						atte[a][choice] = (atte[a][choice]*(choice_count[a,choice]-1)+Payoff[t,a])/choice_count[a,choice] #Average Updating
					else:
						atte[a][choice] += aphi[a]*(Payoff[t,a]-atte[a][choice]) # Reinforcement Learning
						
				Convergence[t] = distshare**2 # Components of Herfindahl Index (For diversity calculation)

				# Influence among agents
				for a in range(N):
					r = np.sum(influence[a]) # Agent gets influenced
					if r>0: # If there is any influence, sum up all attractions of connections and weigh with average
						proximate_attr = np.zeros(S)
						for a2 in range(N):
							if influence[a,a2]:
								proximate_attr += att[a2]

						atte[a] = w*atte[a]+(1-w)*proximate_attr/r

				# New attractions for the next time step are the updated ones
				att = atte.copy()

				#RECORDING RESULTS
				PAYOFF[i,j,t] = float(np.mean(Payoff[t]))
				SIMILARITY[i,j,t] = Similarity[t]
				RESOLUTION[i,j,t] = float(np.mean(Resolution[t]))
				OPTIMAL[i,j,t] = float(np.mean(Optimal[t]))
				CONVERGENCE[i,j,t] = float(np.sum(Convergence[t]))
				if t>0:
					PAYOFF_CHANGE[i,j,t] = np.abs(PAYOFF[i,j,t]-PAYOFF[i,j,t-1])
					for a in range(N):
						ACTION_CHANGE[i,j,t] += int(agent_state[t,a] != agent_state[t-1,a])
			
	# Averaging outputs over environments
	for i in range(I):
		for t in range(T):
			En[i,t] = float(np.mean(PAYOFF[i,:,t]))
			Es[i,t] = float(np.mean(OPTIMAL[i,:,t]))
			Er[i,t] = float(np.mean(RESOLUTION[i,:,t]))
			Eo[i,t] = float(np.mean(SIMILARITY[i,:,t]))
			Ec[i,t] = float(np.mean(CONVERGENCE[i,:,t]))
			Est[i,t] = float(np.mean(PAYOFF_CHANGE[i,:,t]))
			Est2[i,t] = float(np.mean(ACTION_CHANGE[i,:,t]))

	# Calculate mean and standard deviation of outputs over influence structures
	for t in range(T):
		In[t] = float(np.mean(En[:,t]))
		Inv[t] = float(np.std(En[:,t]))
		Io[t] = float(np.mean(Eo[:,t]))
		Iov[t] = float(np.std(Eo[:,t]))
		Ir[t] = float(np.mean(Er[:,t]))
		Irv[t] = float(np.std(Er[:,t]))
		Is[t] = float(np.mean(Es[:,t]))
		Isv[t] = float(np.std(Es[:,t]))
		Ic[t] = float(np.mean(Ec[:,t]))
		Icv[t] = float(np.std(Ec[:,t]))
		Ist[t] = float(np.mean(Est[:,t]))
		Ist2[t] = float(np.mean(Est2[:,t]))

	# Aggregating outputs in one variable output
	for t in range(T):
		output[t,0] = t+1
		output[t,1] = Ic[t]
		output[t,2] = In[t]
		output[t,3] = Icv[t]
		output[t,4] = Inv[t]
		output[t,5] = Is[t]
		output[t,6] = Isv[t]
		output[t,7] = Ir[t]
		output[t,8] = Irv[t]
		output[t,9] = Io[t]
		output[t,10] = Iov[t]
		output[t,11] = Ist[t]
		output[t,12] = Ist2[t]
	
	return output

# Enter all parameter values to run simulations on 
times = [50]
taus = [0.05]
chis = [1.] # np.arange(0,11)/10
deltas = [(0,1)] #[(i,k) for k in np.arange(0,11)/10. for i in np.arange(0,11)/10. if k>=i]
ws = [0.5]

# Crowd - No influence
IS = np.zeros((N,N))

# Flat team
#IS = np.ones((N,N))
#for n in range(N):
#	IS[n,n] = 0

# Hierarchical Team
#IS = np.ones((N,N))
#for n in range(N):
#	IS[0,n] = 0
#	IS[n,n] = 0

# Star-like Team
#IS = np.zeros((N,N))
#for n in range(1,N):
#	IS[n,0] = 1

# All combinations of parameters to run simulation with
combinations = [(t,c,ta,d,w) for t in times for c in chis for ta in taus for d in deltas for w in ws]
sym = 0

payoffs = {}
for icombo,combination in enumerate(combinations):
	print(icombo,len(combinations),combination)
	start = time.time()
	T,c,tau,delta,w = combination
	payoffs[combination] = simulate(T,c,tau,delta,IS,N = N,S = S, w = w)
	print('Secs',time.time()-start)

with open('Results.pkl','wb') as op:
	pickle.dump(payoffs,op)

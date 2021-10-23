from jax import grad, jit, vmap, random, nn
from matplotlib import pyplot as plt
import jax.numpy as np
from tqdm import tqdm
import argparse
import gym


class Buffer(object):

	def __init__(self, gamma):
		self.gamma=gamma
		self.reset()

	def store(self, a, o, p_o, r, v, done):
		
		self.act.append(a)
		self.obs.append(o)
		self.prev_obs.append(p_o)
		self.rew.append(r)
		self.done.append(done)
		self.val.append(v)

	def calculate_advantages(self):

		for i in reversed(range(len(self.rew)-1)):
			self.rew[i] += self.rew[i+1]*self.gamma*(1-self.done[i])
		
		self.rew = np.array(self.rew)
		self.done = np.array(self.done)
		self.val = np.array(self.val)

		print(self.rew.shape, self.val.shape)

		self.adv =  self.rew[:-1] + self.val[1:] - self.val[:-1]

		plt.plot(self.adv)
		plt.show()

	def get(self):

		a = np.array(self.act)
		o = np.array(self.obs)
		p_o = np.array(self.prev_obs)

		self.calculate_advantages()

		r = self.rew
		d = self.done
		v = self.val

		

		adv = np.array(self.adv)

		return a, o, p_o, r, d, adv

	def reset(self):

		self.act = []
		self.obs = []
		self.prev_obs = []
		self.rew = []
		self.adv = []
		self.done = []
		self.val = []


class ActorCritic(object):

	def __init__(self, sizes, key):
		
		self.key = random.PRNGKey(key)
		self.p_params = init_network_params(sizes, self.key)
		sizes[-1] = 1
		self.v_params = init_network_params(sizes, self.key)

	def act(self, x):
		
		pi, v = self.forward(x)
		_, self.key = random.split(self.key)
		act = random.categorical(self.key, pi)[0]
		return act.item(), v

	def f(self, params, x):

		out = x

		for w, b in params[:-1]:
			out = np.dot(out, w) + b
			out = nn.relu(out)

		w, b = params[-1]
		out = np.dot(out, w) + b

		return out

	def forward(self, x):

		pi = self.f(self.p_params, x)
		pi = nn.log_softmax(pi)

		v = self.f(self.v_params, x)
		
		return pi, v


def mse(x, y):
	return (x-y)**2 * 1/x.shape[0]


def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (1, n))


def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def pi_loss(log_pi, adv, alpha):
	return (log_pi * adv).mean()

#@jit
def update(agent, buffer, pi_lr, v_lr):

	a, o, p_o, r, d, adv = buffer.get()

	pi, _ = agent.forward(o)
	log_pi = pi
	grads = grad(pi_loss)(agent.p_params, log_pi, adv)
	agent.pi_params = [(w - alpha * dw, b - alpha * db)
		  for (w, b), (dw, db) in zip(agent.p_params, grads)]

	for i in range(10):

		grads = grad(v_loss)(agent.v_params, v, r)
		agent.v_params = [(w - alpha * dw, b - alpha * db)
			  for (w, b), (dw, db) in zip(agent.v_params, grads)]

#def optimize(v_grads):


def vpg(
	agent,
	env,
	buffer,
	num_epochs,
	max_steps,
	pi_lr,
	v_lr,
	gamma,
	lamda,
	render,
	*args,
	**kwargs
	):
	
	for epoch in range(num_epochs):

		prev_obs = env.reset()

		for step in range(max_steps):

			if render:
				env.render()

			act, val = agent.act(prev_obs)

			obs, rew, done, _ = env.step(act)

			buffer.store(act, obs, prev_obs, rew, val, done)

			if step >= max_steps or done:
				break

		update(agent, buffer, pi_lr, v_lr)
	
	env.close()


def get_env_dim(env):
	act_dim = env.action_space.n
	obs_dim = env.observation_space.shape[0]
	return obs_dim, act_dim


def run(args):

	env = gym.make(args.env_name)
	obs_dim, act_dim = get_env_dim(env)
	sizes = [obs_dim, 256, 256, act_dim]
	agent = ActorCritic(sizes, key=123123)
	buffer = Buffer(args.gamma)

	vpg(agent=agent, env=env, buffer=buffer, **vars(args))


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', default='vpg', help='name of experiment')
	parser.add_argument('--env_name', default='CartPole-v0', help='name of gym env')
	parser.add_argument('--pi_lr', default=1e-3, help='policy learning rate')
	parser.add_argument('--v_lr', default=1e-3, help='value learning rate')
	parser.add_argument('--num_epochs', default=1, help='number of epochs')
	parser.add_argument('--num_steps', default=200, help='steps per epoch')
	parser.add_argument('--max_steps', default=200, help='max steps per episode')
	parser.add_argument('--gamma', default=0.1, help='discount factor')
	parser.add_argument('--lamda', default=1, help='discount factor')
	parser.add_argument('--hid_dim', default=100, help='network width')
	parser.add_argument('--seed', default=1, help='random seed')
	parser.add_argument('--save', default=False, help='is saving results')
	parser.add_argument('--render', default=True, help='render env')

	args = parser.parse_args()

	run(args)


if __name__ == '__main__':
	main()
import gym
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm


class MLP(torch.nn.Module):


	def __init__(self, in_dim=4, out_dim=2) -> None:
		super().__init__()

		self.l1 = torch.nn.Linear(in_dim, 256)
		self.l2 = torch.nn.Linear(256, 256)
		self.l3 = torch.nn.Linear(256, out_dim)
		self.relu = torch.nn.LeakyReLU()
		self.softmax = torch.nn.Softmax(0)

		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

	def forward(self, x):

		out = torch.Tensor(x)

		out = self.l1(out)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)
		out = self.softmax(out)

		return out

	
class Policy(MLP):

	def act(self, x):

		out = self.forward(x)
		dist = Categorical(out)

		action = dist.sample()
		log_prob = dist.log_prob(action)

		return action.numpy(), log_prob





def collect_trajectories(policy, num_steps=1000, render=False):

	if render: 
		env = gym.make("CartPole-v1", render_mode="human")

	else:
		env = gym.make("CartPole-v1")
	
	observation, info = env.reset(seed=42)

	trajectories = []

	for _ in range(num_steps):

		action, log_prob = policy.act(observation)

		prev_observation = observation

		observation, reward, done, trunc, info = env.step(action)

		trajectories.append([prev_observation, action, log_prob, reward, done])

		if done or trunc:
			observation, info = env.reset()

		if render: env.render()

	env.close()

	return trajectories




def main():
	
	theta = None
	phi = None

	policy = Policy(in_dim=4, out_dim=2)
	critic = MLP(in_dim=4, out_dim=1)

	pbar = tqdm(total=100)

	for k in range(100):

		# collect trajectories
		t = collect_trajectories(policy, num_steps=1000)


		# compute rewards-to-go
		for idx in range(1, len(t)):
			# if done reset
			if t[idx][4]:
				continue
			# cumulative sum
			t[idx][3] += t[idx-1][3]



		# compute advantage estimates
		adv_est = []
		for idx in range(1, len(t)):

			if t[idx-1][4] is True:
				adv_est.append(t[idx-1][3])
				continue

			adv_est.append(t[idx][3] - t[idx-1][3])

		adv_est.append(t[idx-1][3])



		# estimate policy gradient
		g_k = -1 * (1/len(t)) * torch.cat([(t[idx][2]*adv_est[idx]).reshape(1, 1) for idx in range(len(t))]).sum()


		# compute policy update
		g_k.backward()
		policy.optimizer.step()
		
		tqdm.write("Updating message...")
		desc = "episodes: {}".format(sum([t[idx][4] for idx in range(len(t))]))
		pbar.set_description(desc, refresh=True)
		pbar.update(1)

	collect_trajectories(policy, num_steps=1000, render=True)


if __name__ == "__main__":
	main()
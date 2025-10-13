from utils.dp_utils import GaussianMomentsAccountant, MomentsAccountant
accountant = GaussianMomentsAccountant(total_examples=60000, max_moment_order=32)

for i in range(500):
    accountant.accumulate_privacy_spending(sigma=4.0, num_examples=600)
eps_deltas = accountant.get_privacy_spent(target_deltas=[1e-5,0.05])
for eps, delta in eps_deltas:
    print("epsilon:",eps, "delta", delta)
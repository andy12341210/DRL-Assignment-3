epsilon_start = 0.9
epsilon_end = 0.03
epsilon_decay = 0.95
num_episodes = 100
gamma=0.99
# lr=1e-3
lr=1e-4
buffer_capacity=1000000
batch_size=32
tau=1e-3

cfg = [  
            gamma,
            lr,
            buffer_capacity,
            batch_size,
            tau,
            num_episodes,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
]
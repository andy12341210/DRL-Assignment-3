epsilon_start = 0.95
epsilon_end = 0.01
epsilon_decay = 0.9995
num_episodes = 30000
gamma=0.99
lr=1e-4
buffer_capacity=100000
batch_size=64
tau=1e-3
steps = 3
region = [[600,1400]]

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
            steps,
            region
]
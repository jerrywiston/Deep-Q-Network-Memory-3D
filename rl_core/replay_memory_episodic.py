import numpy as np
import torch 
import random
from collections import deque

class Memory(object):
    def __init__(self, memory_size=1000, seq_len=8):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.seq_len = seq_len
        self.local_memory = []
        
    def __len__(self):
        return len(self.buffer)
        
    def push(self, state, action, reward, next_state, done, end_episode=False):
        data = (state, action, reward, next_state, done)
        self.local_memory.append(data)
        if done or end_episode:
            if len(self.local_memory) >= self.seq_len:
                self.memory.append(self.local_memory)
            self.local_memory = []

    def merge_state_dict(self, state_list):
        state_seq = {}
        for state in state_list:
            for key in state:
                if key not in state_seq:
                    state_seq[key] = []
                state_seq[key].append(np.expand_dims(state[key], 0))
        for key in state_seq:
            state_seq[key] = np.concatenate(state_seq[key], 0)
        return state_seq

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = {}, [], [], {}, []
        p = np.array([len(episode) for episode in self.memory])
        p = p / p.sum()
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)

        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - self.seq_len)
            transitions = episode[start:start + self.seq_len]
            state_seq, action_seq, reward_seq, next_state_seq, done_seq = tuple(zip(*transitions))
            state_seq = self.merge_state_dict(state_seq)
            action_seq = np.array(action_seq).reshape(-1,1)
            reward_seq = np.array(reward_seq).reshape(-1,1)
            next_state_seq = self.merge_state_dict(next_state_seq)
            done_seq = np.array(done_seq).reshape(-1,1)
            
            actions.append(action_seq)
            rewards.append(reward_seq)
            dones.append(done_seq)
            # state
            for key in state_seq:
                if key in states:
                    states[key].append(state_seq[key])
                else:
                    states[key] = [state_seq[key]]
            # next state
            for key in next_state_seq:
                if key in next_states:
                    next_states[key].append(next_state_seq[key])
                else:
                    next_states[key] = [next_state_seq[key]]
        return states, actions, rewards, next_states, dones
    
    def sample_torch(self, batch_size, device, discrete_control=True):
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        
        if discrete_control:
            b_a = torch.LongTensor(np.stack(actions, 0)).to(device)
        else:
            b_a = torch.FloatTensor(np.stack(actions, 0)).to(device)
        
        b_r = torch.FloatTensor(np.stack(rewards, 0)).to(device)
        b_d = torch.FloatTensor(np.stack(dones, 0)).to(device)
        b_s, b_s_ = {}, {}
        for key in states:
            b_s[key] = torch.FloatTensor(np.stack(states[key], 0)).to(device)
            b_s_[key] = torch.FloatTensor(np.stack(next_states[key], 0)).to(device)
        
        #b_a, b_r, b_d = b_a[:,0,...], b_r[:,0,...], b_d[:,0,...]
        #for key in b_s:
        #    b_s[key], b_s_[key] = b_s[key][:,0,...], b_s_[key][:,0,...]
        #print(b_s['obs'].shape, b_a.shape, b_r.shape, b_d.shape)
        return b_s, b_a, b_r, b_s_, b_d
import torch
import torch.nn.functional as F
from .q_network import QNetwork

class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        grad_clip: float = 5.0,
        device: str | None = None,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_clip = grad_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        self.q = QNetwork(obs_dim, num_actions).to(self.device)
        self.q_target = QNetwork(obs_dim, num_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, epsilon: float):
        # obs: np.array shape (obs_dim,)
        import random
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        qvals = self.q(x)  # (1, A)
        return int(torch.argmax(qvals, dim=1).item())
    
    @torch.no_grad()
    def q_values(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x).squeeze(0)  # shape (A,)
        return q.detach().cpu().numpy()

    def update(self, batch):
        obs, act, rew, nxt, done = batch

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew_t = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        nxt_t = torch.tensor(nxt, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(obs_t).gather(1, act_t)

        with torch.no_grad():
            # standard DQN target: r + gamma * max_a' Q_target(s', a') * (1-done)
            max_next = self.q_target(nxt_t).max(dim=1, keepdim=True).values
            target = rew_t + self.gamma * max_next * (1.0 - done_t)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optim.step()

        return float(loss.item()), float(q_sa.mean().item())

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({"model": self.q.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["model"])
        self.sync_target()
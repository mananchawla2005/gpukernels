import torch
from tqdm import tqdm
class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the benefits of:

    Momentum: keeps an exponentially decaying average of past gradients (1st moment).

    RMSProp: keeps an exponentially decaying average of squared gradients (2nd moment).

    It adapts the learning rate for each parameter based on how frequently and how drastically it changes.
    """
    def __init__(self, params , lr=1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params] # First Moment
        self.v = [torch.zeros_like(v.data) for v in self.params] # Second Moment
        self.t = 0 # Time Step
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()

    def step(self):
        self.t +=1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad.data
            
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            self.m[i] = self.beta1*self.m[i]+(1-self.beta1)*grad
            self.v[i] = self.beta2*self.v[i]+(1-self.beta2)*grad*grad

            # Bias Correction

            m_hat = self.m[i] / (1-self.beta1**self.t)
            v_hat = self.v[i] / (1-self.beta2**self.t)

            # Updating parameters

            param.data.sub_(self.lr * m_hat / (torch.sqrt(v_hat) + self.eps))

model = torch.nn.Linear(10, 1)
optimizer = AdamOptimizer(model.parameters(), lr=1e-3)

x = torch.randn(32, 10)
target = torch.randn(32, 1)
loss_fn = torch.nn.MSELoss()

for step in tqdm(range(100)):
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    
    print(f"Step {step}: Loss = {loss.item():.4f}")
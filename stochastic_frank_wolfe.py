"""Stochastic Frank Wolfe Algorithm."""

import math

import torch


class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale="diameter", momentum=0.9):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if not (rescale in ["diameter", "gradient", None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # Parameters
        self.rescale = rescale

        defaults = dict(lr=learning_rate, momentum=momentum)
        super(SFW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Add momentum
                momentum = group["momentum"]
                if momentum > 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = d_p.detach().clone()
                    else:
                        param_state["momentum_buffer"].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state["momentum_buffer"]

                v = constraints[idx].lmo(d_p)  # LMO optimal solution

                if self.rescale == "diameter":
                    # Rescale lr by diameter
                    factor = 1.0 / constraints[idx].get_diameter()
                elif self.rescale == "gradient":
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)  # type: ignore
                else:
                    # No rescaling
                    factor = 1

                lr = max(0.0, min(factor * group["lr"], 1.0))  # Clamp between [0, 1]

                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss


class PositiveKSparsePolytope:
    """Polytopes with vertices v in {0, 1}^n such that exactly k entries are nonzero.

    This is exactly the intersection of B_1(k) with B_inf(1) and {x | x>=0}
    Note: Here the dimension is n, but the dimension of the oracle input/solution is bs*n
    """

    def __init__(self, n, bs, k=1):
        self.k = min(k, n)
        self.diameter = math.sqrt(2 * k) if 2 * k <= n else math.sqrt(n)

    @torch.no_grad()
    def get_diameter(self):
        return self.diameter

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in PositiveKSparsePolytope minimizing v*x"""
        # NOTE: we have to do this per image
        v = torch.zeros_like(x)
        minIndices = torch.topk(x.flatten(start_dim=1), k=self.k, largest=False).indices
        v.flatten(start_dim=1).scatter_(1, minIndices, 1.0)
        v[x >= 0] = 0.0

        return v

    @torch.no_grad()
    def shift_inside(self, x, check_necessity=False):
        """
        Projects x to the PositiveKSparsePolytope.
        This is a valid projection, although not the one mapping to minimum distance points.

        Args:
            x (torch.tensor): Input data.
            check_necessity (bool, optional): _description_. Defaults to False.

        Returns:
            v: Projected input onto k-sparse vertices of polytope.
        """
        # Check if necessary
        if check_necessity:
            l1 = torch.norm(x.flatten(start_dim=1), p=1, dim=1)  # type: ignore
            linf = torch.norm(x, p=float("inf"))  # type: ignore
            if linf <= 1 and (l1 <= self.k).all() and (x >= 0).all():
                # Is in the polytope, no need to shift
                return x

        # For now, we just take the closest vertex
        # Note: we have to do this per image
        v = torch.zeros_like(x).flatten(start_dim=1)
        maxIndices = torch.topk(torch.abs(x.flatten(start_dim=1)), k=self.k).indices.to(x.device)
        v.scatter_(1, maxIndices, 1.0)
        v.requires_grad = True

        return v.reshape(x.shape)

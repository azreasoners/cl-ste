import torch

####################################################################################
# Definitions of General Functions (under binary logic)
####################################################################################

def bp(x):
    """ binarization function bp(x) = 1 if x >= 0.5; bp(x) = 0 if x < 0.5

    @param x: a real number in [0,1] denoting a probability
    """
    return torch.clamp(torch.sign(x-0.5) + 1, max=1)

def binarize(x):
    """ binarization function binarize(x) = 1 if x >= 0; binarize(x) = -1 if x < 0

    Remark:
        This function is indeed the b(x) function in the paper.
        We use binarize(x) instead of b(x) here to differentiate function B(x) later.

    @param x: a real number of any value
    """
    return torch.clamp(torch.sign(x) + 1, max=1)

def sSTE(grad_output, x=None):
    """
    @param grad_output: a tensor denoting the gradient of loss w.r.t. Bs(x)
    @param x: the value of input x
    """
    return grad_output * (torch.le(x, 1) * torch.ge(x, -1)).float() # clipped Relu with range [-1,1]

# B(x) denotes bp(x) with iSTE
class Disc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return bp(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
B = Disc.apply

# Bi(x) denotes binarize(x) with iSTE
class DiscBi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
Bi = DiscBi.apply

# Bs(x) denotes binarize(x) with sSTE
class DiscBs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = sSTE(grad_output, x)
        return grad_input
Bs = DiscBs.apply

def one(x):
    return (x == 1).int()

def minusOne(x):
    return (x == -1).int()

def zero(x):
    return (x == 0).int()

def noneZero(x):
    return (x != 0).int()

####################################################################################
# Definitions of regularizers
####################################################################################


##########
# Bound
# we limit the size of NN output values
##########

def reg_bound(output):
    return output.pow(2).mean()

##########
# CNF
##########

def reg_cnf(C, v, g):
    """
    @param C: a matrix of shape (m, n) where m is the number of clauses, n is the number of atoms
    @param v: a vector in {0,1}^n, denoting the False/True values for n atoms
              from NN outputs as well as given information
    @param g: a vector in {-1,0,1}^n, denoting the False/Unknown/True values for n atoms 
              solely from given information (e.g. labels)
    """
    if len(C.shape) == 2:
        C = C.unsqueeze(0) # (1, m, n)

    v, g = v.unsqueeze(1), g.unsqueeze(1) # (batchSize, 1, n)
    L_v, L_g = one(C) * v + minusOne(C) * (1-v), C * g # (batchSize, m, n)
    unsat = (1 - L_v).prod(dim=-1) # (batchSize, m)
    up = (C.abs().sum(dim=-1) - minusOne(L_g).sum(dim=-1)) == 1 # (batchSize, m)

    # 1. loss from clauses where unit propagation cannot be applied # (batchSize, m)
    keep = (one(L_v) * (1-L_v) + zero(L_v) * L_v).sum(dim=-1)
    L_sat = (keep * zero(unsat)).mean()
    L_unsat = (unsat * one(unsat)).mean()

    # 2. loss from clauses where unit propagation can be applied 
    L_deduce = unsat[up & (unsat == 1)].sum(dim=-1).mean()
    return L_sat, L_unsat, L_deduce

##########
# hint from given digits in Sudoku
##########

def reg_hint(probs, g):
    """
    Args:
        probs: a vector in R^n, denoting the NN predictions in probabilities; (batchSize, n)
        g: a vector in {-1,0,1}^n, denoting the False/Unknown/True values for n atoms
           solely from given information (e.g. labels); (batchSize, n)
    """
    L_hint = one(g) * (1 - bp(probs)) * (1 - B(probs)) + minusOne(g) * bp(probs) * B(probs) # (batchSize, n)
    return L_hint.mean()

##########
# continuous regularizer for Sudoku
##########

def reg_sudoku_cont(probs, A_row, A_col, A_box, A_same):
    """Return a real number loss
    Args:
        probs: a tensor of shape (batch_size,9,9,9) of probabilities
    """
    y = probs.view(-1, 81, 9) # (batch_size,81,9)
    Ay_row = torch.einsum('ij,ljk->lik', (A_row + A_same, y)) # (batch_size,81,9)
    Ay_col = torch.einsum('ij,ljk->lik', (A_col + A_same, y)) # (batch_size,81,9)
    Ay_box = torch.einsum('ij,ljk->lik', (A_box + A_same, y)) # (batch_size,81,9)
    return (Ay_row - 1).pow(2).mean() + (Ay_col - 1).pow(2).mean() + (Ay_box - 1).pow(2).mean()

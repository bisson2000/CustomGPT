"""
bigram

The bigram is not great with the multinomial approach.

The bigram with the gradient descent is much more flexible
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

# How to create the bigram manually
#bigram_count = {}
#for w in words:
#    chs = ["<S>"] + list(w) + ["<E>"] # start and end
#    for ch1, ch2 in zip(chs, chs[1:]):
#        bigram = (ch1, ch2)
#        bigram_count[bigram] = bigram_count.get(bigram, 0) + 1


#print(sorted(bigram_count.items(), key = lambda kv: -kv[1]))


chars = sorted(list(set(''.join(words))))
string_to_int = {s: i+1 for i,s in enumerate(chars)}
string_to_int["."] = 0
int_to_string = {i: s for s, i in string_to_int.items()}

N = torch.zeros((27, 27), dtype=torch.int32)

# N is the bigram tensor
# Fill
for w in words:
    chs = ["."] + list(w) + ["."] # start and end
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = string_to_int[ch1]
        ix2 = string_to_int[ch2]
        N[ix1, ix2] += 1

#plt.imshow(N)
#plt.waitforbuttonpress(100)


g = torch.Generator().manual_seed(2147483647)
#probability_start = N[0].float()
#probability_start = probability_start / probability_start.sum() # Normalized
#index_start = torch.multinomial(probability_start, num_samples=1, replacement=True, generator=g)

# Generating names out of multinomial probability
#for _ in range(10):
#    out = []
#    ix = 0
#    while True:
#        p = N[ix].float()
#        p = p / p.sum()
#        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#        out.append(int_to_string[ix])
#        if ix == 0:
#            break
#
#    print(''.join(out))

P = (N+1).float() # The +1 is for model smoothing to prevent -infinity probabilities
P /= P.sum(1, keepdim=True) # do counts on rows, then normalize rows

for _ in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(int_to_string[ix])
        if ix == 0:
            break

    print(''.join(out))


log_likelihood = 0.0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."] # start and end
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = string_to_int[ch1]
        ix2 = string_to_int[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

# Likelihood: product of probabilities
# Log-Likelihood: take the log of probability
print(f'{log_likelihood=}')
nll = -log_likelihood # loss function. The lowest is 0 (good). The higher it is, the worst it is
print(f'{nll=}')
average_nll = nll / n
print(f'{average_nll=}')


# We want to maximize the likelihood
# aka maximize the log_likelihood
# aka minimize the nll
# aka minimize the average nll




# Gradient-based optimization to tune the model

# Create the training set of birgrams (x, y) (input, expected)
xs = []
ys = []
for w in words:
    chs = ["."] + list(w) + ["."] # start and end
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = string_to_int[ch1]
        ix2 = string_to_int[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.nelement()

# init network
W = torch.randn((27, 27), generator=g, requires_grad=True) # Weights

# gradient descent:
for k in range(100):
    # Forward pass
    # one-hot encoding: encode integers
    x_encoded = F.one_hot(xs, num_classes=27).float()
    # Let's build the neurons
    logits = x_encoded @ W # log-counts
    counts = logits.exp() # Equivalent to N
    probs = counts / counts.sum(1, keepdim=True) # Normalized probability. this is a softmax
    # result shape: (size_xs x 27) @ (27 x 27) = size_xs x 27
    # (x_encoded @ W)[3, 13]: firing rate of 13 neuron looking at the 3rd input

    # calculate loss
    # regularisation loss
    REGULARISATION_LOSS_RATE = 0.01 # Like a spring that pushes W towards 0
    loss = -probs[torch.arange(num_examples), ys].log().mean() + REGULARISATION_LOSS_RATE*(W**2).mean()

    # backward pass
    W.grad = None # set gradient to zero
    loss.backward()

    # update
    LEARNING_RATE = 50
    W.data += -LEARNING_RATE * W.grad
    print("loss: ", loss.item())


# sample from the neural network
for _ in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # predict log-counts
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(int_to_string[ix])
        if ix == 0:
            break

    print(''.join(out))


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set(''.join(words))))
string_to_int = {s: i+1 for i,s in enumerate(chars)}
string_to_int["."] = 0
int_to_string = {i: s for s, i in string_to_int.items()}


block_size = 3 # context length
inputs = []
labels = []
for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = string_to_int[ch]
        inputs.append(context)
        labels.append(ix)

        context = context[1:] + [ix] # crop and append

X = torch.tensor(inputs)
Y = torch.tensor(labels)

print(X.shape, X.dtype, Y.shape, Y.dtype)

def build_dataset(words):
    block_size = 3
    X = []
    Y = []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = string_to_int[ch]
            X.append(context)
            Y.append(ix)

            context = context[1:] + [ix] # crop and append
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# create lookup table
C = torch.randn((27, 2))

#F.one_hot(torch.tensor(5), num_classes=27).float() @ C # same as C[5]
#print(C)
#print(C.dim_order())
#print(C[1, 0])

embedding = C[X] # size is total x 3 x 2
#print(embedding)
#print(embedding.dim_order())
#print(embedding[2])
#print(embedding[2, 0, 1])

# building hidden layer
W1 = torch.randn((6, 100)) # 6 because of 3 dimensional embeddings x 2
b1 = torch.randn(100) # 100 chosen arbitrarly

# transform embedding to a size of: total x 6
#torch.cat([embedding[:, 0, :], embedding[:, 1, :], embedding[:, 2, :]], 1)
#torch.cat(torch.unbind(embedding, 1), 1)
hidden = embedding.view(-1, 6) @ W1 + b1 # -1 to make pytorch infer it is the same size
hidden = torch.tanh(hidden) # total x 100

W2 = torch.randn((100, 27)) # 27 possible caracters
b2 = torch.randn(27) # 27 possible caracters
logits = hidden @ W2 + b2

counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)

#
loss = prob[torch.arange(Y.shape[0]), Y] # for every row a in prob, select the column that matches Y[a]
loss = -loss.log().mean() # The loss we want to minimize


# Code more concatenated:
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100)) # 6 because of 3 dimensional embeddings x 2
b1 = torch.randn(100) # 100 chosen arbitrarly
W2 = torch.randn((100, 27)) # 27 possible caracters
b2 = torch.randn(27) # 27 possible caracters
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# to figure out the learning rate
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri = []
lossi = []

for i in range(10000):
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,))

    # forward pass
    embedding = C[X[ix]]
    hidden = torch.tanh(embedding.view(-1, 6) @ W1 + b1) # total x 100
    logits = hidden @ W2 + b2
    # equivalent to counts, prob and loss calculation before, 
    # but more efficient and won't make floats go out of range
    loss = F.cross_entropy(logits, Y[ix]) 
    #print("Loss: ", loss.item())

    for p in parameters:
        p.grad = None

    # backward pass
    loss.backward()

    # update
    LEARNING_RATE = 0.1#lrs[i]
    # you first find a good learning rate, and then do a learning rate
    # decay (reduced learning rate) at the end

    for p in parameters:
        p.data += -LEARNING_RATE * p.grad

    # track stat
    # lri.append(lre[i])
    # lossi.append(loss.item())

# to find out the best learning rate
# plt.plot(lre, lossi) # we find out that .1 is good
# plt.waitforbuttonpress(10)
print("Loss: ", loss.item())

# split dataset in 3 slits:
# training split, dev/validation split, test split
# 80%, 10%, 10%











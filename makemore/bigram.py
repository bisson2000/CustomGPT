"""
bigram

The bigram is not great.
"""
import torch
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

P = N.float()
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



for w in words[:3]:
    chs = ["."] + list(w) + ["."] # start and end
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = string_to_int[ch1]
        ix2 = string_to_int[ch2]
        prob = P[ix1, ix2]
        print(f'{ch1}{ch2}: {prob:.4f}')

# Likelyhood: product of probabilities
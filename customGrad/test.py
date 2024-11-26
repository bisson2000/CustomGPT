from customGrad import Value
from neuron import Neuron, Layer, MLP
from graphviz import Digraph

# Visualization helper
def trace(root: Value):
    # set of nodes and edges in a graph
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()
    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root : Value):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"}) # left to right
    
    nodes, edges = trace(root)
    # Create Value nodes
    for n in nodes:
        uid = str(id(n))
        # square node
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape = "record")

        # op node
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            # connect the op node to its result node
            dot.edge(uid + n._op, uid)

    # Connect child to parent
    for child, parent in edges:
        # connect nodes to the op
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot



a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
d = a * b
d.label = "d"
e = d + c
e.label = "e"
f = Value(-2.0, label="f")
L = e * f
L.label = "L"
L.grad = 1.0


# backprop manually
#L._backward()
#e._backward()
#d._backward()

#def topological_sort(end: Value):
#    res: list[Value] = []
#    visited = set()
#    def build_topological_sort(v: Value):
#        if v in visited:
#            return
#        visited.add(v)
#        for child in v._prev:
#            build_topological_sort(child)
#        res.append(v)
#    
#    build_topological_sort(end)
#    return res
#
#topo = topological_sort(L) # end node is at the end of the list
#for node in reversed(topo):
#    node._backward()

L.backward()

#print(L)
#draw_dot(L).view()

LEARNING_RATE = 0.5

#a.data += LEARNING_RATE * a.grad
#b.data += LEARNING_RATE * b.grad
#c.data += LEARNING_RATE * c.grad
#f.data += LEARNING_RATE * f.grad


#x = [2.0, 3.0]
x = [2.0, 3.0, -1.0]
#n = Layer(2, 3)
#n = Neuron(2)
n = MLP(len(x), [4, 4, 1]) # size x -> 4 neurons -> 4 neurons -> 1 output
#print(n(x))
#draw_dot(n(x)[0]).view()

# Example of loss calculation
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
y_truths = [1.0, -1.0, -1.0, 1.0] # desired targets
y_preds = [n(x)[0] for x in xs]
print("predictions")
print(y_preds)

losses: list[Value] = [(y_output - y_truth)**2 for y_truth, y_output in zip(y_truths, y_preds)]
#print("Individual mean squared losses")
#print(losses)

loss = sum(losses)
print("Initial loss")
print(loss)

loss.backward()
print("A weight's gradient")
print(n.layers[0].neurons[0].w[0].grad)

print("A weight's value")
print(n.layers[0].neurons[0].w[0].data)

#print("All parameters")
#print(n.parameters())

# Update parameters:
# The gradient is a vector that points in the direction
# of an increase in loss
# if we follow this vector, we will increase the loss.
# Hence the -1.0
for p in n.parameters():
    p.data += -1.0 * LEARNING_RATE * p.grad

print("A weight's new value")
print(n.layers[0].neurons[0].w[0].data)

y_preds = [n(x)[0] for x in xs]
print("new predictions")
print(y_preds)

losses: list[Value] = [(y_output - y_truth)**2 for y_truth, y_output in zip(y_truths, y_preds)]
loss = sum(losses)
print("New loss, should be lower")
print(loss)

# training loop
for k in range(100):
    # forward pass
    y_preds = [n(x)[0] for x in xs]
    
    # evaluate loss
    loss: Value = sum((y_output - y_truth)**2 for y_truth, y_output in zip(y_truths, y_preds))
    
    # flush the grad (zero grad)
    for p in n.parameters():
        p.grad = 0.0
    
    # backwards pass
    loss.backward()
    
    # update parameters
    for p in n.parameters():
        p.data += -1.0 * LEARNING_RATE * p.grad
    
    print("Iteration ", k, loss.data)

print("Final predictions: ", y_preds)
## Script following Andrej Karpathy youtube lecture on Building Micrograd.
## Contains classes to assist with neuron arithmetic and differentiation,
## resulting in a simple Autograd engine which implements backprop (for scalars)
## and an example neural net with a simple training loop.

import math
import numpy as np
from graphviz import Digraph

## Next two functions are just for visualizing the neural net.
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

## Used for ordering the nodes in preparation for backprop.
## Note: mutates vis and ret.
def topologicalSort(node, vis, ret):
    if node not in vis:
        vis.add(node)
        for c in node._prev:
            topologicalSort(c, vis, ret)
        ret.append(node)


## Value class is used for nodes in the neural net, with ability to track 
## value and transition function (in terms of child nodes and operation).
## Features arithmetic operations, stored derivatives, and backprop functionality.
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self,other), '+')
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self,other), '*')
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data / other.data, (self,other), '/')
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def relu(self):
        return Value(self.data if self.data>0 else 0.0,(self,), _op='relu')
    
    def tanh(self):
        return Value(math.tanh(self.data),(self,), _op='tanh')
    
    def backprop(self):
        if self._op=='+':
            for c in self._prev:
                c.grad += (self.grad*(-(len(self._prev))+3))
        elif self._op=='*':
            if len(self._prev)==1:
                for c in self._prev:
                    c.grad += (self.grad*2*c.data)
            else:
                for c in self._prev:
                    temp = self.grad
                    for d in self._prev:
                        if c!=d:
                            temp*=d.data
                    c.grad+=temp
        elif self._op=='relu':
            for c in self._prev:
                c.grad = (self.grad if c.data>0 else 0.0)
        elif self._op=='tanh':
            for c in self._prev:
                c.grad = self.grad*(1-(self.data**2))
        
    def start_backprop(self):
        vis = set()
        backprop_order = []
        #Note: topologicalSort mutates vis and fills backprop_order
        topologicalSort(self,vis,backprop_order)
        self.grad = 1.0
        for node in reversed(backprop_order):
            node.backprop()

class Neuron:

    def __init__(self, nin):
        rng = np.random.default_rng()
        self.w = [Value(rng.random()*2 - 1) for _ in range(nin)]
        self.b = Value(rng.random()*2 - 1)
    
    def __call__(self, x):
        return sum((wi*xi for wi,xi in zip(self.w,x)), self.b).tanh()
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]


n = MLP(3, [4,4,1])
xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]
ys = [1.0,-1.0,-1.0,1.0]

for i in range(1000):
    ## Forward step
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)*(yout - ygt) for ygt,yout in zip(ys,ypred))
    ## Backward step
    for p in n.parameters():
        p.grad = 0
    loss.start_backprop()
    for p in n.parameters():
        p.data += (-0.2 * p.grad)
    print(f"Epoch {i+1}\n------------------------")
    print(f"Expect: {ys}")
    print(f"Predic: {[round(yp.data,4) for yp in ypred]}\n")

## If you want to see the network!  Note: requires graphviz executables. 
#draw_dot(loss).render(view="True")
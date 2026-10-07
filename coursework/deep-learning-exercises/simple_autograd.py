import torch

class Value:
    """A tiny scalar wrapper that delegates all gradient work to PyTorch autograd."""

    def __init__(self, data, _tensor=None):
        # leaf node: create fresh tensor with grad; internal node: reuse tensor
        self._t = _tensor if _tensor is not None else torch.tensor(float(data), requires_grad=True)
        # make sure every Tensor (leaf or not) keeps its grad for printing
        self._t.retain_grad()
        self.data=data
        self.grad=0
        self.children=None


    # ------- conveniences -------
    @property
    def data(self):
        return self._t.item()

    @property
    def grad(self):
        g = self._t.grad
        return 0 if g is None else g.item()


    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)
        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    # ensure rhs is Value
    def _wrap(self, other):
        return other if isinstance(other, Value) else Value(other)

    # ------- arithmetic ops -------
    def __add__(self, other):
        # Implement addition. Coerce `other` to a Value via self._wrap,
        # then build a new Value whose underlying tensor is self._t + other._t.
        other=self._wrap(other)
        out=self._t+other._t
        result=Value(out,requires_grad=True)
        result.children=[self,other]
        return result
        pass

    __radd__ = __add__

    def __mul__(self, other):
        # Implement multiplication using self._t and other._t.
        pass

    __rmul__ = __mul__

    # ------- activation -------
    def relu(self):
        # Implement ReLU using torch.relu on self._t.
        pass

    # ------- back-prop entry -------
    def backward(self):
        # Trigger PyTorch autograd on the underlying tensor.
        pass
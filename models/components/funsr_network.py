import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder


class FUNSRNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        bias=0.5,
        scale=1,
        multires=0,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super().__init__()

        dims = [d_in] + [d_hidden] * n_layers + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            print("PE")
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    std = 0.0001
                    if not inside_outside:
                        nn.init.normal_(
                            lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=std
                        )
                        nn.init.constant_(lin.bias, -bias)
                    else:
                        nn.init.normal_(
                            lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=std
                        )
                        nn.init.constant_(lin.bias, bias)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        return x / self.scale

    def sdf(self, x):
        return self.forward(x)

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)

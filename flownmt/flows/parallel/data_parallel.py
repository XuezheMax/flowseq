from overrides import overrides
from typing import Tuple
import torch
from torch.nn.parallel.replicate import replicate
from flownmt.flows.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.data_parallel import _check_balance

from flownmt.flows.flow import Flow


class DataParallelFlow(Flow):
    """
    Implements data parallelism at the flow level.
    """
    def __init__(self, flow: Flow, device_ids=None, output_device=None, dim=0):
        super(DataParallelFlow, self).__init__(flow.inverse)

        if not torch.cuda.is_available():
            self.flow = flow
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.flow = flow
        self.device_ids = device_ids
        self.output_device = output_device

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.flow.cuda(device_ids[0])

    @overrides
    def forward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.device_ids:
            return self.flow.forward(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.flow.forward(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.flow, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    @overrides
    def backward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.device_ids:
            return self.flow.backward(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.flow.backward(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.flow, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs, backward=True)
        return self.gather(outputs, self.output_device)

    @overrides
    def init(self, *input, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.init(*input, **kwargs)

    def replicate(self, flow, device_ids):
        return replicate(flow, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs, backward=False):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)], backward=backward)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

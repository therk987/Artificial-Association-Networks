import torch
import torch.nn as nn

from aan.models.encoders.recursive_encoder import RecursiveAssociationNeuralNetworks
from aan.models.encoders.flat_recursive_encoder import FlatRecursiveAssociationNeuralNetworks
from aan.models.feature_encoders.multiencoder_connector import MultiExtractionConnector
from aan.models.feature_decoders.multidecoder_connector import MultiRestorationConnector
from aan.models.subtasks.multisubtask_connector import MultiSubTaskConnector
from aan.models.maintasks.multimaintask_connector import MultiMainTaskConnector


class ArtificialAssociationNeuralNetworks(nn.Module):
    """End-to-end AAN (paper, Section III-K).

    forward: build features per domain (Psi) -> DFC over the neurotree
    (RAN-series cell) -> node-level subtasks (Phi_s) -> main task on the
    root hidden state (Phi_m).
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            feature_encoders, feature_decoders, subtask_networks, maintask_networks,
            version='gaau',
            engine='flat',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # empty (domain-less) association nodes get a zero feature vector,
        # so the connector's zero-feature size must match the extractor output
        self.multi_feature_extraction_networks = MultiExtractionConnector(input_dim, feature_encoders)
        self.multi_restoration_networks = MultiRestorationConnector(input_dim, feature_decoders)
        self.multi_sub_task_networks = MultiSubTaskConnector(hidden_dim, subtask_networks)
        self.multi_main_task_networks = MultiMainTaskConnector(hidden_dim, maintask_networks)

        # The DFD (decoder) pass reads per-node state written by the
        # recursive engine, so decoder-equipped models use it; otherwise the
        # flat engine executes the same model with level-batched kernels.
        if engine == 'flat' and len(feature_decoders) > 0:
            engine = 'recursive'
        if engine == 'flat':
            encoder_cls = FlatRecursiveAssociationNeuralNetworks
        elif engine == 'recursive':
            encoder_cls = RecursiveAssociationNeuralNetworks
        else:
            raise ValueError("unknown engine: {} (expected 'flat' or 'recursive')".format(engine))
        self.engine = engine

        self.ran = encoder_cls(
            input_dim, hidden_dim,
            self.multi_feature_extraction_networks,
            self.multi_sub_task_networks,
            version=version,
        )
        if engine == 'recursive':
            # root dx/dh accumulation is only needed when a decoder is used
            self.ran.store_deconv_inputs = len(feature_decoders) > 0

    def forward(self, batchNeuroTree, tasks, node_level=False):
        h_root, batchNeuroTree = self.propagate(batchNeuroTree, node_level)
        main_task_outputs = self.multi_main_task_networks(h_root, batchNeuroTree, tasks)
        return main_task_outputs, h_root, batchNeuroTree

    def propagate(self, batchNeuroTree, node_level=False):
        outputs = self.ran(batchNeuroTree, node_level)
        if isinstance(outputs, torch.Tensor):  # flat engine returns (B, H)
            return outputs, batchNeuroTree
        return torch.stack(outputs, dim=0), batchNeuroTree

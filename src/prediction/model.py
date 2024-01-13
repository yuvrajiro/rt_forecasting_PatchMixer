from __future__ import annotations
import torch.nn.functional as F
from collections.abc import Callable
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.tft_submodels import get_embedding_size
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data import (
    MixedCovariatesSequentialDataset,
    MixedCovariatesTrainingDataset,
    TrainingDataset,
)
logger = get_logger(__name__)
MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
from torch import Tensor


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Backbone(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len= 16, stride = 8, padding_patch = 'end'):
        super(Backbone, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Patching
        self.patch_len = patch_len  # 16
        self.stride = stride   # 8
        self.patch_num = patch_num = int((seq_len - patch_len) / stride + 1)
        self.padding_patch = padding_patch
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num = patch_num = patch_num + 1

        # 1
        d_model = patch_len * patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.dropout_embed = nn.Dropout(0.3)

        # 2
        # self.lin_res = nn.Linear(seq_len, pred_len) # direct res, seems bad
        self.lin_res = nn.Linear(patch_num * d_model, pred_len)
        self.dropout_res = nn.Dropout(0.3)

        # 3.1
        self.depth_conv = nn.Conv1d(patch_num, patch_num, kernel_size=patch_len, stride=patch_len, groups=patch_num)
        self.depth_activation = nn.GELU()
        self.depth_norm = nn.BatchNorm1d(patch_num)
        self.depth_res = nn.Linear(d_model, patch_len)
        # 3.2
        # self.point_conv = nn.Conv1d(patch_len,patch_len,kernel_size=1, stride=1)
        # self.point_activation = nn.GELU()
        # self.point_norm = nn.BatchNorm1d(patch_len)
        self.point_conv = nn.Conv1d(patch_num, patch_num, kernel_size=1, stride=1)
        self.point_activation = nn.GELU()
        self.point_norm = nn.BatchNorm1d(patch_num)
        # 4
        self.mlp = Mlp(patch_len * patch_num, pred_len * 2, pred_len)

    def forward(self, x): # B, L, D -> B, H, D
        B, _, D = x.shape
        L = self.patch_num
        P = self.patch_len

        # z_res = self.lin_res(x.permute(0, 2, 1)) # B, L, D -> B, H, D
        # z_res = self.dropout_res(z_res)

        # 1
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(x.permute(0, 2, 1))  # B, L, D -> B, D, L -> B, D, L
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride) # B, D, L, P
        z = z.reshape(B * D, L, P, 1).squeeze(-1)
        z = self.embed(z) # B * D, L, P -> # B * D, L, d
        z = self.dropout_embed(z)

        # 2
        z_res = self.lin_res(z.reshape(B, D, -1)) # B * D, L, d -> B, D, L * d -> B, D, H
        z_res = self.dropout_res(z_res)

        # 3.1
        res = self.depth_res(z) # B * D, L, d -> B * D, L, P
        z_depth = self.depth_conv(z) # B * D, L, d -> B * D, L, P
        z_depth = self.depth_activation(z_depth)
        z_depth = self.depth_norm(z_depth)
        z_depth = z_depth + res
        # 3.2
        z_point = self.point_conv(z_depth) # B * D, L, P -> B * D, L, P
        z_point = self.point_activation(z_point)
        z_point = self.point_norm(z_point)
        z_point = z_point.reshape(B, D, -1) # B * D, L, P -> B, D, L * P

        # 4
        z_mlp = self.mlp(z_point) # B, D, L * P -> B, D, H

        return (z_res + z_mlp).permute(0,2,1)


class _PatchMixer(PLMixedCovariatesModule):
    def __init__(
            self,
            n_input_channels: int,
            n_extra_channels: int,
            output_dim: Tuple[int, int],
            variables_meta: Dict[str, Dict[str, List[str]]],
            num_static_components: int,
            add_relative_index: bool,
            patch_len: int = 16,
            stride: int = 8,
            padding_patch: str = 'end',
            **kwargs,
    ):

        """
        A metaclass for all PatchMixer models.

        Parameters
        ----------
        n_input_channels
            number of input channels
        n_extra_channels
            number of extra channels
        output_dim
            output dimension
        variables_meta
            variables meta
        num_static_components
            number of static components
        hidden_size
            hidden size
        num_block
            number of blocks
        hidden_continuous_size
            hidden continuous size
        dropout
            dropout rate
        add_relative_index
            whether to add relative index
        norm_type
            type of normalization
        kwargs
            all parameters required for :class:`darts.model.forecasting_models.TorchForecastingModel` base class.

        """

        super().__init__(**kwargs)

        self.n_targets, self.loss_size = output_dim
        self.variables_meta = variables_meta
        self.num_static_components = num_static_components
        self.add_relative_index = add_relative_index
        self.n_input_channels = n_input_channels
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.batch_size_last = -1
        self.relative_index = None
        self.static_channel_provided = num_static_components > 0
        self.seq_len = self.input_chunk_length
        self.pred_len = self.output_chunk_length
        self.fc_out = nn.Linear(self.n_input_channels + n_extra_channels, self.n_targets)
        self.batch_size_last = -1
        self.relative_index = None
        self.rev = RevIN(num_features=self.n_targets, subtract_last=True)
        self.backbone = Backbone(self.seq_len, self.pred_len, patch_len=self.patch_len, stride=self.stride,
                                 padding_patch=self.padding_patch)

    @property
    def reals(self) -> List[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta["model_config"]["reals_input"]

    @property
    def static_variables(self) -> List[str]:
        """
        List of all static variables in model
        """
        return self.variables_meta["model_config"]["static_input"]

    @property
    def numeric_static_variables(self) -> List[str]:
        """
        List of numeric static variables in model
        """
        return self.variables_meta["model_config"]["static_input_numeric"]

    @property
    def categorical_static_variables(self) -> List[str]:
        """
        List of categorical static variables in model
        """
        return self.variables_meta["model_config"]["static_input_categorical"]

    @property
    def encoder_variables(self) -> List[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_encoder_input"]

    @property
    def decoder_variables(self) -> List[str]:
        """
        List of all decoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_decoder_input"]

    @staticmethod
    def expand_static_context(context: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, time_steps, -1)

    @staticmethod
    def get_relative_index(
            encoder_length: int,
            decoder_length: int,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Returns scaled time index relative to prediction point.
        """
        index = torch.arange(
            encoder_length + decoder_length, dtype=dtype, device=device
        )
        prediction_index = encoder_length - 1
        index[:encoder_length] = index[:encoder_length] / prediction_index
        index[encoder_length:] = index[encoder_length:] / prediction_index
        return index.reshape(1, len(index), 1).repeat(batch_size, 1, 1)




    @io_processor
    def forward(
            self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """PatchMixer model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor
        """
        x_cont_past, x_cont_future, x_static = x_in
        dim_samples, dim_time, dim_variable = 0, 1, 2
        device = x_in[0].device




        x_cont_past_target = x_cont_past[:, :, : self.n_targets]
        x_cont_past_covariates = x_cont_past[:, :, self.n_targets:]
        x_cont_past_target = self.rev(x_cont_past_target, mode='norm')
        x_cont_past = torch.cat([x_cont_past_target, x_cont_past_covariates], dim=-1)

        batch_size = x_cont_past.shape[dim_samples]
        encoder_length = self.input_chunk_length
        decoder_length = self.output_chunk_length
        if batch_size != self.batch_size_last:
            if self.add_relative_index:
                self.relative_index = self.get_relative_index(
                    encoder_length=encoder_length,
                    decoder_length=decoder_length,
                    batch_size=batch_size,
                    device=device,
                    dtype=x_cont_past.dtype,
                )

            self.batch_size_last = batch_size

        if self.add_relative_index:
            x_cont_past = torch.cat(
                [
                    ts[:, :encoder_length, :]
                    for ts in [x_cont_past, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )
            x_cont_future = torch.cat(
                [
                    ts[:, -decoder_length:, :]
                    for ts in [x_cont_future, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )
        z = self.backbone(x_cont_past)  # B, L, D -> B, H, D
        z = self.fc_out(z)
        z = self.rev(z, 'denorm')  # B, L, D -> B, H, D
        return z.unsqueeze(-1)



class PatchMixer(MixedCovariatesTorchModel):
    def __init__(
            self,
            input_chunk_length: int,
            output_chunk_length: int,
            n_input_channels: int,
            n_extra_channels: int,
            add_relative_index: bool = False,
            use_static_covariates: bool = True,
            patch_len: int = 16,
            stride: int = 8,
            padding_patch: str = 'end',
            **kwargs,
    ):
        """

        PyTorch implementation of the New PatchMixer model from `this paper <https://arxiv.org/pdf/2303.06053.pdf>`_.
        The implemntation use Darts Forescasting Module, this class works as a plugin for Darts.

        Parameters
        ----------
        input_chunk_length
            length of the input chunk
        output_chunk_length
            length of the output chunk
        n_input_channels
            number of input channels
        n_extra_channels
            number of extra channels
        hidden_size
            hidden size of the model
        num_block
            number of blocks
        dropout
            dropout rate
        hidden_continuous_size
            hidden size for continuous variables
        add_relative_index
            whether to add relative index
        norm_type
            type of normalization
        use_static_covariates
            whether to use static covariates
        kwargs
            all parameters required for :class:`darts.model.forecasting_models.TorchForecastingModel` base class.

n_input_channels: int,
            n_extra_channels: int,
            output_dim: Tuple[int, int],
            variables_meta: Dict[str, Dict[str, List[str]]],
            num_static_components: int,
            add_relative_index: bool,
            patch_len: int = 16,
            stride: int = 8,
            padding_patch: str = 'end',

        """

        model_kwargs = {key: val for key, val in self.model_params.items()}

        model_kwargs["loss_fn"] = nn.MSELoss()
        super().__init__(**self._extract_torch_model_params(**model_kwargs))
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.n_input_channels = n_input_channels
        self.n_extra_channels = n_extra_channels
        self.add_relative_index = add_relative_index
        self.output_dim: Optional[Tuple[int, int]] = None
        self._considers_static_covariates = use_static_covariates
        categorical_embedding_sizes = None
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.categorical_embedding_sizes = (
            categorical_embedding_sizes
            if categorical_embedding_sizes is not None
            else {}
        )

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        `train_sample` contains the following tensors:
            (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates,
            future_target)

            each tensor has shape (n_timesteps, n_variables)
            - past/historic tensors have shape (input_chunk_length, n_variables)
            - future tensors have shape (output_chunk_length, n_variables)
            - static covariates have shape (component, static variable)

        Darts Interpretation of pytorch-forecasting's TimeSeriesDataSet:
            time_varying_knowns : future_covariates (including historic_future_covariates)
            time_varying_unknowns : past_targets, past_covariates

            time_varying_encoders : [past_targets, past_covariates, historic_future_covariates, future_covariates]
            time_varying_decoders : [historic_future_covariates, future_covariates]

        `variable_meta` is used in TFT to access specific variables
        """

        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            future_target,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length:]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )

        self.output_dim = (
            (future_target.shape[1], 1)
            if self.likelihood is None
            else (future_target.shape[1], self.likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
            "model_config": {},
        }

        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                            self.static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index(
                            [i for i in range(static_covariates.shape[1])]
                        )
                    else:
                        static_cols = self.static_covariates.columns
                    numeric_mask = ~static_cols.isin(self.categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                            zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = self.categorical_embedding_sizes[col_name]
                            raise_if_not(
                                isinstance(embedding, (int, tuple)),
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read "
                                "the PatchMixer documentation for more information.",
                                logger,
                            )
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )
        print(f"static covariates: {static_covariates}")
        n_static_components = (
            len(static_covariates[0]) if static_covariates is not None else 0
        )
        print(f"n_static_components: {n_static_components}")

        self.categorical_embedding_sizes = categorical_embedding_sizes

        return _PatchMixer(
            n_input_channels=self.n_input_channels,
            n_extra_channels=self.n_extra_channels,
            output_dim=self.output_dim,
            variables_meta=variables_meta,
            num_static_components=n_static_components,
            add_relative_index=self.add_relative_index,
            patch_len=self.patch_len,
            stride=self.stride,
            padding_patch=self.padding_patch,
            **self.pl_module_params,
        )

    def _build_train_dataset(
            self,
            target: Sequence[TimeSeries],
            past_covariates: Optional[Sequence[TimeSeries]],
            future_covariates: Optional[Sequence[TimeSeries]],
            max_samples_per_ts: Optional[int],
    ) -> MixedCovariatesSequentialDataset:

        raise_if(
            future_covariates is None and not self.add_relative_index,
            "PatchMixer requires future covariates. The model applies multi-head attention queries on future "
            "inputs. Consider specifying a future encoder with `add_encoders` or setting `add_relative_index` "
            "to `True` at model creation (read TFT model docs for more information). "
            "These will automatically generate `future_covariates` from indexes.",
            logger,
        )

        return MixedCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self.uses_static_covariates,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(
            isinstance(train_dataset, MixedCovariatesTrainingDataset),
            "PatchMixer requires a training dataset of type MixedCovariatesTrainingDataset.",
        )

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_static_covariates(self) -> bool:
        return False





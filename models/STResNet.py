import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class STResBranch(nn.Module):
    def __init__(self, in_channels, nb_filter, nb_flow, nb_residual_unit):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, nb_filter, kernel_size=3, padding=1)
        self.res_units = nn.Sequential(
            *[ResidualUnit2D(nb_filter) for _ in range(nb_residual_unit)]
        )
        self.conv_out = nn.Conv2d(nb_filter, nb_flow, kernel_size=3, padding=1)

    def forward(self, x):
        hidden = self.conv_in(x)
        hidden = self.res_units(hidden)
        return self.conv_out(hidden)


class ExternalFC(nn.Module):
    def __init__(self, external_dim, hidden_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(external_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, external_features):
        return self.network(external_features)


class STResNet(nn.Module):
    def __init__(
        self,
        H,
        W,
        external_dim,
        nb_flow=1,
        len_c=3,
        len_p=1,
        len_t=1,
        period_interval=None,
        trend_interval=None,
        nb_filter=64,
        nb_residual_unit=12,
        use_external=True,
        external_hidden=64,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.nb_flow = nb_flow
        self.len_c = len_c
        self.len_p = len_p
        self.len_t = len_t
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.use_external = use_external and external_dim > 0
        self.total_input_channels = (len_c + len_p + len_t) * nb_flow

        if nb_flow != 1:
            raise ValueError("This implementation currently supports nb_flow=1 only.")

        self.branch_c = self._build_branch(len_c, nb_filter, nb_flow, nb_residual_unit)
        self.branch_p = self._build_branch(len_p, nb_filter, nb_flow, nb_residual_unit)
        self.branch_t = self._build_branch(len_t, nb_filter, nb_flow, nb_residual_unit)

        if self.branch_c is not None:
            self.Wc = nn.Parameter(torch.ones(nb_flow, H, W))
        if self.branch_p is not None:
            self.Wp = nn.Parameter(torch.ones(nb_flow, H, W))
        if self.branch_t is not None:
            self.Wt = nn.Parameter(torch.ones(nb_flow, H, W))

        if self.use_external:
            self.external = ExternalFC(
                external_dim=external_dim,
                hidden_dim=external_hidden,
                out_dim=nb_flow * H * W,
            )

    @staticmethod
    def _build_branch(length, nb_filter, nb_flow, nb_residual_unit):
        if length <= 0:
            return None
        return STResBranch(
            in_channels=length * nb_flow,
            nb_filter=nb_filter,
            nb_flow=nb_flow,
            nb_residual_unit=nb_residual_unit,
        )

    def forward(self, demands_series, weather, time, sample_idx=None):
        demands_series = demands_series.to(torch.float32)
        batch_size, channels, height, width = demands_series.shape

        if channels != self.total_input_channels:
            raise ValueError(
                f"Expected {self.total_input_channels} input channels, got {channels}."
            )
        if height != self.H or width != self.W:
            raise ValueError(
                f"Expected spatial size {(self.H, self.W)}, got {(height, width)}."
            )

        outputs = []
        offset = 0

        if self.branch_c is not None:
            closeness = demands_series[:, offset:offset + self.len_c]
            outputs.append(self.Wc.unsqueeze(0) * self.branch_c(closeness))
            offset += self.len_c

        if self.branch_p is not None:
            period = demands_series[:, offset:offset + self.len_p]
            outputs.append(self.Wp.unsqueeze(0) * self.branch_p(period))
            offset += self.len_p

        if self.branch_t is not None:
            trend = demands_series[:, offset:offset + self.len_t]
            outputs.append(self.Wt.unsqueeze(0) * self.branch_t(trend))

        prediction_map = torch.stack(outputs, dim=0).sum(dim=0)

        if self.use_external:
            external_features = torch.cat(
                [weather.to(torch.float32), time.to(torch.float32)],
                dim=1,
            )
            external_map = self.external(external_features).view(
                batch_size,
                self.nb_flow,
                self.H,
                self.W,
            )
            prediction_map = prediction_map + external_map

        return prediction_map.view(batch_size, -1)

import time
from gan.utils.data import *
from model import FactorMoE
import numpy as np
import random
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler
import os
from gan.utils import (
    load_pickle, get_blds_list_df)
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret
from gan.utils.builder import exprs2tensor
from alphagen.data.tree import ExpressionBuilder

from config_FactorMoE import FACTOR_MOE_CONFIG


def get_feat_sign(feat, names):
    to_add = []
    for i, name in enumerate(names):
        if name.split('_')[-1] == 'mean':
            to_add.append(feat[:, :, i:i + 1].sign())
    return torch.cat(to_add, dim=-1)


def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list = []
    cur_fct = 0
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i + chunk_size], y[i:i + chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list


def get_tensor_metrics(x, y):
    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)

    # ric_s = ic_s
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0)
    ric_s = torch.nan_to_num(ric_s, nan=0)
    ret_s = torch.nan_to_num(ret_s, nan=0)

    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item()
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item()
    ret_s_mean = ret_s.mean().item()
    ret_s_std = ret_s.std().item()

    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,

    )
    return result


def get_tensor_metrics_raw(x, y):
    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)

    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0)
    ric_s = torch.nan_to_num(ric_s, nan=0)
    ret_s = torch.nan_to_num(ret_s, nan=0)
    torch.cuda.empty_cache()

    return ic_s, ric_s, ret_s


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def align_fct_and_tgt_by_max_valid_stocks(fct_tensor: torch.Tensor, tgt_tensor: torch.Tensor):
    """
    input:
        fct_tensor: [T, N, F]
        tgt_tensor: [T, N, 1] or [T, N]
    output:
        aligned_fct: [T, max_n, F]
        aligned_tgt: [T, max_n, 1]
    """
    T, N, F = fct_tensor.shape
    if tgt_tensor.ndim == 2:
        tgt_tensor = tgt_tensor.unsqueeze(-1)
    valid_mask = torch.isfinite(tgt_tensor.squeeze(-1))  # [T, N]
    valid_counts = valid_mask.sum(dim=1)     # [T]
    max_n = valid_counts.max().item()
    aligned_fct = torch.zeros((T, max_n, F), device=fct_tensor.device, dtype=fct_tensor.dtype)
    aligned_tgt = torch.full((T, max_n, 1), float('nan'), device=tgt_tensor.device, dtype=tgt_tensor.dtype)
    for t in range(T):
        idx = valid_mask[t].nonzero(as_tuple=True)[0]
        n_valid = len(idx)
        n_fill = min(n_valid, max_n)
        if n_fill > 0:
            aligned_fct[t, :n_fill] = fct_tensor[t, idx[:n_fill]]
            aligned_tgt[t, :n_fill, 0] = tgt_tensor[t, idx[:n_fill], 0]
    return aligned_fct, aligned_tgt

def sliding_window_batch(input_tensor, target_tensor=None, windows=60, batch_size=8, device=None):
    """
    input_tensor: [T, N, F+M]
    target_tensor: [T, N] or [T, N, 1]
    return: (batch_x, batch_y) 或 batch_x
    """
    num_days = input_tensor.shape[0]
    indices = list(range(0, num_days - windows + 1))
    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start:batch_start + batch_size]
        batch_x = torch.stack([input_tensor[i:i + windows] for i in batch_idx], dim=0)  # [B, windows, N, F+M]
        if device is not None:
            batch_x = batch_x.to(device)
        if target_tensor is not None:
            batch_y = torch.stack([target_tensor[i + windows - 1].squeeze(-1) for i in batch_idx], dim=0)  # [B, N]
            if device is not None:
                batch_y = batch_y.to(device)
            if torch.isfinite(batch_y).sum() == 0:
                continue
            yield batch_x, batch_y
        else:
            yield batch_x


def sliding_window_metrics(ic_s, ric_s, ret_s, windows, shift=20):
    """
    ic_s, ric_s, ret_s: [T, F]
    : [num_batch, F, 9]
    """
    # [num_batch, windows-shift, F]
    ic_win = ic_s.unfold(0, windows - shift, 1)
    ic_windows = ic_s.unfold(0, windows - shift, 1).permute(0, 2, 1)
    ric_windows = ric_s.unfold(0, windows - shift, 1).permute(0, 2, 1)
    ret_windows = ret_s.unfold(0, windows - shift, 1).permute(0, 2, 1)

    # [num_batch, F]
    ic_mean = ic_windows.mean(dim=1)
    ic_std = ic_windows.std(dim=1)
    icir = torch.where(ic_std > 0, ic_mean / ic_std, torch.zeros_like(ic_std))

    ric_mean = ric_windows.mean(dim=1)
    ric_std = ric_windows.std(dim=1)
    ricir = torch.where(ric_std > 0, ric_mean / ric_std, torch.zeros_like(ric_std))

    ret_mean = ret_windows.mean(dim=1)
    ret_std = ret_windows.std(dim=1)
    retir = torch.where(ret_std > 0, ret_mean / ret_std, torch.zeros_like(ret_std))
    # [num_batch, F, 9]
    metrics = torch.stack([
        ic_mean, ic_std, icir,
        ric_mean, ric_std, ricir,
        ret_mean, ret_std, retir
    ], dim=-1)
    return metrics


def calc_market_regime_indicators(data_all, market_tensor_temp, mb, mf, n=5):
    """
    Calculate daily market regime indicators
    """
    feature_names = ['open', 'close', 'high', 'low', 'volume', 'vwap']
    feature_indices = {}
    for i in range(data_all.data.shape[1]):
        if hasattr(data_all, '_features'):
            f = data_all._features[i]
            if hasattr(f, 'name'):
                name = f.name.lower()
            else:
                name = str(f).lower()
            if name in feature_names:
                feature_indices[name] = i
        else:
            feature_indices = {k: v for v, k in enumerate(feature_names)}
            break

    open_idx = feature_indices.get('open', 0)
    close_idx = feature_indices.get('close', 1)
    high_idx = feature_indices.get('high', 2)
    low_idx = feature_indices.get('low', 3)
    volume_idx = feature_indices.get('volume', 4)
    vwap_idx = feature_indices.get('vwap', 5)

    open_prices_full = data_all.data[:, open_idx, :]  # [T, N]
    close_prices_full = data_all.data[:, close_idx, :]  # [T, N]
    high_prices_full = data_all.data[:, high_idx, :]  # [T, N]
    low_prices_full = data_all.data[:, low_idx, :]  # [T, N]
    volume_prices_full = data_all.data[:, volume_idx, :]  # [T, N]
    vwap_prices_full = data_all.data[:, vwap_idx, :]  # [T, N]

    price_today = close_prices_full[1:]  # [T-1, N]
    price_yesterday = close_prices_full[:-1]  # [T-1, N]
    valid_mask = torch.isfinite(price_today) & torch.isfinite(price_yesterday)
    price_diff = torch.zeros_like(price_today)
    price_diff[valid_mask] = price_today[valid_mask] - price_yesterday[valid_mask]
    up_count = (price_diff > 0).float().sum(dim=1)  # [T-1]
    down_count = (price_diff < 0).float().sum(dim=1)  # [T-1]
    total_count = valid_mask.sum(dim=1).float()  # [T-1]
    up_ratio = up_count / total_count
    down_ratio = down_count / total_count
    net_up = up_count - down_count
    up_down_ratio = up_count / down_count
    up_count = torch.log1p(up_count)
    down_count = torch.log1p(down_count)
    net_up = torch.sign(net_up) * torch.log1p(net_up.abs())

    close_up_open = (close_prices_full > open_prices_full).float().sum(dim=1)  # [T]
    close_down_open = (close_prices_full < open_prices_full).float().sum(dim=1)  # [T]
    close_up_open_ratio = close_up_open / (torch.isfinite(close_prices_full).sum(dim=1))
    close_down_open_ratio = close_down_open / (torch.isfinite(close_prices_full).sum(dim=1))
    close_up_open = torch.log1p(close_up_open)
    close_down_open = torch.log1p(close_down_open)
    # VWAP
    close_up_vwap = (close_prices_full > vwap_prices_full).float().sum(dim=1)  # [T]
    close_down_vwap = (close_prices_full < vwap_prices_full).float().sum(dim=1)  # [T]
    close_up_vwap_ratio = close_up_vwap / (torch.isfinite(close_prices_full).sum(dim=1))
    close_down_vwap_ratio = close_down_vwap / (torch.isfinite(close_prices_full).sum(dim=1))
    mean_close_vwap_rel = ((close_prices_full - vwap_prices_full) / vwap_prices_full).nanmean(dim=1)  # [T]
    close_up_vwap = torch.log1p(close_up_vwap)
    close_down_vwap = torch.log1p(close_down_vwap)

    # volume_prices_full: [T, N]
    T, N = volume_prices_full.shape
    vol_ma = torch.full_like(volume_prices_full, float('nan'))  # [T, N]
    for t in range(n, T):
        prev_vol = volume_prices_full[t - n:t]  # [n, N]
        valid_mask = torch.isfinite(prev_vol)  # [n, N]
        valid_sum = torch.where(valid_mask, prev_vol,
                                torch.tensor(0., device=volume_prices_full.device, dtype=volume_prices_full.dtype)).sum(
            dim=0)
        valid_count = valid_mask.sum(dim=0)
        mean_vol = torch.where(valid_count > 0, valid_sum / valid_count.clamp_min(1),
                               torch.tensor(float('nan'), device=volume_prices_full.device))
        vol_ma[t] = mean_vol
    # [T, N]
    vol_valid_mask = torch.isfinite(volume_prices_full) & torch.isfinite(vol_ma)
    vol_up = (volume_prices_full > vol_ma) & vol_valid_mask
    vol_down = (volume_prices_full < vol_ma) & vol_valid_mask
    vol_up_count = vol_up.float().sum(dim=1)  # [T]
    vol_down_count = vol_down.float().sum(dim=1)  # [T]
    vol_up_ratio = vol_up_count / (vol_valid_mask.sum(dim=1) + 1e-6)
    vol_down_ratio = vol_down_count / (vol_valid_mask.sum(dim=1) + 1e-6)
    vol_up_down_ratio = vol_up_count / (vol_down_count + 1e-6)

    start = mb
    end = -mf if mf > 0 else None
    market_tensor = market_tensor_temp[start:end]
    up_count = up_count[start - 1:end]
    down_count = down_count[start - 1:end]
    up_ratio = up_ratio[start - 1:end]
    down_ratio = down_ratio[start - 1:end]
    net_up = net_up[start - 1:end]
    up_down_ratio = up_down_ratio[start - 1:end]
    close_up_open = close_up_open[start:end]
    close_down_open = close_down_open[start:end]
    close_up_open_ratio = close_up_open_ratio[start:end]
    close_down_open_ratio = close_down_open_ratio[start:end]
    close_up_vwap = close_up_vwap[start:end]
    close_down_vwap = close_down_vwap[start:end]
    close_up_vwap_ratio = close_up_vwap_ratio[start:end]
    close_down_vwap_ratio = close_down_vwap_ratio[start:end]
    mean_close_vwap_rel = mean_close_vwap_rel[start:end]
    vol_up_ratio = vol_up_ratio[start:end]
    vol_down_ratio = vol_down_ratio[start:end]
    vol_up_down_ratio = vol_up_down_ratio[start:end]

    market_tensor = torch.cat([
        market_tensor,
        # 54 + 18
        up_count.unsqueeze(-1),
        down_count.unsqueeze(-1),
        up_ratio.unsqueeze(-1),
        down_ratio.unsqueeze(-1),
        net_up.unsqueeze(-1),
        up_down_ratio.unsqueeze(-1),
        close_up_open.unsqueeze(-1),
        close_down_open.unsqueeze(-1),
        close_up_open_ratio.unsqueeze(-1),
        close_down_open_ratio.unsqueeze(-1),
        close_up_vwap.unsqueeze(-1),
        close_down_vwap.unsqueeze(-1),
        close_up_vwap_ratio.unsqueeze(-1),
        close_down_vwap_ratio.unsqueeze(-1),
        mean_close_vwap_rel.unsqueeze(-1),
        vol_up_ratio.unsqueeze(-1),
        vol_down_ratio.unsqueeze(-1),
        vol_up_down_ratio.unsqueeze(-1),
    ], dim=-1)
    return market_tensor


def calc_market_feature(market_feature):
    """
    input: market_feature [T, M, N]
    output: [T, M]
    """
    market_feature = torch.where(torch.isinf(market_feature), torch.zeros_like(market_feature), market_feature)
    # [T, M, N]
    mask = ~torch.isnan(market_feature)
    masked = torch.where(mask, market_feature,
                         torch.tensor(0., device=market_feature.device, dtype=market_feature.dtype))
    count = mask.sum(dim=2, keepdim=True).clamp_min(1)
    means = masked.sum(dim=2, keepdim=True) / count  # [T, M, 1]
    # std
    sq_diff = torch.where(mask, (market_feature - means) ** 2,
                          torch.tensor(0., device=market_feature.device, dtype=market_feature.dtype))
    stds = (sq_diff.sum(dim=2, keepdim=True) / count).sqrt()
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)
    zscore = (market_feature - means) / stds  # [T, M, N]
    # nan mean
    zscore = torch.where(mask, zscore, torch.tensor(0., device=market_feature.device, dtype=market_feature.dtype))
    zscore_sum = zscore.sum(dim=2)
    zscore_count = mask.sum(dim=2).clamp_min(1)
    market_tensor_temp = zscore_sum / zscore_count  # [T, M]
    return market_tensor_temp


def main(
        train_end_year: int = 2022,
        train_start_year: int = 2012,
        freq: str = 'day',
        target_f: str = 'close',
        seeds: str = '[0,1,2,3,4]',
        cuda: int = 0,
        save_name: str = 'test',
        patience_num: int = 4,
        shift: int = 20,
        market_dropout: float = 0.5,
        metric_dropout: float = 0.5,
        temperature: float = 3.0,
        n_factors: int = 10,
        instruments: str = "csi300",
):

    config = FACTOR_MOE_CONFIG.get(instruments, {}).get(n_factors)
    if config is None:
        return
    for key, value in config.items():
        locals()[key] = value
    market_hidden_dim = config.get('market_hidden_dim', 128)
    market_num_experts = config.get('market_num_experts', 8)
    market_chain_depth = config.get('market_chain_depth', 3)
    market_num_heads = config.get('market_num_heads', 4)
    metric_chain_depth = config.get('metric_chain_depth', 2)
    metric_num_heads = config.get('metric_num_heads', 3)
    metric_hidden_dim = config.get('metric_hidden_dim', 32)
    metric_num_experts = config.get('metric_num_experts', 2)
    market_bete = config.get('market_bete', 0.5)
    metric_bete = config.get('metric_bete', 2)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 10)
    windows = config.get('windows', 40)
    k = config.get('k', 8)

    if isinstance(seeds, str):
        seeds = eval(seeds)
    assert isinstance(seeds, list)

    train_end = train_end_year

    # read data
    returned = get_data_by_year(
        train_start=train_start_year, train_end=train_end, valid_year=train_end + 1, test_year=train_end + 2,
        instruments=instruments, target=target, freq=freq,
    )
    data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = returned


    returned2 = get_csi_data_by_year(
        train_start=train_start_year, train_end=train_end, valid_year=train_end + 1, test_year=train_end + 2,
        instruments=instruments, target=target, freq=freq,
    )
    data_all_m, data_m, data_valid_m, data_valid_withhead_m, data_test_m, data_test_withhead_m, _ = returned2

    mb = data_all.max_backtrack_days
    mf = data_all.max_future_days
    market_tensor_temp = calc_market_feature(data_all_m.data)
    market_tensor = calc_market_regime_indicators(data_all, market_tensor_temp, mb, mf)
    print(market_tensor.shape)

    all_valid_ic, all_valid_ric, all_test_ic, all_test_ric = [], [], [], []
    all_valid_ar, all_valid_ir, all_test_ar, all_test_ir = [], [], [], []

    for seed in seeds:
        set_seed(seed)

        path = f"out/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final_{train_start_year}_{n_factors}{target_f}.pkl"
        zoo = load_pickle(path)
        df = get_blds_list_df([zoo]).sort_values('score', ascending=False, key=lambda x: abs(x))
        df = df.iloc[:n_factors]
        factor_exprs = df['exprs_str'].tolist()

        fct_tensor = exprs2tensor(df['exprs'], data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)
        aligned_fct, aligned_tgt = align_fct_and_tgt_by_max_valid_stocks(fct_tensor, tgt_tensor)
        fct_tensor = aligned_fct
        tgt_tensor = aligned_tgt

        ic_list = []
        ric_list = []
        ret_list = []
        from tqdm import tqdm
        for cur in tqdm(range(fct_tensor.shape[-1])):
            ic_s, ric_s, ret_s = get_tensor_metrics_raw(fct_tensor[..., cur], tgt_tensor[..., 0])
            ic_list.append(ic_s)
            ric_list.append(ric_s)
            ret_list.append(ret_s)
        ic_s = torch.stack(ic_list, dim=-1)
        ric_s = torch.stack(ric_list, dim=-1)
        ret_s = torch.stack(ret_list, dim=-1)

        T, N, F = fct_tensor.shape
        market_tensor_expand = market_tensor.unsqueeze(1).expand(-1, N, -1)
        input_tensor = torch.cat([fct_tensor, market_tensor_expand], dim=-1)

        num_train = data_valid_withhead.n_days + data_test_withhead.n_days
        num_valid = data_valid.n_days
        num_test = data_test.n_days
        T = input_tensor.shape[0]
        train_tensor = input_tensor[:T - num_valid - num_test]
        valid_tensor = input_tensor[T - num_valid - num_test:T - num_test]
        test_tensor = input_tensor[T - num_test:]
        train_tgt = tgt_tensor[:T - num_valid - num_test]
        valid_tgt = tgt_tensor[T - num_valid - num_test:T - num_test]
        test_tgt = tgt_tensor[T - num_test:]
        train_ic = ic_s[:T - num_valid - num_test]
        valid_ic = ic_s[T - num_valid - num_test:T - num_test]
        test_ic = ic_s[T - num_test:]
        train_ric_s = ric_s[:T - num_valid - num_test]
        valid_ric_s = ric_s[T - num_valid - num_test:T - num_test]
        test_ric_s = ric_s[T - num_test:]
        train_ret_s = ret_s[:T - num_valid - num_test]
        valid_ret_s = ret_s[T - num_valid - num_test:T - num_test]
        test_ret_s = ret_s[T - num_test:]
        train_metrics = sliding_window_metrics(train_ic, train_ric_s, train_ret_s, windows=windows, shift=shift)[
                        :-shift]
        valid_metrics = sliding_window_metrics(valid_ic, valid_ric_s, valid_ret_s, windows=windows, shift=shift)[
                        :-shift]
        test_metrics = sliding_window_metrics(test_ic, test_ric_s, test_ret_s, windows=windows, shift=shift)[:-shift]

        train_tensor = torch.where(torch.isinf(train_tensor), torch.zeros_like(train_tensor), train_tensor)
        valid_tensor = torch.where(torch.isinf(valid_tensor), torch.zeros_like(valid_tensor), valid_tensor)
        test_tensor = torch.where(torch.isinf(test_tensor), torch.zeros_like(test_tensor), test_tensor)

        d_feat = fct_tensor.shape[-1]
        market_dim = market_tensor.shape[-1]
        metric_dim = train_metrics.shape[-1]
        gate_input_start_index = d_feat
        gate_input_end_index = d_feat + market_dim

        # --- train ---
        model = FactorMoE(
            market_dim=market_dim,
            metric_dim=metric_dim,
            factor_dim=d_feat,
            market_hidden_dim=market_hidden_dim,
            metric_hidden_dim=metric_hidden_dim,
            market_num_experts=market_num_experts,
            metric_num_experts=metric_num_experts,
            market_chain_depth=market_chain_depth,
            market_num_heads=market_num_heads,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            market_dropout=market_dropout,
            metric_dropout=metric_dropout,
            market_bete=market_bete,
            metric_bete=metric_bete,
            metric_chain_depth=metric_chain_depth,
            metric_num_heads=metric_num_heads,
            k=k,
            temperature=temperature,
        ).cuda()

        num_train_days = train_tensor.shape[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=2e-3,
            steps_per_epoch=(num_train_days - windows + 1) + 1,
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        loss_fn = torch.nn.MSELoss()
        scaler = GradScaler('cuda')
        model.train()

        best_val_loss = float('inf')
        patience = patience_num
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0
            num_samples = 0
            model.train()
            for i, (batch_x, batch_y) in enumerate(
                    sliding_window_batch(train_tensor, train_tgt, windows=windows, batch_size=batch_size,
                                         device='cuda')):
                batch_metrics = train_metrics[i:i + batch_x.shape[0]].to(batch_x.device)
                mask = torch.isfinite(batch_y)
                optimizer.zero_grad()
                pred = model(batch_x, batch_metrics)
                loss = loss_fn(pred[mask], batch_y[mask])
                if loss == 0:
                    continue
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += loss.item()
                num_samples += 1
            avg_loss = epoch_loss / max(num_samples, 1)

            # --- eval ---
            model.eval()
            val_loss = 0
            val_samples = 0
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(
                        sliding_window_batch(valid_tensor, valid_tgt, windows=windows, batch_size=batch_size,
                                             device='cuda')):
                    batch_metrics = valid_metrics[i:i + batch_x.shape[0]].to(batch_x.device)
                    mask = torch.isfinite(batch_y)
                    pred = model(batch_x, batch_metrics)
                    loss = loss_fn(pred[mask], batch_y[mask])
                    val_loss += loss.item()
                    val_samples += 1
            avg_val_loss = val_loss / max(val_samples, 1)

            torch.cuda.empty_cache()
            epoch_time = (time.time() - start_time) / 60
            print(
                f"[Seed {seed}] Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f} min")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Seed {seed}] Early stopping at epoch {epoch + 1}")
                    break

        # save
        model_path = f"out/{save_name}_{instruments}_{train_end}_{seed}/FactorMoE_{n_factors}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)

        # --------- test ---------
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            valid_preds = []
            for i, batch_x in enumerate(
                    sliding_window_batch(valid_tensor, windows=windows, batch_size=batch_size, device='cuda')):
                batch_metrics = valid_metrics[i:i + batch_x.shape[0]].to(batch_x.device)
                pred = model(batch_x, batch_metrics)
                valid_preds.append(pred.cpu())
            pred_valid = torch.cat(valid_preds, dim=0)
            valid_ic = batch_pearsonr(pred_valid, valid_tgt[windows - 1:].cpu().squeeze(-1))
            valid_ric = batch_spearmanr(pred_valid, valid_tgt[windows - 1:].cpu().squeeze(-1))

            test_preds = []
            test_gate_info = []
            for i, batch_x in enumerate(
                    sliding_window_batch(test_tensor, windows=windows, batch_size=batch_size, device='cuda')):
                batch_metrics = test_metrics[i:i + batch_x.shape[0]].to(batch_x.device)
                pred = model(batch_x, batch_metrics)
                test_preds.append(pred.cpu())
            pred_test = torch.cat(test_preds, dim=0)
            test_ic = batch_pearsonr(pred_test, test_tgt[windows - 1:].cpu().squeeze(-1))
            test_ric = batch_spearmanr(pred_test, test_tgt[windows - 1:].cpu().squeeze(-1))


        all_valid_ic.append(valid_ic)
        all_valid_ric.append(valid_ric)
        all_test_ic.append(test_ic)
        all_test_ric.append(test_ric)


        print(f"[Seed {seed}] Valid IC mean: {valid_ic.mean().item():.4f}, std: {valid_ic.std().item():.4f}")
        print(f"[Seed {seed}] Valid RIC mean: {valid_ric.mean().item():.4f}, std: {valid_ric.std().item():.4f}")
        print(f"[Seed {seed}] Test IC mean: {test_ic.mean().item():.4f}, std: {test_ic.std().item():.4f}")
        print(f"[Seed {seed}] Test RIC mean: {test_ric.mean().item():.4f}, std: {test_ric.std().item():.4f}")
        print(f"[Seed {seed}] Valid ICIR: {(valid_ic.mean() / valid_ic.std()).item():.4f}")
        print(f"[Seed {seed}] Valid RICIR: {(valid_ric.mean() / valid_ric.std()).item():.4f}")
        print(f"[Seed {seed}] Test ICIR: {(test_ic.mean() / test_ic.std()).item():.4f}")
        print(f"[Seed {seed}] Test RICIR: {(test_ric.mean() / test_ric.std()).item():.4f}")


        torch.cuda.empty_cache()

    all_valid_ic = torch.stack(all_valid_ic)
    all_valid_ric = torch.stack(all_valid_ric)
    all_test_ic = torch.stack(all_test_ic)
    all_test_ric = torch.stack(all_test_ric)

    print(f"\n[Average over {len(seeds)} seeds]")
    print(f"Valid IC mean: {all_valid_ic.mean().item():.4f}, std: {all_valid_ic.std().item():.4f}")
    print(f"Valid RIC mean: {all_valid_ric.mean().item():.4f}, std: {all_valid_ric.std().item():.4f}")
    print(f"Test IC mean: {all_test_ic.mean().item():.4f}, std: {all_test_ic.std().item():.4f}")
    print(f"Test RIC mean: {all_test_ric.mean().item():.4f}, std: {all_test_ric.std().item():.4f}")
    print(f"Valid ICIR mean: {(all_valid_ic.mean() / all_valid_ic.std()).item():.4f}")
    print(f"Valid RICIR mean: {(all_valid_ric.mean() / all_valid_ric.std()).item():.4f}")
    print(f"Test ICIR mean: {(all_test_ic.mean() / all_test_ic.std()).item():.4f}")
    print(f"Test RICIR mean: {(all_test_ric.mean() / all_test_ric.std()).item():.4f}")


if __name__ == '__main__':
    import fire

    fire.Fire(main)

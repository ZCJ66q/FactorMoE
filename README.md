# FactorMoE

## 📦Data Preparation
Refer to [AlphaForge](https://github.com/dulyhao/alphaforge) and [AlphaGen](https://github.com/RL-MLDM/alphagen) to first obtain raw data using [Qlib](https://github.com/microsoft/qlib).

Modify `qlib_base_data_path` in `data_collection/fetch_baostock_data.py`.  

```shell
python fetch_baostock_data.py
```

## ⛏️Alpha Mining
Also modify `QLIB_PATH` in the methods `get_data_by_year` and `get_csi_data_by_year` in `gan/utils/data.py`.

An example of alpha mining using AlphaForge is provided below. AlphaGen and [AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent) can be used similarly; examples are omitted here.

```shell
python Alpha_mining.py --instruments=csi300 --train_start=2012 --train_end_year=2022 --seeds=[0,1,2,3,4] --save_name=test --zoo_size=10
```

## 🚀Run Our Model

```shell
python train_FactorMoE.py
```

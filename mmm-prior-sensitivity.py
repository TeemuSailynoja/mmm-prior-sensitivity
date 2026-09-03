# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "arviz==1.3.0",
#     "arviz-stats==1.3.1",
#     "marimo>=0.24.0",
#     "matplotlib==3.11.1",
#     "numpy==2.4.6",
#     "nutpie==0.16.11",
#     "pandas==3.0.5",
#     "pymc==6.2.0",
#     "pymc-extras==0.14.0",
#     "pymc-marketing==1.1.0",
#     "xarray==2026.7.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import arviz as az
    from arviz_stats.psense import power_scale_dataset
    from pymc_marketing.mmm import MMM
    from pymc_marketing.mmm.components.saturation import LogisticSaturation
    from pymc_marketing.mmm.components.adstock import GeometricAdstock
    from pymc_extras.prior import Prior
    import xarray as xr
    import pandas as pd
    import numpy as np
    import pymc as pm

    # Set random seed for reproducibility
    seed = sum(map(ord, "mmm"))
    rng = np.random.default_rng(seed=seed)


@app.cell(hide_code=True)
def _():
    # Original data from https://raw.githubusercontent.com/sibylhe/mmm_stan/main/data.csv
    raw_df = pd.read_csv(
        "https://raw.githubusercontent.com/sibylhe/mmm_stan/main/data.csv"
    )

    # 1. control variables
    # We just keep the holidays columns
    control_columns = [col for col in raw_df.columns if "hldy_" in col]

    # 2. media variables
    channel_columns_raw = sorted(
        [
            col
            for col in raw_df.columns
            if "mdsp_" in col
            and col != "mdsp_viddig"
            and col != "mdsp_auddig"
            and col != "mdsp_sem"
        ]
    )

    channel_mapping = {
        "mdsp_dm": "Direct Mail",
        "mdsp_inst": "Insert",
        "mdsp_nsp": "Newspaper",
        "mdsp_audtr": "Radio",
        "mdsp_vidtr": "TV",
        "mdsp_so": "Social Media",
        "mdsp_on": "Online Display",
    }

    channel_columns = sorted(list(channel_mapping.values()))

    # 3. sales variables
    sales_col = "sales"

    data_df = raw_df[
        ["wk_strt_dt", sales_col, *channel_columns_raw, *control_columns]
    ]
    data_df = data_df.rename(columns=channel_mapping)

    # 4. Date column
    data_df["wk_strt_dt"] = pd.to_datetime(data_df["wk_strt_dt"])
    date_column = "wk_strt_dt"

    # 5. Target variable
    target_column = "sales"

    # 6. train test split
    train_test_split_date = pd.to_datetime("2018-02-01")

    train_mask = data_df.wk_strt_dt <= train_test_split_date
    test_mask = data_df.wk_strt_dt > train_test_split_date

    train_df = data_df[train_mask]
    test_df = data_df[test_mask]

    X_train = train_df.drop(columns=sales_col)
    X_test = test_df.drop(columns=sales_col)

    y_train = train_df[sales_col]
    y_test = test_df[sales_col]
    return (
        X_train,
        channel_columns,
        control_columns,
        data_df,
        date_column,
        sales_col,
        target_column,
        y_train,
    )


@app.cell(hide_code=True)
def _(channel_columns, data_df, sales_col):
    data_df.set_index("wk_strt_dt")[[sales_col] + channel_columns].plot.line(
        subplots=True, figsize=(12, 6)
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(control_columns, data_df):
    data_df.set_index("wk_strt_dt")[control_columns].plot.line(
        subplots=True, figsize=(12, 6)
    )
    plt.gca()
    return


@app.cell
def _(
    X_train,
    channel_columns,
    control_columns,
    date_column,
    target_column,
    y_train,
):
    model_config = {
        "intercept": Prior("Normal", mu=0.2, sigma=0.05),
        "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        "likelihood": Prior(
            "TruncatedNormal", lower=0, sigma=Prior("HalfNormal", sigma=1)
        ),
    }

    sampler_config = {"progressbar": True}

    mmm = MMM(
        model_config=model_config,
        sampler_config=sampler_config,
        target_column=target_column,
        date_column=date_column,
        adstock=GeometricAdstock(l_max=6),
        saturation=LogisticSaturation(),
        channel_columns=channel_columns,
        control_columns=control_columns,
        yearly_seasonality=5,
    )

    mmm.build_model(X_train, y_train)

    mmm.add_original_scale_contribution_variable(
        [
            "y",
            "intercept_contribution",
            "control_contribution",
            "channel_contribution",
            "fourier_contribution",
            "yearly_seasonality_contribution",
        ]
    )

    _ = mmm.fit(
        X=X_train,
        y=y_train,
        chains=4,
        tune=500,
        target_accept=0.85,
        random_seed=rng,
        nuts_sampler="nutpie",
    )
    return mmm, model_config


@app.cell
def _(mmm):
    with mmm.model:
        mmm.idata.update(pm.sample_prior_predictive(1000))
        mmm.idata.update(pm.sample_posterior_predictive(mmm.idata))
        pm.compute_log_likelihood(mmm.idata)
        pm.stats.compute_log_prior(mmm.idata)
    idata = mmm.idata
    return (idata,)


@app.cell
def _(idata, mmm):
    az.summary(
        idata,
        kind="diagnostics",
        var_names=[var.name for var in mmm.model.free_RVs],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prior Sensitivity Analysis

    `az.psense_summary` lets us check how sensitive posterior quantities are to our prior choices –
    **without refitting the model**. It uses PSIS importance weighting to ask "what if I changed this prior?"
    and measures the answer.

    **Step A**: First, let's see sensitivity on a model parameter (`saturation_lam`).
    """)
    return


@app.cell
def _(idata, mmm):
    mo.vstack(
        [
            mo.md(
                f"Assess prior and likelihood sensitivity for the {len(mmm.model.free_RVs)} multidimensional parameters."
            ),
            pd.concat(
                [
                    az.psense_summary(
                        idata, var_names=[var.name], prior_var_names=[var.name]
                    )
                    for var in mmm.model.free_RVs
                ]
            ),
        ]
    )
    return


@app.cell
def _(idata):
    # Step A: Sensitivity on model parameter (saturation_lam)
    az.psense_summary(
        idata,
        var_names=["saturation_lam", "saturation_beta"],
        prior_var_names=["saturation_lam", "saturation_beta"],
        likelihood_var_names=["y"],
        threshold=0.05,
    )
    return


@app.cell
def _(idata):
    az.plot_psense_dist(
        idata,
        var_names=["saturation_lam", "saturation_beta"],
        prior_var_names=["saturation_lam", "saturation_beta"],
        likelihood_var_names=["y"],
        coords={"channel": "Radio"},
    )
    return


@app.cell
def _(idata):
    az.plot_psense_quantities(
        idata,
        var_names=["saturation_lam"],
        prior_var_names=["saturation_lam"],
        likelihood_var_names=["y"],
    )
    return


@app.cell
def _(idata):
    # Step B: Add ROAS to idata and check sensitivity directly on it
    idata["posterior"]["ROAS"] = idata["posterior"][
        "channel_contribution_original_scale"
    ].sum("date") / idata["constant_data"]["channel_data"].sum("date")

    # Now check sensitivity on ROAS (a KPI stakeholders understand)
    az.psense_summary(
        idata,
        var_names=["ROAS"],
        # prior_var_names=["intercept_contribution"],
        prior_var_names=["adstock_alpha", "saturation_beta", "saturation_lam"],
        likelihood_var_names=["y"],
        # coords={"channel": ["Radio"]},
        threshold=0.05,
    )
    return


@app.cell
def _(idata):
    az.plot_psense_quantities(
        idata,
        var_names=["ROAS"],
        prior_var_names=["adstock_alpha", "saturation_beta", "saturation_lam"],
        prior_coords={"channel": ["TV", "Radio"]},
        likelihood_var_names=["y"],
        coords={"channel": ["TV", "Radio"]},
    )
    return


@app.cell
def _(idata):
    az.plot_psense_dist(
        idata,
        var_names=["ROAS"],
        prior_var_names=["adstock_alpha", "saturation_beta", "saturation_lam"],
        prior_coords={"channel": ["TV", "Radio"]},
        likelihood_var_names=["y"],
        coords={"channel": ["TV", "Radio"]},
        kind="ecdf",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Step B**: Now let's check all channels' ROAS sensitivity to the holiday prior (`gamma_control`).
    """)
    return


@app.cell
def _(idata):
    az.psense_summary(
        idata,
        var_names=["ROAS"],
        prior_var_names=["gamma_control", "gamma_fourier"],
        threshold=0.05,
    )
    return


@app.cell
def _(
    X_train,
    channel_columns,
    control_columns,
    date_column,
    model_config,
    target_column,
    y_train,
):
    # Rebuild model with lift tests
    mmm2 = MMM(
        model_config=model_config,
        sampler_config={"progressbar": False},
        target_column=target_column,
        date_column=date_column,
        adstock=GeometricAdstock(l_max=6),
        saturation=LogisticSaturation(),
        channel_columns=channel_columns,
        control_columns=control_columns,
        yearly_seasonality=5,
    )
    mmm2.build_model(X_train, y_train)

    # Multiple lift tests for TV across saturation curve range
    mmm2.add_lift_test_measurements(
        pd.DataFrame(
            [
                {
                    "channel": "TV",
                    "x": 400_000,
                    "delta_x": 100_000,
                    "delta_y": 8_000_000,
                    "sigma": 500_000,
                },
                {
                    "channel": "TV",
                    "x": 500_000,
                    "delta_x": 100_000,
                    "delta_y": 8_000_000,
                    "sigma": 500_000,
                },
            ]
        )
    )
    mmm2.add_original_scale_contribution_variable(
        [
            "y",
            "intercept_contribution",
            "control_contribution",
            "channel_contribution",
            "fourier_contribution",
            "yearly_seasonality_contribution",
        ]
    )
    mmm2.fit(
        X=X_train,
        y=y_train,
        chains=4,
        tune=500,
        draws=1000,
        target_accept=0.85,
        random_seed=rng,
        nuts_sampler="nutpie",
    )

    with mmm2.model:
        pm.sample_prior_predictive(1000)
        pm.sample_posterior_predictive(mmm2.idata)
        pm.compute_log_likelihood(mmm2.idata)
        pm.stats.compute_log_prior(mmm2.idata)

    mmm2.idata["posterior"]["ROAS"] = mmm2.idata["posterior"][
        "channel_contribution_original_scale"
    ].sum("date") / mmm2.idata["constant_data"]["channel_data"].sum("date")
    return (mmm2,)


@app.cell
def _(mmm2):
    az.summary(
        mmm2.idata,
        kind="diagnostics",
        var_names=[var.name for var in mmm2.model.free_RVs],
    )
    return


@app.cell
def _(mmm2):
    p_scaled = power_scale_dataset(
        mmm2.idata,
        group="likelihood",
        sample_dims=["chain", "draw"],
        group_var_names=["lift_measurements"],
        group_coords={},
        alphas=(0.8, 1.25),
    )
    return (p_scaled,)


@app.cell
def _(p_scaled):
    p_scaled_mod = (
        p_scaled.sel(channel="TV")
        .drop_vars(["draw", "chain"])
        .rename_dims({"sample": "draw"})
        .assign_coords({"draw": np.arange(4000)})
        .drop_vars(["sample", "channel"])
    )
    for var in p_scaled_mod.data_vars:
        p_scaled_mod[var] = (
            p_scaled_mod[var]
            .expand_dims("chain")
            .assign_coords({"chain": np.arange(1)})
        )
    return (p_scaled_mod,)


@app.cell
def _():
    dummy_sat = LogisticSaturation(
        prefix="saturation",
        priors={
            "lam": Prior("Gamma", alpha=3, beta=1, dims="alpha"),
            "beta": Prior("HalfNormal", sigma=2, dims="alpha"),
        },
    )
    return (dummy_sat,)


@app.cell
def _(dummy_sat, p_scaled_mod):
    dummy_sat.plot_curve(dummy_sat.sample_curve(p_scaled_mod))
    return


if __name__ == "__main__":
    app.run()

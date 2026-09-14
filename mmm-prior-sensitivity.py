# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "arviz==1.3.0",
#     "arviz-plots",
#     "arviz-stats==1.3.1",
#     "marimo>=0.24.0",
#     "matplotlib==3.11.1",
#     "numpy==2.4.6",
#     "nutpie==0.16.11",
#     "pandas==3.0.5",
#     "preliz>=0.20.0,<1.0",
#     "pymc==6.2.0",
#     "pymc-extras==0.14.0",
#     "pymc-marketing==1.1.0",
#     "seaborn",
#     "xarray==2026.7.0",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import warnings

    import arviz as az
    import arviz_plots as azp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import preliz as pz
    import pymc as pm
    import seaborn as sns
    import xarray as xr
    from pymc_extras.prior import Prior
    from pymc_marketing.hsgp_kwargs import HSGPKwargs
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM
    from xarray import DataArray

    warnings.filterwarnings("ignore", category=FutureWarning)
    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.facecolor"] = "white"

    seed = sum(map(ord, "mmm_prior_sensitivity_roas"))
    rng = np.random.default_rng(seed=seed)

    biz_prior_x1 = pz.distributions.Normal(100, 20)
    biz_prior_x2 = pz.distributions.Normal(150, 50)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Prior Sensitivity in the MMM Workflow

    This case study uses the synthetic, confounded data from PyMC-Marketing's
    [ROAS lift-test case study](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html).
    The goal is not to introduce every MMM diagnostic, but to show one practical workflow:

    1. fit an initial MMM with a stakeholder-informed **business prior** on ROAS,
    2. distinguish sensitivity in internal parameters from sensitivity in the ROAS estimate,
    3. communicate the decision-relevant conflict instead of tuning it away, and
    4. add lift-test evidence to see whether the conflict and prior sensitivity disappear.

    We keep the synthetic true ROAS hidden until the end, so the modeling decisions are based on diagnostics and domain assumptions rather than the answer key.
    """)
    return


@app.cell
def _():
    DATA_URL = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_roas_data.csv"
    raw_df = pd.read_csv(DATA_URL, parse_dates=["date"])
    model_df = raw_df.filter(["date", "x1", "x2", "y"]).copy()

    channel_columns = ["x1", "x2"]
    target_column = "y"
    date_column = "date"
    X = model_df.drop(columns=[target_column])
    y = model_df[target_column]
    return X, channel_columns, date_column, model_df, raw_df, target_column, y


@app.cell(hide_code=True)
def _(model_df):
    mo.vstack(
        [
            mo.md(
                """
                ## Setup and Data

                The observed data contain 2.5 years of weekly spend for two media channels, `x1` and `x2`, plus the target `y`.
                The data-generating process includes an unobserved confounder, `z`, that affects spend and sales.
                A model that cannot observe `z` can predict sales well while assigning the wrong ROAS to the channels.
                """
            ),
            model_df.head(),
        ]
    )
    return


@app.cell
def _(model_df):
    _fig, _axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, layout="constrained"
    )
    sns.lineplot(x="date", y="y", data=model_df, color="black", ax=_axes[0])
    _axes[0].set_title("Target")
    spend_long = model_df.melt(
        id_vars=["date"],
        value_vars=["x1", "x2"],
        var_name="channel",
        value_name="spend",
    )
    sns.lineplot(
        x="date", y="spend", hue="channel", data=spend_long, ax=_axes[1]
    )
    _axes[1].set_title("Channel spend")
    _fig
    return


@app.cell(hide_code=True)
def _(business_prior_df):
    mo.vstack(
        [
            mo.md(r"""
            ## 1. Initial Model: Business Prior and Observational Data

            Stakeholders expect ROAS near 100 for `x1` and 150 for `x2`.
            We call these expectations the **business prior**: functionally, they express prior understanding before fitting the MMM.
            Because ROAS is a derived quantity rather than a free model parameter, PyMC-Marketing represents this prior with `add_cost_per_target_calibration`, an extra model factor on the implied ROAS.

            The initial model also has weakly informative media priors and a flexible time-varying baseline whose amplitude prior allows substantial non-media movement.
            """),
            business_prior_df,
        ]
    )
    return


@app.cell
def _():
    # Encode business prior to a dataframe passed to add_cost_per_target_calibration:
    business_prior_df = pd.DataFrame(
        {
            "channel": ["x1", "x2"],
            "roas": [100, 150],
            "sigma": [20, 50],
        }
    )

    baseline_model_config = {
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=.5)),
        "gamma_fourier": Prior("Normal", mu=0, sigma=2, dims="fourier_mode"),
        "intercept_tvp_config": HSGPKwargs(
            m=50, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None
        ),
        "adstock_alpha": Prior("Beta", alpha=2, beta=3, dims="channel"),
        "saturation_lam": Prior("Gamma", alpha=2, beta=2, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1, dims="channel"),
    }

    sampler_config = {
        "tune": 1_000,
        "chains": 6,
        "cores": 6,
        "draws": 2_000,
        "random_seed": rng,
        "target_accept": 0.94,
    }
    lift_sampler_config = {
        **sampler_config,
        "target_accept": 0.96,
    }
    return (
        baseline_model_config,
        business_prior_df,
        lift_sampler_config,
        sampler_config,
    )


@app.cell
def _(channel_columns, date_column, target_column):
    def build_mmm(model_config):
        return MMM(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            date_column=date_column,
            channel_columns=channel_columns,
            target_column=target_column,
            time_varying_intercept=True,
            time_varying_media=False,
            yearly_seasonality=5,
            model_config=model_config,
        )

    def finish_model(mmm, X, y, sampler_config, business_prior_df=None):
        mmm.add_original_scale_contribution_variable(
            var=[
                "channel_contribution",
                "fourier_contribution",
                "intercept_contribution",
            ]
        )
        mmm.add_cost_per_target_calibration(
            data=X,
            calibration_data=business_prior_df,
            name_prefix="business_prior",
            target_column="roas",
            target_per_cost=True,
        )
        _ = mmm.fit(X, y, **sampler_config)
        _ = mmm.sample_posterior_predictive(
            X, extend_idata=True, combined=True, random_seed=rng
        )
        # The sensitivity analysis
        with mmm.model:
            pm.compute_log_likelihood(mmm.idata)
            pm.stats.compute_log_prior(mmm.idata)

        # Internally, the business prior is expressed as a likelihood and
        # must be moved to the log_prior group for the sensitivity analysis.
        mmm.idata["log_prior"]["business_prior"] = mmm.idata["log_likelihood"][
            "business_prior"
        ]
        mmm.idata["posterior"]["ROAS"] = (
            mmm.incrementality.compute_incremental_contribution("all_time")
            / mmm.idata["constant_data"]["channel_data"].sum("date")
        )
        return mmm

    def roas_summary(mmm):
        summary = az.summary(
            mmm.idata,
            var_names=["ROAS"],
            ci_prob=0.94,
            ci_kind="hdi",
        )
        interval_columns = [
            column for column in summary.columns if column.startswith("hdi_")
        ]
        return summary[["mean", "sd", *interval_columns]].round(1)

    def convergence_summary(mmm):
        diagnostics = az.summary(
            mmm.idata,
            var_names=[var.name for var in mmm.model.free_RVs],
            kind="diagnostics",
        )
        return pd.Series(
            {
                "divergences": int(
                    mmm.idata["sample_stats"]["diverging"].sum().item()
                ),
                "max_r_hat": diagnostics["r_hat"].max(),
                "min_ess_bulk": diagnostics["ess_bulk"].min(),
                "min_ess_tail": diagnostics["ess_tail"].min(),
            },
            name="diagnostic",
        ).round(3)

    return build_mmm, convergence_summary, finish_model, roas_summary


@app.cell
def _(
    X,
    baseline_model_config,
    build_mmm,
    business_prior_df,
    finish_model,
    sampler_config,
    y,
):
    business_mmm = build_mmm(baseline_model_config)
    business_mmm.build_model(X, y)
    business_mmm = finish_model(
        business_mmm,
        X,
        y,
        sampler_config=sampler_config,
        business_prior_df=business_prior_df,
    )
    return (business_mmm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model diagnostics

    Before interpreting ROAS, we verify that the fit has no divergences, a maximum R-hat below 1.01, and adequate bulk and tail effective sample sizes.
    """)
    return


@app.cell
def _(business_mmm, convergence_summary):
    convergence_summary(business_mmm)
    return


@app.cell
def _(business_mmm, model_df):
    # We will reuse this later
    def contribution_plot(mmm, model_df, title):
        fig, axes = mmm.plot.contributions_over_time(
            var=[
                "channel_contribution_original_scale",
                "intercept_contribution_original_scale",
                "fourier_contribution_original_scale",
            ],
            dims={"channel": ["x1", "x2"]},
            combine_dims=True,
            hdi_prob=0.94,
            figsize=(12, 7),
        )
        sns.lineplot(
            x="date",
            y="y",
            data=model_df,
            color="black",
            label="observed y",
            ax=axes[0, 0],
        )
        legend = axes[0, 0].get_legend()
        legend.set_bbox_to_anchor((0.8, -0.12))
        fig.suptitle(title, fontweight="bold")
        return fig

    contribution_plot(
        business_mmm, model_df, "Initial model: component contributions"
    )
    return (contribution_plot,)


@app.cell
def _(business_mmm):
    _pc = azp.plot_dist(
        business_mmm.idata["posterior"]["ROAS"].to_dataset(name="roas"),
        col_wrap=1,
        figure_kwargs={
            "figsize": (10, 6),
            "sharex": True,
            "layout": "constrained",
        },
    )
    _fig = _pc.viz["/"]["figure"].values.item()
    _axes = _fig.axes
    biz_prior_x1.plot_pdf(
        color="C2", linestyle="--", linewidth=2, ax=_axes[0], legend=None
    )
    biz_prior_x2.plot_pdf(
        color="C2", linestyle="--", linewidth=2, ax=_axes[1], legend=None
    )
    _axes[0].lines[-1].set_label("Business prior")
    _axes[1].lines[-1].set_label("Business prior")
    _axes[0].legend(loc="upper right")
    _axes[0].set(title="Initial ROAS: x1")
    _axes[1].legend(loc="upper right")
    _axes[1].set(title="Initial ROAS: x2", xlabel="ROAS")
    _fig.suptitle(
        "Initial model: ROAS posterior and business prior", fontweight="bold"
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior Sensitivity Check

    We next assess the prior and likelihood sensitivity of ROAS. This shows how strongly each component influences the reported estimates and helps identify unexpectedly influential priors.

    To answer this, we use **power-scaling sensitivity analysis** ([Kallioinen et al., 2024](https://link.springer.com/article/10.1007/s11222-023-10366-5)), a diagnostic that evaluates how much influence each prior or likelihood component exerts on the posterior by scaling its log-density with a power parameter $\alpha$.

    The standard posterior is

    $$p(\theta \mid y) \propto p(y \mid \theta) \, p(\theta)$$

    and we consider two power-scaling variants:

    - **Prior sensitivity**: $p(\theta \mid y) \propto p(y \mid \theta) \, p(\theta)^\alpha$
    - **Likelihood sensitivity**: $p(\theta \mid y) \propto p(y \mid \theta)^\alpha \, p(\theta)$

    When $\alpha = 1$, the model is fitted to the original specification. When $\alpha < 1$, the scaled component is weakened; when $\alpha > 1$, it is amplified. By examining how posterior quantities of interest change as we perturb $\alpha$, we can see how tension between the priors and likelihood affects the estimates we want to report.

    The following figure illustrates the effect of power scaling on a standard Normal distribution.
    """)
    return


@app.cell
def _():
    grid = np.linspace(-4, 4, 500)
    _fig, _ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for alpha, color in [(0.7, "C0"), (1.0, "black"), (1.5, "C3")]:
        sigma = 1 / np.sqrt(alpha)
        density = np.exp(-0.5 * (grid / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
        _ax.plot(
            grid, density, color=color, linewidth=2, label=rf"$\alpha={alpha}$"
        )
    _ax.set(
        title="Power scaling a standard Normal density",
        ylabel="Density",
    )
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A **key practical advantage is that we do not need to refit the model**, and we can approximate the posterior at different $\alpha$ values using **Pareto-smoothed importance sampling (PSIS)** ([Vehtari et al., 2024](https://www.jmlr.org/papers/v25/19-556.html)) from the original posterior samples. This makes the check computationally efficient. Because importance reweighting can reduce the effective sample size, we deliberately retain 2,000 draws across each of six chains, for 12,000 posterior draws in total, to provide a sufficiently large sample for the sensitivity analysis.

    We distinguish between sensitivity in internal model parameters and sensitivity in the ROAS estimate that we plan to report to the marketing team. For each posterior target, we test four prior blocks:

    - **Media priors**: adstock (`adstock_alpha`) and saturation (`saturation_lam`, `saturation_beta`) parameters.
    - **Baseline prior**: the prior on the time-varying intercept, which allows smooth temporal variation in non-media demand. Although the prior permits movement in either direction, the fitted baseline increases over this observation period.
    - **Business prior**: stakeholder expectations for channel ROAS.
    - **Seasonality prior**: the Fourier coefficients (`gamma_fourier`) governing recurring yearly variation.

    Values above 0.05 are a practical flag that a posterior target changes considerably when that block is slightly strengthened or weakened. We read this as evidence for investigation, with the usual Monte Carlo uncertainty caveat, rather than as a mechanical pass/fail test.
    """)
    return


@app.cell
def _():
    prior_blocks = {
        "media_priors": ["adstock_alpha", "saturation_lam", "saturation_beta"],
        "baseline_prior": [
            "intercept_baseline",
            "intercept_latent_process_raw_eta",
            "intercept_latent_process_raw_ls",
            "intercept_latent_process_raw_hsgp_coefs_offset",
        ],
        "business_prior": ["business_prior"],
        "seasonality_prior": ["gamma_fourier"],
    }
    posterior_targets = {
        "ROAS": ["ROAS"],
        "media_parameters": [
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
        ],
        "seasonality": ["gamma_fourier"],
    }

    def psense_by_target_and_block(idata):
        likelihood_var_names = [
            var
            for var in ["y", "lift_measurements"]
            if var in idata["log_likelihood"].data_vars
        ]
        detail_tables = []
        matrix_rows = []
        for target_name, target_var_names in posterior_targets.items():
            row = {"posterior_target": target_name}
            for block_name, prior_names in prior_blocks.items():
                summary = az.psense_summary(
                    idata,
                    var_names=target_var_names,
                    prior_var_names=prior_names,
                    likelihood_var_names=likelihood_var_names,
                ).reset_index(names="posterior_variable")
                summary.insert(0, "prior_block", block_name)
                summary.insert(0, "posterior_target", target_name)
                detail_tables.append(summary)
                row[block_name] = summary["prior"].max()
                row["likelihood"] = summary["likelihood"].max()
            matrix_rows.append(row)
        matrix = (
            pd.DataFrame(matrix_rows)
            .set_index("posterior_target")
            .round(3)
        )
        details = pd.concat(detail_tables, ignore_index=True)
        return matrix, details

    return (psense_by_target_and_block,)


@app.cell
def _(business_mmm, psense_by_target_and_block):
    business_psense_matrix, business_psense_details = (
        psense_by_target_and_block(business_mmm.idata)
    )
    business_psense_matrix
    return business_psense_details, business_psense_matrix


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The matrix separates sensitivity in internal model parameters from sensitivity in the reporting target. Individual saturation and seasonality parameters can be sensitive because multiple model components can explain similar temporal patterns. That does not automatically imply that all-time ROAS is sensitive: ROAS depends on the joint media response rather than on either saturation parameter in isolation.

    The seasonality coefficients respond to the baseline prior. This overlap is not decision-relevant here: both components describe non-media fluctuations for which we have no observed predictors, and we do not need to attribute those fluctuations precisely between a recurring seasonal pattern and a smooth time-varying baseline. Crucially, all-time ROAS remains insensitive to both prior blocks.

    We inspect the media parameters in more detail below. Sensitivity to the business prior is expected because that factor is designed to inform the implied media contribution. The more relevant modeling check is whether the supposedly weak media priors conflict with the observational likelihood.
    """)
    return


@app.cell
def _(business_psense_details):
    business_psense_details.query(
        "posterior_target == 'media_parameters' and prior_block == 'media_priors'"
    ).set_index("posterior_variable")[["prior", "likelihood", "diagnosis"]]
    return


@app.cell
def _(business_mmm):
    az.psense_summary(business_mmm.idata,var_names=["ROAS", "saturation_lam", "saturation_beta"], prior_var_names=["business_prior"], likelihood_var_names=["y"])
    return


@app.cell
def _(business_mmm):
    az.plot_psense_quantities(
        business_mmm.idata,
        var_names=["saturation_lam", "saturation_beta"],
        prior_var_names=[
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
        ],
        likelihood_var_names=["y"],
        coords={"channel": ["x1", "x2"]},
        quantities=["mean", "sd"],
    )
    return


@app.cell
def _(business_mmm):
    az.plot_psense_quantities(
        business_mmm.idata,
        var_names=["ROAS"],
        prior_var_names=["business_prior"],
        likelihood_var_names=["y"],
        quantities=["mean", "sd"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The saturation-rate and saturation-amplitude parameters are individually sensitive to their priors and to the likelihood, indicating that the observational data do not identify each parameter separately. However, their joint implications are more stable: all-time ROAS remains below the sensitivity threshold for the media, baseline, and seasonality prior blocks.

    The remaining decision-relevant conflict is concentrated in `x1` ROAS, which is sensitive to both the stakeholder-informed business prior and the observational likelihood. We cannot resolve that conflict by deciding that either source must be correct. Instead of tuning the model toward either one, we return to the marketing team and recommend lift tests that directly inform the channel response.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Lift-Test Update: Adding New Evidence

    Suppose the team runs two lift tests for `x1` at different spend levels.
    Each row below is a future experimental observation: pre-test spend `x`, spend change `delta_x`, measured incremental sales `delta_y`, and its uncertainty `sigma`.
    These measurements add likelihood information anchored to the saturation curve, so we expect the ROAS estimate to become less prior-sensitive.
    """)
    return


@app.cell
def _(X):
    df_lift_test = pd.DataFrame(
        data={
            "channel": ["x1", "x1"],
            "x": [0.25, 0.8],
            "delta_x": [0.25, 0.8],
            "delta_y": [23.34703279570842, 74.71050494626694],
            "sigma": [3, 3],
            "date": pd.to_datetime(
                [
                    X["date"].max() - pd.Timedelta(weeks=50),
                    X["date"].max() - pd.Timedelta(weeks=14),
                ]
            ),
        }
    )
    df_lift_test
    return (df_lift_test,)


@app.cell
def _(
    X,
    baseline_model_config,
    build_mmm,
    business_prior_df,
    df_lift_test,
    finish_model,
    lift_sampler_config,
    y,
):
    lift_mmm = build_mmm(baseline_model_config)
    lift_mmm.build_model(X, y)
    lift_mmm.add_lift_test_measurements(df_lift_test=df_lift_test)
    lift_mmm = finish_model(
        lift_mmm,
        X,
        y,
        sampler_config=lift_sampler_config,
        business_prior_df=business_prior_df,
    )
    return (lift_mmm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model diagnostics

    We again verify convergence before interpreting the lift-calibrated model.
    """)
    return


@app.cell
def _(convergence_summary, lift_mmm):
    convergence_summary(lift_mmm)
    return


@app.cell
def _(contribution_plot, lift_mmm, model_df):
    contribution_plot(
        lift_mmm, model_df, "Lift-calibrated model: component contributions"
    )
    return


@app.cell
def _(business_mmm, lift_mmm, roas_summary):
    pd.concat(
        {
            "observational": roas_summary(business_mmm),
            "lift_calibrated": roas_summary(lift_mmm),
        },
        names=["model"],
    )
    return


@app.cell
def _(business_mmm, lift_mmm):
    roas_model_comparison = xr.concat(
        [
            business_mmm.idata["posterior"]["ROAS"],
            lift_mmm.idata["posterior"]["ROAS"],
        ],
        dim="model",
    ).assign_coords(model=["observational", "lift_calibrated"])

    _fig, _axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 4), layout="constrained"
    )
    for _channel, _ax in zip(["x1", "x2"], _axes, strict=False):
        for _model_name, _color in [
            ("observational", "C0"),
            ("lift_calibrated", "C1"),
        ]:
            _values = roas_model_comparison.sel(
                model=_model_name, channel=_channel
            ).values.ravel()
            _ax.hist(
                _values,
                bins=40,
                density=True,
                alpha=0.45,
                color=_color,
                label=_model_name,
            )
        if _channel == "x1":
            biz_prior_x1.plot_pdf(
                color="C2", linestyle="--", linewidth=2, ax=_ax
            )
        else:
            biz_prior_x2.plot_pdf(
                color="C2", linestyle="--", linewidth=2, ax=_ax
            )
        _ax.lines[-1].set_label("Business prior")
        _ax.set(title=f"{_channel} ROAS", xlabel="ROAS")
    _axes[0].legend()
    _fig.suptitle(
        "ROAS posterior before and after lift tests", fontweight="bold"
    )
    _fig
    return


@app.cell
def _(business_psense_matrix, lift_mmm, psense_by_target_and_block):
    lift_psense_matrix, lift_psense_details = psense_by_target_and_block(
        lift_mmm.idata
    )
    pd.concat(
        {
            "observational": business_psense_matrix,
            "lift_calibrated": lift_psense_matrix,
        },
        names=["model"],
    )
    return (lift_psense_details,)


@app.cell
def _(business_psense_details, lift_psense_details):
    def media_prior_details(details):
        return details.query(
            "posterior_target == 'media_parameters' and prior_block == 'media_priors'"
        ).set_index("posterior_variable")[["prior", "likelihood", "diagnosis"]]

    pd.concat(
        {
            "observational": media_prior_details(business_psense_details),
            "lift_calibrated": media_prior_details(lift_psense_details),
        },
        names=["model"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The observational sales and lift-test measurements together form the likelihood for the updated model. The sales observtions alone are confounded, but the lift tests add causal evidence and help correct the resulting ROAS bias. In this case that evidence happens to align with the business prior.

    Some internal parameters remain sensitive after adding the experiments; two lift tests for `x1` cannot identify every media parameter or resolve every internal decomposition. The reporting target is nevertheless stable: ROAS remains below the practical threshold for the media, baseline, and seasonality prior blocks, while sensitivity to the business prior falls below the threshold. The earlier ROAS prior-likelihood conflict is therefore no longer detected by the power-scaling diagnostic.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Retrospective Validation and Takeaways

    Only now do we reveal the simulation truth.
    In real projects this row would not exist; here it checks whether the workflow moved us in the right direction without using the answer key to choose the model.
    """)
    return


@app.cell
def _(raw_df):
    true_roas_x1 = (raw_df["y"] - raw_df["y01"]).sum() / raw_df["x1"].sum()
    true_roas_x2 = (raw_df["y"] - raw_df["y02"]).sum() / raw_df["x2"].sum()
    true_roas = DataArray(
        [true_roas_x1, true_roas_x2],
        dims="channel",
        coords={"channel": ["x1", "x2"]},
        name="true_roas",
    )
    return (true_roas,)


@app.cell
def _(business_mmm, lift_mmm):
    posterior_roas = xr.concat(
        [
            business_mmm.idata["posterior"]["ROAS"],
            lift_mmm.idata["posterior"]["ROAS"],
        ],
        dim="model",
    ).assign_coords(model=["observational", "lift_calibrated"])
    return (posterior_roas,)


@app.cell
def _(posterior_roas, true_roas):
    _fig, _axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 4), layout="constrained"
    )
    for _channel, _ax in zip(["x1", "x2"], _axes, strict=False):
        for _model_name, _color in [
            ("observational", "C0"),
            ("lift_calibrated", "C3"),
        ]:
            _values = posterior_roas.sel(
                model=_model_name, channel=_channel
            ).values.ravel()
            _ax.hist(
                _values,
                bins=40,
                density=True,
                alpha=0.35,
                color=_color,
                label=_model_name,
            )
        _ax.axvline(
            float(true_roas.sel(channel=_channel)),
            color="black",
            linestyle="--",
            label="truth",
        )
        _ax.set(title=f"{_channel} ROAS", xlabel="ROAS")
    _axes[0].legend()
    _fig.suptitle(
        "Retrospective check against hidden synthetic truth", fontweight="bold"
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Reporting checklist**

    - Define the quantity of interest before fitting; here it is all-time channel ROAS.
    - Check prior sensitivity for that quantity after the initial fit, not only generic parameter diagnostics.
    - Distinguish sensitivity in internal parameters from sensitivity in the reporting target.
    - If the model and business prior disagree, report the conflict and propose evidence that would resolve it.
    - Lift tests can add enough causal information that ROAS is no longer materially sensitive to the original prior choices.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    - Kallioinen, N., Paananen, T., Bürkner, P.-C., & Vehtari, A. (2024). Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Statistics and Computing*, 34, 46.
    - Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling. *Journal of Machine Learning Research*, 25(72), 1–58.
    - PyMC-Marketing ROAS case study: [Mitigating Unobserved Confounders in MMMs with Lift Test Likelihoods](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html).
    - PyMC-Marketing ROAS calibration case study: [Calibrating an MMM with ROAS Estimates](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas_calibration.html).
    """)
    return


if __name__ == "__main__":
    app.run()

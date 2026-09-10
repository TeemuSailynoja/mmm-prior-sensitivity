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
#     "pymc==6.2.0",
#     "pymc-extras==0.14.0",
#     "pymc-marketing==1.1.0",
#     "seaborn",
#     "xarray==2026.7.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import warnings

    import arviz as az
    import arviz_plots as azp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import seaborn as sns
    import xarray as xr
    from arviz_stats.psense import power_scale_dataset
    from matplotlib.lines import Line2D
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Resolving prior-data conflict with lift tests

    This notebook uses the PyMC-Marketing ROAS case-study model from
    [Mitigating Unobserved Confounders in MMMs with Lift Test Likelihoods](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html).

    In this case study, I demonstrate how **lift tests can resolve conflicts between business expectations and observational data**.

    We fit two models:

    1. **Business-prior MMM**: the marketing team provides their expectations for channel ROAS and these are incorporated to the MMM priors.
    2. **Lift-calibrated MMM**: the same model plus lift-test likelihoods on the saturation curves.

    We focus on media priors (adstock and saturation parameters) and business priors (channel ROAS expectations) as the key sources of prior-observation conflict.
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

    amplitude = 100
    true_roas_x1 = (raw_df["y"] - raw_df["y01"]).sum() / raw_df["x1"].sum()
    true_roas_x2 = (raw_df["y"] - raw_df["y02"]).sum() / raw_df["x2"].sum()
    true_roas = DataArray(
        [true_roas_x1, true_roas_x2],
        dims="channel",
        coords={"channel": channel_columns},
        name="true_roas",
    )
    return (
        X,
        channel_columns,
        date_column,
        model_df,
        target_column,
        true_roas,
        true_roas_x1,
        true_roas_x2,
        y,
    )


@app.cell(hide_code=True)
def _(model_df, true_roas_x1, true_roas_x2):
    mo.vstack(
        [
            mo.md(
                f"""
                ## Data

                Our observations contain 2.5 years of spend values for two media channels, `x1` and `x2`, as well as the observed target variable, `y`. The data generating process contains an unobserved confounder, `z`, which affects both spend and sales, so our model that cannot see `z` will still forecast well while producing biased ROAS estimates.

                Known all-time true ROAS (for validation only):
                - `x1`: **{true_roas_x1:0.1f}**
                - `x2`: **{true_roas_x2:0.1f}**
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


@app.cell
def _():
    baseline_model_config = {
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        "gamma_fourier": Prior("Normal", mu=0, sigma=2, dims="fourier_mode"),
        "intercept_tvp_config": HSGPKwargs(
            m=100, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None
        ),
        "adstock_alpha": Prior("Beta", alpha=2, beta=3, dims="channel"),
        "saturation_lam": Prior("Gamma", alpha=2, beta=2, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1, dims="channel"),
    }

    # Keep the interactive case study reasonably light. Increase draws for publication-quality figures.
    sampler_config = {
        "tune": 500,
        "chains": 4,
        "draws": 1_000,
        "target_accept": 0.95,
        "random_seed": rng,
    }
    return baseline_model_config, sampler_config


@app.cell
def _(channel_columns, date_column, target_column):
    def add_roas_to_idata(mmm):
        mmm.idata["posterior"]["ROAS"] = mmm.idata["posterior"][
            "channel_contribution_original_scale"
        ].sum("date") / mmm.idata["constant_data"]["channel_data"].sum("date")
        return mmm.idata

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

    return (build_mmm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Business-prior MMM

    The model incorporates the marketing team's expectations for channel ROAS as priors. These expectations are elicited before any modeling is done, based on the marketing team's knowledge of their customer base, marketing strategies, and pricing.

    The business provided the following ROAS expectations:

    - Channel `x1`: ROAS ~ Normal(100, 20)
    - Channel `x2`: ROAS ~ Normal(150, 50)

    These priors are incorporated via `add_cost_per_target_calibration`, which adds Normal likelihood terms for each calibration row. We also set priors on the media parameters (adstock, saturation) that reflect our assumptions about how spend translates to sales, as well as a weakly informative prior on the channel efficiency.

    The key observation we will see after fitting the model is: **the business-informed priors are in conflict with the observed sales!**
    """)
    return


@app.cell
def _(X, baseline_model_config, build_mmm, sampler_config, y):
    business_priors_df = pd.DataFrame(
        {
            "channel": ["x1", "x2"],
            "roas": [100, 150],
            "sigma": [20, 50],
        }
    )

    business_mmm = build_mmm(baseline_model_config)
    business_mmm.build_model(X, y)
    business_mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "fourier_contribution",
            "intercept_contribution",
        ]
    )
    business_mmm.add_cost_per_target_calibration(
        data=X,
        calibration_data=business_priors_df,
        name_prefix="business_prior",
        target_column="roas",
        target_per_cost=True,
    )

    _ = business_mmm.fit(X, y, **sampler_config)
    # for _group_name in ["prior", "prior_predictive", "observed_data"]:
    #     if _group_name in business_prior:
    #         business_mmm.idata[_group_name] = business_prior[_group_name]
    _ = business_mmm.sample_posterior_predictive(
        X, extend_idata=True, combined=True, random_seed=rng
    )
    with business_mmm.model:
        pm.compute_log_likelihood(business_mmm.idata)
        pm.stats.compute_log_prior(business_mmm.idata)
        business_mmm.idata["log_prior"]["business_prior"] = business_mmm.idata[
            "log_likelihood"
        ]["business_prior"]

    business_mmm.idata["posterior"]["ROAS"] = (
        business_mmm.incrementality.compute_incremental_contribution(
            "all_time"
        )
        / business_mmm.idata["constant_data"]["channel_data"].sum("date")
    )
    return business_mmm, business_priors_df


@app.cell
def _(business_mmm, true_roas_x1, true_roas_x2):
    _pc = azp.plot_dist(
        business_mmm.idata["posterior"]["ROAS"].to_dataset(name="roas"),
        col_wrap=1,
        figure_kwargs={
            "figsize": (10, 6),
            "sharex": True,
            "layout": "constrained",
        },
    )
    business_roas_fig = _pc.viz["/"]["figure"].values.item()
    business_roas_axes = business_roas_fig.axes
    business_roas_axes[0].axvline(
        true_roas_x1,
        color="black",
        linestyle="--",
        linewidth=2,
        label="true ROAS",
    )
    business_roas_axes[0].legend(loc="upper right")
    business_roas_axes[0].set(title="Business-prior ROAS: x1")
    business_roas_axes[1].axvline(
        true_roas_x2,
        color="black",
        linestyle="--",
        linewidth=2,
        label="true ROAS",
    )
    business_roas_axes[1].legend(loc="upper right")
    business_roas_axes[1].set(title="Business-prior ROAS: x2", xlabel="ROAS")
    business_roas_fig.suptitle(
        "Business-prior MMM: prior-observation conflict for x1",
        fontweight="bold",
    )
    business_roas_fig
    return


@app.cell
def _(business_mmm, true_roas_x1, true_roas_x2):
    roas_x1_mean = (
        business_mmm.idata["posterior"]["ROAS"].sel(channel="x1").mean().values
    )
    roas_x2_mean = (
        business_mmm.idata["posterior"]["ROAS"].sel(channel="x2").mean().values
    )
    mo.vstack(
        [
            mo.md(f"""
            ### Conflict detected!

            The business-prior model produces ROAS estimates that conflict with the business's prior expectations for channel `x1`:

            | Channel | Business prior | Posterior mean | True ROAS |
            |---------|---------------|----------------|-----------|
            | x1      | 100           | {roas_x1_mean:.1f} | {true_roas_x1:.1f} |
            | x2      | 150           | {roas_x2_mean:.1f} | {true_roas_x2:.1f} |

            Channel `x1` shows significant prior-observation conflict: the business expects ROAS around 100, but the model's posterior is far from that expectation. This conflict is valuable—it signals that the observational data alone cannot disentangle the channel effect from the unobserved confounder `z`.

            **This is not a failure.** It is a learning opportunity: when the model and the business disagree, we should ask *what additional information* would resolve the conflict. As discussed in the [PyMC-Marketing experimentation guide](https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_roas_experimentation.html), we can estimate how much a lift test would reduce our uncertainty on ROAS—and use that to justify whether the experiment is worth running.
            """)
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ROAS prior sensitivity to media priors

    We now ask whether all-time ROAS depends on the media priors. To answer this, we use **power scaling sensitivity analysis** ([Kallioinen et al., 2024](https://link.springer.com/article/10.1007/s11222-023-10366-5)), a diagnostic that evaluates how much influence each prior or likelihood component exerts on the posterior by scaling its log-density with a power parameter α.

    The standard posterior is

    $$p(\theta \mid y) \propto p(y \mid \theta) \, p(\theta)$$

    and we consider two power-scaling variants:

    - **Prior sensitivity**: $p(\theta \mid y) \propto p(y \mid \theta) \, p(\theta)^\alpha$
    - **Likelihood sensitivity**: $p(\theta \mid y) \propto p(y \mid \theta)^\alpha \, p(\theta)$

    When α = 1 the model is fitted to the original specification; when α < 1 that component is weakened, letting the rest of the model dominate; and when α > 1 it is amplified. By examining how posterior quantities of interest change across a range of α values, we can see how the tension between the prior and the likelihood plays out—and how the posterior shifts when we tilt the balance toward one side or the other.

    A key practical advantage is that we can approximate the posterior at different α values using **Pareto-smoothed importance sampling (PSIS)** ([Vehtari et al., 2024](https://www.jmlr.org/papers/v25/19-556.html)) from the original posterior samples, without refitting the model. This makes the whole process computationally very efficient and fast.

    We now ask whether all-time ROAS depends on the media priors:

    - **Media priors**: adstock (`adstock_alpha`) and saturation (`saturation_lam`, `saturation_beta`) parameters.
    - **Business priors**: channel ROAS expectations (`business_prior`).

    If ROAS is sensitive to these priors, that is evidence that the observational data alone cannot fully identify the channel effects—there is room for the priors to pull the posterior in different directions.
    """)
    return


@app.cell
def _(business_mmm):
    def roas_psense_by_block(idata):
        blocks = {
            "media_priors": [
                "adstock_alpha",
                "saturation_lam",
                "saturation_beta",
                "business_prior",
            ],
            "seasonality_prior": ["gamma_fourier"],
            "trend_baseline_priors": [
                "intercept_baseline",
                "intercept_latent_process_raw_eta",
                "intercept_latent_process_raw_ls",
                "intercept_latent_process_raw_hsgp_coefs_offset",
            ],
        }
        likelihood_var_names = [
            var
            for var in ["y", "lift_measurements"]
            if var in idata["log_likelihood"].data_vars
        ]
        return pd.concat(
            {
                block_name: az.psense_summary(
                    idata,
                    var_names=["ROAS"],
                    prior_var_names=prior_names,
                    likelihood_var_names=likelihood_var_names,
                    threshold=0.05,
                )
                for block_name, prior_names in blocks.items()
            },
            names=["prior_block"],
        )

    business_roas_psense = roas_psense_by_block(business_mmm.idata)
    business_roas_psense
    return business_roas_psense, roas_psense_by_block


@app.cell
def _(business_mmm):
    az.plot_psense_dist(
        business_mmm.idata,
        var_names=["ROAS"],
        prior_var_names=[
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
            "business_prior",
        ],
        likelihood_var_names=["y"],
        coords={"channel": ["x1", "x2"]},
        kind="ecdf",
    )
    return


@app.cell
def _(business_mmm):
    az.plot_psense_quantities(
        business_mmm.idata,
        var_names=["ROAS"],
        prior_var_names=[
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
            "business_prior",
        ],
        likelihood_var_names=["y"],
        coords={"channel": ["x1", "x2"]},
    )
    return


@app.cell
def _(business_mmm, model_df):
    fig, axes = business_mmm.plot.contributions_over_time(
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
        x="date", y="y", data=model_df, color="black", label="y", ax=axes[0, 0]
    )
    legend = axes[0, 0].get_legend()
    legend.set_bbox_to_anchor((0.8, -0.12))
    plt.gca()
    return


@app.cell
def _(baseline_model_config):
    adjusted_model_config = {
        **baseline_model_config,
        **{
            "intercept_tvp_config": HSGPKwargs(
                m=50,
                L=None,
                eta_lam=1.0,
                ls_mu=5.0,
                ls_sigma=5.0,
                cov_func=None,
            )
        },
    }
    return (adjusted_model_config,)


@app.cell
def _(
    X,
    adjusted_model_config,
    build_mmm,
    business_priors_df,
    sampler_config,
    y,
):
    adjusted_mmm = build_mmm(adjusted_model_config)
    adjusted_mmm.build_model(X, y)
    adjusted_mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "fourier_contribution",
            "intercept_contribution",
        ]
    )
    adjusted_mmm.add_cost_per_target_calibration(
        data=X,
        calibration_data=business_priors_df,
        name_prefix="business_prior",
        target_column="roas",
        target_per_cost=True,
    )

    _ = adjusted_mmm.fit(X, y, **sampler_config)
    _ = adjusted_mmm.sample_posterior_predictive(
        X, extend_idata=True, combined=True, random_seed=rng
    )
    with adjusted_mmm.model:
        pm.compute_log_likelihood(adjusted_mmm.idata)
        pm.stats.compute_log_prior(adjusted_mmm.idata)
        adjusted_mmm.idata["log_prior"]["business_prior"] = adjusted_mmm.idata[
            "log_likelihood"
        ]["business_prior"]

    adjusted_mmm.idata["posterior"]["ROAS"] = (
        adjusted_mmm.incrementality.compute_incremental_contribution(
            "all_time"
        )
        / adjusted_mmm.idata["constant_data"]["channel_data"].sum("date")
    )
    return (adjusted_mmm,)


@app.cell
def _(adjusted_mmm, model_df):
    _fig, _axes = adjusted_mmm.plot.contributions_over_time(
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
        x="date", y="y", data=model_df, color="black", label="y", ax=_axes[0, 0]
    )
    _legend = _axes[0, 0].get_legend()
    _legend.set_bbox_to_anchor((0.8, -0.12))
    plt.gca()
    return


@app.cell
def _(adjusted_mmm, roas_psense_by_block):
    roas_psense_by_block(adjusted_mmm.idata)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## MMM with an adjusted baseline prior

    Before proceeding to the lift test, the modeling team reviewed the initial model and decided that the baseline prior was too informative for our purposes. We do not want the baseline (trend and seasonality) to be pulling the channel estimates away from the data — we want that component of the model to be primarily driven by the observations.

    To this end, we refit the business-prior model with a weaker baseline prior: we reduce the number of HSGP basis functions from `m=100` to `m=50` and tighten the lengthscale prior from `ls_sigma=10` to `ls_sigma=5`. This makes the baseline more flexible and less constrained by the prior.

    Visually, the contributions-over-time plot shows that the baseline estimate has not changed dramatically — the overall decomposition of sales into channel, trend, and seasonality components looks similar. However, the modeling team's preference is for the baseline to yield to the data rather than impose its own structure. We proceed with this adjusted model going forward, and we still observe the same conflict between the business-informed media priors and the observational data — which brings us to the lift test.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Lift-calibrated MMM

    Lift tests add likelihood terms directly on the saturation curves. They are not just a prior on a summary ROAS; each experiment contributes information about the marginal response at a specific spend level.

    Here we use the same synthetic lift tests as the PyMC-Marketing ROAS notebook. Because this is simulated data, the lift-test means are generated from the known true ROAS.

    The lift test provides additional information that helps disentangle the channel effect from the unobserved confounder. The question is: **does the lift test resolve the prior-observation conflict we saw for channel `x1`?**
    """)
    return


@app.cell
def _(X, true_roas_x1):
    # df_lift_test = pd.DataFrame(
    #     data={
    #         "channel": ["x1", "x2", "x1", "x2"],
    #         "x": [0.25, 0.1, 0.8, 0.25],
    #         "delta_x": [0.25, 0.1, 0.8, 0.25],
    #         "delta_y": [
    #             true_roas_x1 * 0.25,
    #             true_roas_x2 * 0.1,
    #             true_roas_x1 * 0.8,
    #             true_roas_x2 * 0.25,
    #         ],
    #         "sigma": [3, 3, 3, 3],
    #         "date": pd.to_datetime(
    #             [
    #                 X["date"].max() - pd.Timedelta(weeks=50),
    #                 X["date"].max() - pd.Timedelta(weeks=30),
    #                 X["date"].max() - pd.Timedelta(weeks=14),
    #                 X["date"].max() - pd.Timedelta(weeks=12),
    #             ]
    #         ),
    #     }
    # )
    df_lift_test = pd.DataFrame(
        data={
            "channel": ["x1", "x1"],
            "x": [0.25, 0.8],
            "delta_x": [0.25, 0.8],
            "delta_y": [
                true_roas_x1 * 0.25,
                true_roas_x1 * 0.8,
            ],
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
    adjusted_model_config,
    build_mmm,
    business_priors_df,
    df_lift_test,
    sampler_config,
    y,
):
    lift_mmm = build_mmm(adjusted_model_config)
    lift_mmm.build_model(X, y)
    lift_mmm.add_lift_test_measurements(df_lift_test=df_lift_test)
    lift_mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "fourier_contribution",
            "intercept_contribution",
        ]
    )

    lift_mmm.add_cost_per_target_calibration(
        data=X,
        calibration_data=business_priors_df,
        name_prefix="business_prior",
        target_column="roas",
        target_per_cost=True,
    )
    _ = lift_mmm.fit(X, y, **sampler_config)
    _ = lift_mmm.sample_posterior_predictive(
        X, extend_idata=True, combined=True, random_seed=rng
    )
    with lift_mmm.model:
        pm.compute_log_likelihood(lift_mmm.idata)
        pm.stats.compute_log_prior(lift_mmm.idata)

    lift_mmm.idata["log_prior"]["business_prior"] = lift_mmm.idata[
        "log_likelihood"
    ]["business_prior"]
    lift_mmm.idata["posterior"]["ROAS"] = (
        lift_mmm.incrementality.compute_incremental_contribution("all_time")
        / lift_mmm.idata["constant_data"]["channel_data"].sum("date")
    )
    return (lift_mmm,)


@app.cell
def _(lift_mmm):
    az.summary(
        lift_mmm.idata,
        kind="diagnostics",
        var_names=["ROAS"] + [var.name for var in lift_mmm.model.free_RVs],
    )
    return


@app.cell
def _(lift_mmm, true_roas_x1, true_roas_x2):
    lift_mmm.idata["posterior"]["ROAS"]
    _pc = azp.plot_dist(
        lift_mmm.idata["posterior"]["ROAS"].to_dataset(name="roas"),
        col_wrap=1,
        figure_kwargs={
            "figsize": (10, 6),
            "sharex": True,
            "layout": "constrained",
        },
    )
    lift_roas_fig = _pc.viz["/"]["figure"].values.item()
    lift_roas_axes = lift_roas_fig.axes
    lift_roas_axes[0].axvline(
        true_roas_x1,
        color="black",
        linestyle="--",
        linewidth=2,
        label="true ROAS",
    )
    lift_roas_axes[0].legend(loc="upper right")
    lift_roas_axes[0].set(title="Lift-calibrated ROAS: x1")
    lift_roas_axes[1].axvline(
        true_roas_x2,
        color="black",
        linestyle="--",
        linewidth=2,
        label="true ROAS",
    )
    lift_roas_axes[1].legend(loc="upper right")
    lift_roas_axes[1].set(title="Lift-calibrated ROAS: x2", xlabel="ROAS")
    lift_roas_fig.suptitle(
        "Lift tests resolve prior-observation conflict", fontweight="bold"
    )
    lift_roas_fig
    return


@app.cell
def _(lift_mmm, roas_psense_by_block):
    lift_roas_psense = roas_psense_by_block(lift_mmm.idata)
    lift_roas_psense
    return (lift_roas_psense,)


@app.cell
def _(lift_mmm):
    az.plot_psense_dist(
        lift_mmm.idata,
        var_names=["ROAS"],
        prior_var_names=[
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
            "business_prior",
        ],
        likelihood_var_names=["y", "lift_measurements"],
        coords={"channel": ["x1", "x2"]},
        kind="ecdf",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparing sensitivity before and after lift calibration

    The ideal pattern is not merely that ROAS moves toward the truth. We also want the prior-observation conflict to resolve: the business prior and the observational data should agree after the lift test provides additional information.
    """)
    return


@app.cell
def _(business_roas_psense, lift_roas_psense):
    psense_comparison = pd.concat(
        {
            "business_prior": business_roas_psense,
            "lift_calibrated": lift_roas_psense,
        },
        names=["model"],
    )
    psense_comparison
    return


@app.cell
def _(adjusted_mmm, lift_mmm, true_roas):
    roas_model_comparison = xr.concat(
        [
            adjusted_mmm.idata["posterior"]["ROAS"],
            lift_mmm.idata["posterior"]["ROAS"],
        ],
        dim="model",
    ).assign_coords(model=["business_prior", "lift_calibrated"])

    _fig, _axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 4), layout="constrained"
    )
    for channel, ax in zip(["x1", "x2"], _axes):
        for model_name, color in [
            ("business_prior", "C0"),
            ("lift_calibrated", "C1"),
        ]:
            values = roas_model_comparison.sel(
                model=model_name, channel=channel
            ).values.ravel()
            ax.hist(
                values,
                bins=40,
                density=True,
                alpha=0.45,
                color=color,
                label=model_name,
            )
        ax.axvline(
            true_roas.sel(channel=channel),
            color="black",
            linestyle="--",
            linewidth=2,
            label="true ROAS",
        )
        ax.set(title=f"{channel} ROAS", xlabel="ROAS")
    _axes[0].legend()
    _fig.suptitle(
        "ROAS posterior: business-prior vs lift-calibrated", fontweight="bold"
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Takeaways

    - **Prior-observation conflict is a feature, not a bug.** When the model's posterior disagrees with the business's prior expectations, it signals that the observational data alone cannot fully identify the channel effects. This is a valuable discussion point.
    - **Lift tests resolve conflict by adding information.** The lift test does not simply "confirm" one side or the other; it provides additional data that helps the model disentangle the channel effect from confounding.
    - **Validate assumptions before fitting.** Taking business priors into account before fitting the model means validating the assumptions are agreed upon between the modeling team and the business side.
    - **The goal is certainty, not proof.** The aim is not to prove the business right or wrong, but to be more certain about the results. This builds trust in the model.
    - **Suggest experiments to resolve disagreements.** When priors and observations conflict, the response should be to design experiments (lift tests, geo experiments, etc.) that provide the additional information needed to resolve the conflict.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    - **Power scaling sensitivity analysis**: Kallioinen, N., Paananen, T., Bürkner, P.-C., & Vehtari, A. (2024). Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Statistical Computing*, 34, 46. [https://link.springer.com/article/10.1007/s11222-023-10366-5](https://link.springer.com/article/10.1007/s11222-023-10366-5)

    - **Pareto-smoothed importance sampling (PSIS)**: Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling. *Journal of Machine Learning Research*, 25(72), 1–58. [https://www.jmlr.org/papers/v25/19-556.html](https://www.jmlr.org/papers/v25/19-556.html)

    - **Exploratory Analysis of Bayesian Models (EABM)**: Worked examples and tutorials on power scaling, LOO, and model diagnostics. [https://arviz-devs.github.io/EABM/](https://arviz-devs.github.io/EABM/)

    - **PyMC-Marketing ROAS case study**: The base model and lift-test methodology. [https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html)
    """)
    return


if __name__ == "__main__":
    app.run()

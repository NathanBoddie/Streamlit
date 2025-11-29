
import io
import math
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats 1.7
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Distribution Fitting Studio", layout="wide", initial_sidebar_state="expanded")

DIST_MAP = {
    "Normal (norm)": scipy.stats.norm,
    "Lognormal (lognorm)": scipy.stats.lognorm,
    "Gamma (gamma)": scipy.stats.gamma,
    "Weibull (weibull_min)":scipy.stats.weibull_min,
    "Exponential (expon)":  scipy.stats.expon,
    "Beta (beta)":  scipy.stats.beta,
    "Chi-squared (chi2)":  scipy.stats.chi2,
    "Pareto (pareto)":  scipy.stats.pareto,
    "Uniform (uniform)":  scipy.stats.uniform,
    "Fisk (fisk / log-logistic)":  scipy.stats.fisk,
    "Nakagami (nakagami)":  scipy.stats.nakagami,
    "Generalized Extreme Value (genextreme)": scipy.stats.genextreme
}


def parse_textarea_to_array(text: str) -> np.ndarray:
    """Parse newline/space/comma separated numbers into numpy array"""
    if not text:
        return np.array([])
   
    tokens = [t.strip() for t in text.replace(",", " ").split()]
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
        
            pass
    return np.array(nums, dtype=float)

def load_csv_bytes(uploaded_bytes: io.BytesIO) -> np.ndarray:
    uploaded_bytes.seek(0)
    try:
        df = pd.read_csv(uploaded_bytes, header=None)
 
        arr = df.values.flatten()
        arr = arr[~pd.isnull(arr)]
        arr = arr.astype(float)
        return arr
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return np.array([])

def hist_and_pdf_rmse(data: np.ndarray, dist, params, bins=40) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute histogram density and the pdf of distribution evaluated at bin centers.
    Return RMSE and arrays (bin_centers, hist_density, pdf_values)
    """
    hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    shape_args, loc, scale = params_to_args(params)
    try:
        pdf_vals = dist.pdf(bin_centers, *shape_args, loc=loc, scale=scale)
    except Exception:
   
        pdf_vals = np.zeros_like(bin_centers)
  
    rmse = np.sqrt(np.mean((hist_vals - pdf_vals) ** 2))
    return rmse, bin_centers, hist_vals, pdf_vals

def params_to_args(params: Dict[str, float]) -> Tuple[List[float], float, float]:
    """
    Convert params dict like {'sh0':..., 'sh1':..., 'loc':..., 'scale':...}
    into (shapes_list, loc, scale) for passing to scipy.
    """
    shape_keys = sorted([k for k in params.keys() if k.startswith("sh")], key=lambda x: int(x[2:]))
    shapes = [params[k] for k in shape_keys]
    loc = params.get("loc", 0.0)
    scale = params.get("scale", 1.0)
    return shapes, loc, scale

def fit_distribution_to_data(dist, data: np.ndarray) -> Dict[str, Any]:
    """
    Use scipy's fit; return dict of parameters with keys sh0, sh1, ... loc, scale
    and fit success flag.
    """
    res = {"success": False}
    try:
        
        fl = dist.fit(data)
       
        num_shapes = len(fl) - 2
        for i in range(num_shapes):
            res[f"sh{i}"] = float(fl[i])
        res["loc"] = float(fl[-2])
        res["scale"] = float(fl[-1])
        res["success"] = True
    except Exception as e:
        res["error"] = str(e)
    return res

def ks_test_pvalue(dist, data: np.ndarray, params: Dict[str, float]) -> float:
    """Compute KS test p-value comparing data to dist with params."""
    sh_args, loc, scale = params_to_args(params)
    # scipy.scipy.scipy.scipy.stats.kstest accepts (rvs, cdf, args=(...)) when cdf is a string name
    try:
        # prepare args tuple: shapes..., loc, scale
        args = tuple(sh_args + [loc, scale])
        # using distribution name string - works for many scipy dists
        pvalue = scipy.stats.kstest(data, dist.name, args=args).pvalue
        return float(pvalue)
    except Exception:
        # fallback: use empirical vs cdf via lambda - less efficient
        try:
            cdf_vals = dist.cdf(np.sort(data), *sh_args, loc=loc, scale=scale)
            # compute KS statistic and approximate pvalue using scipy.scipy.scipy.stats.kstwo? We'll return nan for pvalue failure
            ks_stat = np.max(np.abs(np.arange(1, len(data)+1)/len(data) - cdf_vals))
            return float(np.nan)
        except Exception:
            return float(np.nan)

def pretty_params_table(params: Dict[str, float]) -> pd.DataFrame:
    """Return DataFrame with parameter name and value"""
    rows = []
    for k, v in params.items():
        if k == "success" or k == "error":
            continue
        rows.append({"parameter": k, "value": float(v)})
    return pd.DataFrame(rows)

# -----------------------
# Sidebar: Data input
# -----------------------
st.sidebar.title("Data Input")
st.sidebar.markdown("Provide 1D numeric data either by typing/pasting, or upload a CSV file (single column or multiple columns).")
input_mode = st.sidebar.radio("Input mode", ["Type / Paste data", "Upload CSV"])

data = np.array([])

if input_mode == "Type / Paste data":
    txt = st.sidebar.text_area("Enter numbers (comma, space or newline separated)", height=180, placeholder="e.g. 1.2, 3.4, 2.1 ...")
    if txt:
        data = parse_textarea_to_array(txt)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        # read file bytes
        data = load_csv_bytes(uploaded)

# Quick sample data suggestions
with st.sidebar.expander("Example datasets"):
    st.write("Try these quick buttons to populate the data area:")
    if st.button("Random normal (n=500)"):
        data = np.random.normal(loc=0.0, scale=1.0, size=500)
    if st.button("Gamma (n=500)"):
        data = np.random.gamma(2.0, 1.0, size=500)
    if st.button("Weibull (n=500)"):
        data = np.random.weibull(1.5, size=500)

if data.size == 0:
    st.sidebar.info("No data provided yet. Use the textarea or upload a CSV, or pick an example dataset.")
else:
    st.sidebar.success(f"Loaded {data.size} data points. min={data.min():.3g}, max={data.max():.3g}")

# -----------------------
# Main UI
# -----------------------
st.title("Distribution Fitting Studio")
st.caption("Fit many SciPy distributions to your 1D data, visualize fits, and manually tweak parameters.")

tabs = st.tabs(["Data & Explore", "Auto Fit", "Manual Fit"])
# -----------------------
# Tab: Data & Explore
# -----------------------
with tabs[0]:
    st.header("Data & Exploratory View")
    if data.size == 0:
        st.info("Please supply data in the sidebar (textarea or CSV upload) or use an example dataset.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Histogram")
            bins = st.slider("Histogram bins", 10, 200, 40, key="bins_hist")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(data, bins=bins, density=True, alpha=0.6)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title("Data histogram")
            st.pyplot(fig)
        with col2:
            st.subheader("Summary")
            st.write(pd.DataFrame({
                "count": [data.size],
                "mean": [np.mean(data)],
                "std": [np.std(data, ddof=1)],
                "min": [np.min(data)],
                "25%": [np.percentile(data, 25)],
                "50% (median)": [np.median(data)],
                "75%": [np.percentile(data, 75)],
                "max": [np.max(data)]
            }).T.rename(columns={0: "value"}))
            if st.expander("Show raw values"):
                st.write(data[:500])  # limit display

# -----------------------
# Tab: Auto Fit
# -----------------------
with tabs[1]:
    st.header("Automatic Fitting")
    if data.size == 0:
        st.info("Provide data first (sidebar).")
    else:
        colA, colB = st.columns([1, 2])
        with colA:
            st.subheader("Choose distribution(s)")
            dist_options = list(DIST_MAP.keys())
            selected = st.multiselect("Select distributions to fit (multiple allowed)", dist_options,
                                      default=["Normal (norm)", "Lognormal (lognorm)", "Gamma (gamma)"])
            st.write("Fitting options:")
            bins = st.slider("Bins for histogram vs pdf comparison", 10, 200, 60)
            show_legend = st.checkbox("Show legend on plot", True)
            do_ks = st.checkbox("Compute KS-test p-value", True)
            #max_display = st.slider("Max distributions to display", 1, len(selected) if selected else 1, min(len(selected), 3))
            if len(selected) == 1:
                max_display = 1
            else:
                max_display = st.slider("Max distributions to display", 
                                        1, len(selected), min(len(selected), 3))
        with colB:
            st.subheader("Fit results")
            if not selected:
                st.warning("Pick at least one distribution to fit.")
            else:
                # Fit each and display in sub-rows
                # We'll plot data histogram once and overlay pdfs for chosen dists
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.hist(data, bins=bins, density=True, alpha=0.4, label="data histogram")
                results = []
                for name in selected:
                    dist = DIST_MAP[name]
                    fitres = fit_distribution_to_data(dist, data)
                    if not fitres.get("success", False):
                        st.error(f"Fit failed for {name}: {fitres.get('error', 'unknown')}")
                        continue
                    # compute pdf over x grid
                    sh_args, loc, scale = params_to_args(fitres)
                    x = np.linspace(np.min(data), np.max(data), 400)
                    try:
                        pdf = dist.pdf(x, *sh_args, loc=loc, scale=scale)
                    except Exception:
                        pdf = np.zeros_like(x)
                    ax.plot(x, pdf, lw=2, label=name if show_legend else None)
                    rmse, _, _, _ = hist_and_pdf_rmse(data, dist, fitres, bins=bins)
                    pval = ks_test_pvalue(dist, data, fitres) if do_ks else float("nan")
                    results.append({
                        "name": name,
                        "params": fitres,
                        "rmse": float(rmse),
                        "ks_pvalue": float(pval) if not math.isnan(pval) else None
                    })
                if show_legend:
                    ax.legend()
                ax.set_title("Data histogram with fitted PDFs")
                st.pyplot(fig)

                # Sort results by RMSE (lower better) and show top N
                if results:
                    results_sorted = sorted(results, key=lambda r: r["rmse"])
                    st.subheader("Top fits (by RMSE vs histogram)")
                    for r in results_sorted[:max_display]:
                        st.markdown(f"**{r['name']}** — RMSE = {r['rmse']:.5g}" + (f", KS p-value = {r['ks_pvalue']:.4g}" if r['ks_pvalue'] else ""))
                        # show parameters in a small table
                        st.dataframe(pretty_params_table(r["params"]), height=150)

# -----------------------
# Tab: Manual Fit
# -----------------------
with tabs[2]:
    st.header("Manual Fitting (interactive sliders)")
    if data.size == 0:
        st.info("Provide data first (sidebar).")
    else:
        left, right = st.columns([1, 1])
        with left:
            dist_name = st.selectbox("Pick a distribution to manually fit", list(DIST_MAP.keys()))
            dist = DIST_MAP[dist_name]
            # Inspect distribution shape parameter names via dist.shapes
            shapes_spec = dist.shapes  # e.g. "a, b" or None
            shape_names = []
            if shapes_spec:
                # shapes_spec is a string like "a, b"
                for i, s in enumerate([s.strip() for s in shapes_spec.split(",")]):
                    shape_names.append(s if s else f"sh{i}")
            # We'll create slider widgets for each shape + loc + scale
            st.write("Adjust parameters (use Reset to revert to automatic fit)")
            # Attempt an automatic fit to get default slider values
            default_fit = fit_distribution_to_data(dist, data)
            # Slider generation strategy:
            slider_params = {}
            # generate sliders for shape parameters
            for i, sname in enumerate(shape_names):
                key = f"manual_{dist.name}_sh{i}"
                default_val = default_fit.get(f"sh{i}", 1.0)
                # choose range heuristically relative to default and data scale
                data_scale = max(1.0, np.std(data))
                minv = -10.0 if default_val < 0 else 0.0
                maxv = max(10.0, abs(default_val) * 5 + data_scale)
                slider_params[f"sh{i}"] = st.slider(f"shape: {sname}", float(minv), float(maxv), float(default_val), step=0.01, key=key)
            # loc and scale
            key_loc = f"manual_{dist.name}_loc"
            key_scale = f"manual_{dist.name}_scale"
            default_loc = default_fit.get("loc", float(np.mean(data)))
            default_scale = default_fit.get("scale", float(np.std(data)))
            loc_val = st.slider("loc (shift)", float(np.min(data) - abs(default_loc) - 1), float(np.max(data) + abs(default_loc) + 1), float(default_loc), step=0.01, key=key_loc)
            scale_val = st.slider("scale", 1e-6, float(max(1.0, np.std(data) * 5, abs(default_scale) * 5)), float(default_scale if default_scale > 0 else np.std(data)), step=0.01, key=key_scale)
            if st.button("Reset to automatic fit"):
                # reset by clearing session state keys used above
                for k in list(st.session_state.keys()):
                    if k.startswith(f"manual_{dist.name}_"):
                        del st.session_state[k]
                st.experimental_rerun()

        with right:
            st.subheader("Manual fit visualization")
            # Build params dict
            params = {}
            for i in range(len(shape_names)):
                params[f"sh{i}"] = float(slider_params[f"sh{i}"])
            params["loc"] = float(loc_val)
            params["scale"] = float(scale_val)

            # plot histogram and pdf using these manual parameters
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(data, bins=60, density=True, alpha=0.5, label="data histogram")
            x = np.linspace(np.min(data), np.max(data), 400)
            shape_args, loc, scale = params_to_args(params)
            try:
                pdf = dist.pdf(x, *shape_args, loc=loc, scale=scale)
            except Exception:
                pdf = np.zeros_like(x)
            ax.plot(x, pdf, lw=2, label="manual pdf", linestyle="--")
            ax.set_title(f"Manual fit: {dist_name}")
            ax.legend()
            st.pyplot(fig)

            # compute metrics for manual fit
            rmse, bin_centers, hist_vals, pdf_vals = hist_and_pdf_rmse(data, dist, params, bins=60)
            pval = ks_test_pvalue(dist, data, params)
            st.markdown("**Fit metrics (manual)**")
            st.write(f"- RMSE (hist vs pdf): **{rmse:.6g}**")
            if not math.isnan(pval):
                st.write(f"- KS-test p-value: **{pval:.4g}**")
            else:
                st.write("- KS-test p-value: **n/a** (could not compute)")

            st.subheader("Parameter values")
            st.dataframe(pretty_params_table(params), height=200)







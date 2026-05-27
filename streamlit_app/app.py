"""DEL-iver Streamlit app.

A web wrapper around the `DEL_iver_results.py` analysis pipeline from
https://github.com/SztainLab/DEL-iver. Upload a DEL screening CSV, confirm
column mappings, and run the enrichment + visualization pipeline locally.
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys
import tempfile
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend; DEL_iver plots call plt.show() which blocks GUI threads.

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "DEL-iver_icon.png"

st.set_page_config(
    page_title="DEL-iver",
    page_icon=str(LOGO) if LOGO.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar entirely.
st.markdown(
    """
    <style>
      [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="collapsedControl"] { display: none !important; }
      section[data-testid="stSidebar"] { width: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _import_deliver():
    import DEL_iver as deliv  # noqa: WPS433
    _patch_windows_cache_rename()
    return deliv


def _patch_windows_cache_rename():
    """Upstream DEL_iver/utils/cache.py renames the temp parquet before closing
    its writer, which fails on Windows with PermissionError.
    """
    if os.name != "nt":
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pv
    from tqdm import tqdm
    from DEL_iver.utils import cache as _cache

    def _convert_csv_to_parquet(self, memory_per_chunk_mb):
        parquet_path = self._get_parquet_path()
        tmp_path = parquet_path.with_suffix(".tmp.parquet")
        block_size_bytes = memory_per_chunk_mb * 1024 * 1024
        read_options = pv.ReadOptions(block_size=block_size_bytes)
        writer = None
        try:
            with pv.open_csv(self.source_file, read_options=read_options) as reader:
                for batch in tqdm(reader, desc=f"Converting {self.source_file.name}", unit="Chunk"):
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)
            if writer:
                writer.close(); writer = None
            tmp_path.rename(parquet_path)
        except Exception:
            if writer:
                writer.close()
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
        return parquet_path

    _cache.CacheManager._convert_csv_to_parquet = _convert_csv_to_parquet


# ---------- helpers ----------------------------------------------------------

SMILES_PATTERN = re.compile(r"smi(le)?", re.IGNORECASE)
BB_PATTERN = re.compile(r"(bb|building.?block|synthon)", re.IGNORECASE)
LABEL_HINTS = ("bind", "hit", "label", "active", "y")
MOL_SMILES_HINTS = ("molecule", "full", "product", "compound")


def looks_like_smiles_series(s: pd.Series, sample: int = 50) -> float:
    vals = s.dropna().astype(str).head(sample)
    if len(vals) == 0:
        return 0.0
    hits = 0
    for v in vals:
        v = v.strip()
        if not v or " " in v:
            continue
        if any(c in v for c in "()[]=#@/\\"):
            hits += 1
        elif any(c in v for c in "Cc") and any(ch.isdigit() for ch in v):
            hits += 1
    return hits / len(vals)


def looks_like_binary_label(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0, 1})


def auto_detect(df: pd.DataFrame):
    bb_cols, mol_col, label_col = [], None, None
    cols = list(df.columns)
    scored = []
    for c in cols:
        s = df[c]
        is_smi = looks_like_smiles_series(s)
        name_smi = bool(SMILES_PATTERN.search(c))
        name_bb = bool(BB_PATTERN.search(c))
        scored.append((c, is_smi, name_smi, name_bb))
    mol_hint_cols = {c for c in cols if any(h in c.lower() for h in MOL_SMILES_HINTS)}
    for c, is_smi, name_smi, name_bb in scored:
        if c in mol_hint_cols:
            continue
        if (name_bb and is_smi >= 0.4) or (name_smi and is_smi >= 0.4):
            bb_cols.append(c)

    def _bb_key(c):
        m = re.search(r"(\d+)", c)
        return (int(m.group(1)) if m else 99, c)
    bb_cols = sorted(bb_cols, key=_bb_key)

    for c in cols:
        if c in bb_cols:
            continue
        if any(h in c.lower() for h in MOL_SMILES_HINTS) and looks_like_smiles_series(df[c]) >= 0.3:
            mol_col = c
            break

    for c in cols:
        if c in bb_cols or c == mol_col:
            continue
        if any(h in c.lower() for h in LABEL_HINTS) and looks_like_binary_label(df[c]):
            label_col = c
            break
    if label_col is None:
        for c in cols:
            if c in bb_cols or c == mol_col:
                continue
            if looks_like_binary_label(df[c]):
                label_col = c
                break

    return bb_cols, mol_col, label_col


@contextmanager
def chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@st.cache_data(show_spinner=False, max_entries=512)
def smiles_to_png_b64(smiles: str, size: int = 220) -> str:
    """Render a SMILES string to a base64 PNG for HTML hover tooltips."""
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    if not smiles:
        return ""
    try:
        # Strip salts (the package's draw_bb has a `remove_ions` option that does this).
        parts = smiles.split(".")
        parts.sort(key=len, reverse=True)
        mol = Chem.MolFromSmiles(parts[0])
        if mol is None:
            return ""
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _grey_info(text: str):
    st.markdown(
        f'<div style="background:#f0f0f0; color:#333; padding:10px 14px; '
        f'border-radius:4px; font-size:14px; margin:4px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


def _render_bokeh(fig) -> str:
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    html = file_html(fig, CDN)
    # Bokeh requires a non-empty tooltips template for the HoverTool callback to fire,
    # but we don't want the floating ID bubble on the dot — all info lives in the side Div.
    hide_tooltip_css = "<style>.bk-Tooltip,.bk-tooltip,.bk-overlay>.bk-Tooltip{display:none !important;}</style>"
    return html.replace("</head>", hide_tooltip_css + "</head>")


def _skip_degenerate(df, x_col, y_col=None):
    """Return True if the axis columns collapse (only one unique value), making the plot uninformative."""
    if df[x_col].nunique() <= 1:
        return True
    if y_col is not None and df[y_col].nunique() <= 1:
        return True
    return False


@st.cache_resource
def _cool_palette():
    """matplotlib's 'cool' colormap, reversed (magenta -> cyan), so low = pink, high = teal."""
    from matplotlib import colormaps
    from matplotlib.colors import rgb2hex
    cmap = colormaps.get_cmap("cool")
    return [rgb2hex(cmap(i / 255.0)) for i in range(255, -1, -1)]


# Two-color key: lower value -> pink (palette[0]), higher value -> teal (palette[-1]).
def _two_color_key():
    p = _cool_palette()
    return [p[0], p[-1]]


def build_bb_bokeh(
    bb_table: pd.DataFrame,
    id_to_smiles: dict,
    metric: str,
    thumb_cache: dict = None,
    title_prefix: str = "BB",
    y_axis_label: str = None,
    force_pbind: bool = False,
):
    """Interactive BB scatter. Tooltip popup shows numeric stats + structure image.
    Image data is stored in a single shared inline JS dict (one entry per unique SMILES);
    each row only references its SMILES key, so total HTML size is O(unique BBs) not O(rows).

    Set force_pbind=True for the bbs.png-equivalent (always plots pbind regardless of metric)."""
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColumnDataSource, CustomJS, ColorBar, LinearColorMapper, Div
    from bokeh.layouts import gridplot

    y_metric = "pbind" if force_pbind else metric

    df = bb_table.copy()
    if "type" in df.columns:
        df = df[df["type"] == "building_block"]
    if "ntotal" in df.columns:
        df = df[df["ntotal"] > 0]
    if df.empty:
        return None
    df["smiles"] = df["chemical_id"].map(lambda i: id_to_smiles.get(int(i), ""))
    df["bb_label"] = df["origin"].astype(str).str.replace("_smiles_positional_id", "", regex=False).str.upper()

    # Shared inline lookup dict.
    lookup = {k: f"data:image/png;base64,{v}" for k, v in (thumb_cache or {}).items() if v}

    js_code = """
        const idx = cb_data.index.indices;
        if (!idx || idx.length === 0) return;
        const i = idx[0];
        const smi = src.data['smiles'][i];
        const url = lookup[smi];
        const imgHtml = url
          ? `<img src="${url}" width="240" height="240"/>`
          : `<div style="color:#aaa; font-size:10px;">(no thumbnail rendered for this BB)</div>`;
        const bb_label = src.data['bb_label'][i];
        const pid = src.data['positional_id'][i];
        const pb = +src.data['pbind'][i], en = +src.data['enrichment'][i], nt = src.data['ntotal'][i];
        div.text = `
          <div style="font:12px helvetica,arial,sans-serif;">
            <div style="font-weight:bold;">${bb_label} positional_id=${pid}</div>
            <div style="font-size:11px; word-break:break-all; max-width:260px;">${smi}</div>
            <div>pbind=${pb.toFixed(4)}<br/>enrichment=${en.toFixed(3)}<br/>n=${nt}</div>
            <div style="margin-top:4px;">${imgHtml}</div>
          </div>
        `;
    """

    rows = []
    origins = sorted(df["origin"].unique())
    palette = _cool_palette()

    for origin in origins:
        sub = df[df["origin"] == origin]
        if sub.empty:
            continue
        # Each subplot gets its own ColorMapper scaled to that BB position's ntotal range,
        # since position-to-position ranges differ a lot in real DELs.
        sub_low = float(sub["ntotal"].min())
        sub_high = max(sub_low + 1.0, float(sub["ntotal"].max()))
        mapper = LinearColorMapper(palette=palette, low=sub_low, high=sub_high)

        src = ColumnDataSource(sub)
        p = figure(
            title=f"{title_prefix} {sub['bb_label'].iloc[0]}",
            x_axis_label="positional ID",
            y_axis_label=y_axis_label or y_metric,
            height=380, width=560,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.scatter(
            x="positional_id", y=y_metric, source=src, size=7,
            color={"field": "ntotal", "transform": mapper},
            alpha=0.85, line_color=None,
        )
        div = Div(text='<div style="color:#888; font:12px helvetica,arial,sans-serif; padding:8px;">Hover a dot to see its structure.</div>',
                  width=340, height=380)
        callback = CustomJS(args=dict(src=src, lookup=lookup, div=div), code=js_code)
        p.add_tools(HoverTool(tooltips=[("", "@positional_id")], callback=callback))
        p.add_layout(ColorBar(color_mapper=mapper, location=(0, 0),
                              title="n_total", label_standoff=8), "right")
        rows.append([p, div])

    return gridplot(rows, toolbar_location="right") if rows else None


def build_disynthon_pair_bokeh(
    dis_table: pd.DataFrame,
    id_to_smiles: dict,
    metric: str,
    pair: tuple,
    thumb_cache: dict = None,
    top_n: int = None,
):
    """2D scatter for ONE disynthon pair (i, j): x=BB_i positional ID, y=BB_j positional ID,
    color = metric, hover = both structures + stats. This is the user's BB2-vs-BB3 view,
    generalized so they can pick any pair from the multiselect."""
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColumnDataSource, ColorBar, LinearColorMapper, CustomJS
    from bokeh.transform import linear_cmap

    i, j = pair
    if i > j:
        i, j = j, i
    origin = f"disynthon_{i}_{j}_id"
    x_col = f"bb{i}_smiles_positional_id"
    y_col = f"bb{j}_smiles_positional_id"
    chem_i = f"bb{i}_smiles_chemical_id"
    chem_j = f"bb{j}_smiles_chemical_id"

    df = dis_table.copy()
    df = df[df["origin"] == origin]
    if "ntotal" in df.columns:
        df = df[df["ntotal"] > 0]
    df = df.dropna(subset=[x_col, y_col, chem_i, chem_j])
    if df.empty:
        return None, f"No data for disynthon_{i}_{j}."
    if _skip_degenerate(df, x_col, y_col):
        return None, f"BB{i} or BB{j} has only one unique value in this dataset, plot would be degenerate."

    df["smi_a"] = df[chem_i].astype(int).map(id_to_smiles).fillna("")
    df["smi_b"] = df[chem_j].astype(int).map(id_to_smiles).fillna("")

    n_total_pre = len(df)
    if top_n and top_n > 0 and top_n < len(df):
        df = df.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)

    # Shared inline lookup, deduped across all rows.
    lookup = {k: f"data:image/png;base64,{v}" for k, v in (thumb_cache or {}).items() if v}

    metric_vals = df[metric].dropna().unique()
    use_categorical = len(metric_vals) <= 2
    palette = _cool_palette()

    if use_categorical:
        sorted_vals = sorted(float(v) for v in metric_vals)
        # Stable string keys for factor_cmap, ordered low -> high.
        factor_strs = [f"{v:g}" for v in sorted_vals]
        df["_color_key"] = df[metric].astype(float).map(lambda v: f"{float(v):g}")
        two = _two_color_key()  # [pink, teal]
        cat_palette = [two[-1]] if len(sorted_vals) == 1 else two
        metric_min, metric_max = sorted_vals[0], sorted_vals[-1]
    else:
        metric_min, metric_max = float(df[metric].min()), float(df[metric].max())
        if metric_max == metric_min:
            metric_max = metric_min + 1.0
        cmap = linear_cmap(metric, palette, metric_min, metric_max)

    src = ColumnDataSource(df)
    p = figure(
        title=f"Disynthon BB{i} x BB{j}, colored by {metric}",
        x_axis_label=f"BB{i} positional ID",
        y_axis_label=f"BB{j} positional ID",
        height=380, width=560,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    if use_categorical:
        from bokeh.transform import factor_cmap
        color = factor_cmap("_color_key", palette=cat_palette, factors=factor_strs)
        p.scatter(
            x=x_col, y=y_col, source=src, size=9, color=color, alpha=0.85, line_color=None,
            legend_field="_color_key",
        )
        # Show actual metric values in the legend instead of "low"/"high".
        for item in p.legend.items:
            label = item.label.get("value") if isinstance(item.label, dict) else None
            if label is not None:
                item.label = {"value": f"{metric} = {label}"}
        p.legend.location = "top_left"
        p.legend.title = metric
        p.legend.click_policy = "hide"
    else:
        p.scatter(x=x_col, y=y_col, source=src, size=9, color=cmap, alpha=0.85, line_color=None)
        color_bar = ColorBar(
            color_mapper=LinearColorMapper(palette=palette, low=metric_min, high=metric_max),
            label_standoff=8, location=(0, 0), title=metric,
        )
        p.add_layout(color_bar, "right")

    from bokeh.models import Div
    from bokeh.layouts import gridplot
    js_code = f"""
        const idx = cb_data.index.indices;
        if (!idx || idx.length === 0) return;
        const i = idx[0];
        const smi_a = src.data['smi_a'][i];
        const smi_b = src.data['smi_b'][i];
        const url_a = lookup[smi_a];
        const url_b = lookup[smi_b];
        const x_v = src.data['{x_col}'][i], y_v = src.data['{y_col}'][i];
        const dis_id = src.data['positional_id'][i];
        const m_v = +src.data['{metric}'][i];
        const pb = +src.data['pbind'][i], en = +src.data['enrichment'][i], nt = src.data['ntotal'][i];
        function cell(label, smi, url) {{
          const img = url ? `<img src="${{url}}" width="145" height="145"/>` : `<div style="color:#aaa;font-size:10px;">(no thumbnail)</div>`;
          return `<td style="vertical-align:top; padding:2px;"><div style="font-size:10px; word-break:break-all; max-width:155px;"><b>${{label}}</b>: ${{smi}}</div>${{img}}</td>`;
        }}
        div.text = `
          <div style="font:12px helvetica,arial,sans-serif; max-width:420px;">
            <div style="font-weight:bold;">BB{i} (${{x_v}}) &times; BB{j} (${{y_v}}) &nbsp; disynthon_id=${{dis_id}}</div>
            <div>{metric}=${{m_v.toFixed(4)}} &nbsp; pbind=${{pb.toFixed(4)}} &nbsp; enrichment=${{en.toFixed(3)}} &nbsp; n=${{nt}}</div>
            <table><tr>${{cell('BB{i}', smi_a, url_a)}}${{cell('BB{j}', smi_b, url_b)}}</tr></table>
          </div>
        `;
    """
    div = Div(text='<div style="color:#888; font:12px helvetica,arial,sans-serif; padding:8px;">Hover a dot to see both structures.</div>',
              width=340, height=380)
    callback = CustomJS(args=dict(src=src, lookup=lookup, div=div), code=js_code)
    p.add_tools(HoverTool(tooltips=[("", "@positional_id")], callback=callback))
    msg = f"Showing top {len(df):,} of {n_total_pre:,} observed disynthon pairs." if (top_n and top_n < n_total_pre) else None
    return gridplot([[p, div]], toolbar_location="right"), msg


# ---------- Pipeline --------------------------------------------------------

@st.cache_data(show_spinner=False)
def _read_csv_preview(file_bytes: bytes, n: int = 20) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), nrows=n)


def _persist_uploaded(file_bytes: bytes, name: str) -> Path:
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = Path(tempfile.mkdtemp(prefix="deliver_"))
    workdir: Path = st.session_state["workdir"]
    target = workdir / name
    if not target.exists() or target.stat().st_size != len(file_bytes):
        target.write_bytes(file_bytes)
    return target


@st.cache_data(show_spinner=False, max_entries=4)
def run_pipeline(
    csv_path: str,
    bb_cols: tuple,
    label: str | None,
    metric: str,
    top_n: int,
    min_occ_bb: int,
    min_occ_dis: int,
    exclude_bb1: bool,
    _on_step=None,  # leading underscore -> not part of cache key
):
    """Execute the full DEL-iver results pipeline. Cached on inputs.

    `_on_step(name, dt)` is called after each step so the UI can stream progress.
    """
    deliv = _import_deliver()
    workdir = Path(csv_path).parent / "results"
    workdir.mkdir(exist_ok=True)

    def _step(name, fn):
        if _on_step is not None:
            _on_step(name, None)  # mark start
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        if _on_step is not None:
            _on_step(name, dt)
        return out

    ddr = _step("Load CSV and convert to parquet cache",
                lambda: deliv.DataReader.from_csv(
                    csv_path,
                    building_blocks=list(bb_cols),
                    output_dir=str(workdir),
                    label=label,
                ))

    _step("Enumerate building blocks and disynthons",
          lambda: deliv.enumerate_building_blocks(ddr))

    _step("Compute pbind and enrichment",
          lambda: deliv.compute_enrichment(ddr, min_occurrences=0))

    stats_path = workdir / "dataset_statistics.csv"
    def _stats():
        with chdir(workdir):
            deliv.data_set_statistics(ddr, print_output=False, write_output=str(stats_path))
    _step("Write dataset statistics", _stats)

    exclude_bb = [1] if exclude_bb1 else None
    exclude_dis = [(1, i) for i in range(2, len(bb_cols) + 1)] if exclude_bb1 else None

    top_bb = _step(f"Find top-{top_n} building blocks (by {metric})",
                   lambda: deliv.find_best_bb(
                       ddr, top_n,
                       min_occurrences=min_occ_bb,
                       sort_by=metric,
                       exclude=exclude_bb,
                   ))
    top_dis = _step(f"Find top-{top_n} disynthons (by {metric})",
                    lambda: deliv.find_best_disynthon(
                        ddr, top_n,
                        min_occurrences=min_occ_dis,
                        sort_by=metric,
                        exclude=exclude_dis,
                    ))

    figs = {}
    def _draw_bb():
        with chdir(workdir):
            deliv.draw_bb(top_bb, ddr, metric=metric, save_png_path="bb_structures.png")
        figs["bb_structures.png"] = workdir / "bb_structures.png"
    _step("Render top-BB structure grid", _draw_bb)

    def _draw_dis():
        with chdir(workdir):
            deliv.draw_disynthons(top_dis, ddr, metric=metric, save_png_path="disynthon_structures.png")
        figs["disynthon_structures.png"] = workdir / "disynthon_structures.png"
    _step("Render top-disynthon structure grid", _draw_dis)

    def _plot_dis():
        with chdir(workdir):
            # IMPORTANT: kwarg is `mode`, not `metric`. Without this, the plot
            # always shows pbind regardless of the user's metric choice.
            deliv.plot_disynthons(ddr, mode=metric, elev=15, azim=35,
                                  min_occurrences=min_occ_bb, output_path="disynthons.png")
        figs["disynthons.png"] = workdir / "disynthons.png"
    _step(f"Plot disynthon landscape ({metric})", _plot_dis)

    def _plot_bb():
        with chdir(workdir):
            # plot_bb does NOT take a metric arg; it always plots pbind.
            deliv.plot_bb(ddr, output_path="bbs.png", exclude_bb1=exclude_bb1)
        figs["bbs.png"] = workdir / "bbs.png"
    _step("Plot BB pbind by position", _plot_bb)

    # Load raw enrichment tables for interactive plots.
    import pyarrow.parquet as pq
    from DEL_iver.utils.cache import CacheNames
    bb_enrich = pq.read_table(ddr.cache._get_output_path(CacheNames.COMPUTE, "bb_enrichment")).to_pandas()
    dis_enrich = pq.read_table(ddr.cache._get_output_path(CacheNames.COMPUTE, "disynthon_enrichment")).to_pandas()
    id_table = pq.read_table(ddr.cache._get_output_path(CacheNames.BB_DICTIONARIES, "id_to_smiles")).to_pandas()
    id_to_smiles = {int(r["id"]): r["smiles"] for _, r in id_table.iterrows()}

    def _to_df(obj):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, "to_pandas"):
            try:
                return obj.to_pandas()
            except Exception:
                pass
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None

    return {
        "stats_csv": str(stats_path),
        "figures": {k: str(v) for k, v in figs.items() if v.exists()},
        "top_bb": _to_df(top_bb),
        "top_disynthons": _to_df(top_dis),
        "bb_enrichment": bb_enrich,
        "disynthon_enrichment": dis_enrich,
        "id_to_smiles": id_to_smiles,
        "metric": metric,
        "bb_cols": list(bb_cols),
    }


# ---------- Tabs ------------------------------------------------------------

tab_about, tab_instructions, tab_run = st.tabs(["About", "Instructions", "Run analysis"])

# ---------- ABOUT ------------------------------------------------------------

with tab_about:
    cols = st.columns([1, 4])
    with cols[0]:
        if LOGO.exists():
            st.image(str(LOGO), width=160)
    with cols[1]:
        st.title("DEL-iver")
        st.markdown(
            "A Python package for processing DNA-encoded library (DEL) data, "
            "training ML models, and picking hits from make-on-demand libraries."
        )

    st.markdown(
        """
        ### What this app does

        This is a web version of the `DEL_iver_results.py` script. Given a DEL
        screening CSV with one row per compound and one SMILES column per
        building block, the app will:

        1. Convert the CSV to a memory-efficient parquet cache on first load.
        2. Enumerate every unique building block and every disynthon pair.
        3. Compute **pbind** (raw hit rate) and **enrichment** (relative to the
           per-position baseline) for every building block and disynthon.
        4. Rank the top building blocks and disynthons.
        5. Render publication-quality structure grids and distribution plots,
           plus interactive Bokeh plots with structure-image hover.

        ### Resource note

        This app runs the full DEL-iver pipeline **in-process on the machine
        hosting Streamlit.** Large CSVs use local CPU and RAM (RDKit +
        matplotlib + pyarrow). Expect a few minutes for files with >100k rows.
        When you run `streamlit run app.py` locally, that means **your
        machine.** When the app is hosted on Streamlit Community Cloud, that
        means **the cloud host's** machine. Either way, nothing is sent to a
        third party.

        ### Citation

        Dolorfino, M.; Perez, D. S.; Fu, Y.; Lin, S.-H.; McCarty, S.;
        O'Meara, M. J.; Sztain, T. *Assessing the Generalizability of Machine
        Learning and Physics Methods for DNA-Encoded Libraries.* bioRxiv,
        April 19, 2026. [doi.org/10.64898/2026.04.18.719394](https://doi.org/10.64898/2026.04.18.719394)

        ### Source

        [github.com/SztainLab/DEL-iver](https://github.com/SztainLab/DEL-iver)

        ### What's not in the web app (yet)

        The CLI package also ships `DEL_iver_models.py` (model training and
        inference) and `DEL_iver_analogs.py` (analog proposal). Those are
        heavier workloads that benefit from a local GPU; this app focuses on
        the results pipeline.
        """
    )

# ---------- INSTRUCTIONS -----------------------------------------------------

with tab_instructions:
    st.title("Instructions")
    st.markdown(
        """
        ### 1. Prepare your CSV

        The CSV must have:

        - **One row per compound.**
        - **One column per building block**, each holding a valid SMILES string.
        - Optionally a binary **label** column with `0` (non-hit) or `1` (hit).
        - Optionally a **molecule SMILES** column for the assembled product.

        DEL-iver supports any number of building block positions (2, 3, 4, ...
        and beyond). The order of the building block columns you pick **matters**
        because it defines the BB1/BB2/BB3/... positions for the analysis.

        Example:

        | bb1_smiles | bb2_smiles | bb3_smiles | binds |
        |------------|------------|------------|-------|
        | CCCO       | c1ccccc1   | CCN        | 1     |
        | CCO        | Cc1ccccc1  | CCC        | 0     |

        ### 2. Upload it on the **Run analysis** tab

        Drag and drop or browse to your CSV. A preview of the first rows
        appears so you can sanity-check.

        ### 3. Confirm column mapping

        The app auto-detects:

        - Which columns hold building-block SMILES (and in what order).
        - Which column, if any, is the binary hit/miss label.
        - Which column, if any, is the full molecule SMILES.

        Auto-detection is heuristic. Different DELs use different column naming
        conventions, and the column order in your CSV may not match the
        intended BB1, BB2, BB3 order. **Always review the mapping** before
        running. Re-order building blocks into the correct position; remove
        anything that isn't really a building block.

        ### 4. Set options and run

        - **Metric**: `pbind` (raw hit rate) or `enrichment` (relative to the
          per-position baseline). `enrichment` requires a label column.
          Note: this metric controls top-N ranking, structure-grid labels,
          and the disynthon landscape. The BB scatter (`plot_bb`) in DEL-iver
          always plots pbind; the interactive Bokeh BB plot below it respects
          your metric choice.
        - **Top-N**: how many building blocks / disynthons to plot.
        - **Min occurrences**: drop building blocks seen fewer than N times.
        - **Exclude BB1**: many DELs use BB1 as a scaffold/linker; excluding
          it from the top-N often gives more chemically meaningful hits.

        Click **Run pipeline**. The pipeline log includes per-step timings.

        ### 5. Download outputs

        Each figure has an individual download button. A separate
        **Download all results (.zip)** button bundles every figure, table,
        and the pipeline log.

        ### Troubleshooting

        - **"Column X does not contain valid SMILES"**: re-check the mapping;
          you may have selected an ID column by accident.
        - **"Label column must be 0/1"**: convert your label to integers
          before upload, or unselect it and re-run with `metric=pbind`.
        - **Out of memory on a large CSV**: reduce the file size, or run the
          CLI script (`DEL_iver_results.py`) on a workstation with more RAM.
        """
    )

# ---------- RUN --------------------------------------------------------------

with tab_run:
    st.title("Run analysis")

    upload = st.file_uploader(
        "Upload your DEL screening CSV",
        type=["csv"],
        help="One row per compound. One SMILES column per building block. Optional 0/1 label.",
    )

    if upload is None:
        _grey_info("Upload a CSV to get started, or try the example below.")
        example_path = Path(__file__).parent / "data" / "example.csv"
        if example_path.exists():
            with example_path.open("rb") as fh:
                st.download_button(
                    "Download example.csv (sEH screen, 20k rows)",
                    data=fh.read(),
                    file_name="example.csv",
                    mime="text/csv",
                )
        st.stop()

    file_bytes = upload.getvalue()
    try:
        preview_df = _read_csv_preview(file_bytes)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

    st.subheader("Preview")
    st.caption(f"Showing the first {len(preview_df)} rows of `{upload.name}`.")
    st.dataframe(preview_df, use_container_width=True, height=240)

    st.subheader("Confirm column mapping")
    st.caption(
        "Auto-detection is heuristic. Re-order building-block columns so the "
        "order matches your intended BB1, BB2, BB3, ... positions, and unselect "
        "anything that isn't a real building-block SMILES column."
    )

    all_cols = list(preview_df.columns)
    auto_bb, _auto_mol, auto_label = auto_detect(preview_df)

    bb_cols = st.multiselect(
        "Building block SMILES columns (order matters: this is BB1, BB2, ...)",
        options=all_cols,
        default=auto_bb,
        help="Pick at least 2. Drag to reorder. Each must contain valid SMILES.",
    )

    label_options = ["(none)"] + [c for c in all_cols if c not in bb_cols]
    label_default = auto_label if (auto_label and auto_label in label_options) else "(none)"
    label_choice = st.selectbox(
        "Label column (binary 0/1, optional, but recommended)",
        options=label_options,
        index=label_options.index(label_default),
        help="Required to rank by enrichment.",
    )

    issues = []
    if len(bb_cols) < 2:
        issues.append("Pick at least 2 building-block columns.")
    for c in bb_cols:
        frac = looks_like_smiles_series(preview_df[c])
        if frac < 0.3:
            issues.append(
                f"Column **`{c}`** doesn't look like SMILES "
                f"(only {frac:.0%} of sampled values match). Did you mean a different column?"
            )
    if label_choice != "(none)":
        if not looks_like_binary_label(preview_df[label_choice]):
            issues.append(
                f"Label column **`{label_choice}`** is not strictly 0/1. "
                f"Convert to integers or pick a different column."
            )

    for msg in issues:
        st.warning(msg)

    with st.expander("Mapping summary", expanded=True):
        summary_rows = [{"Role": f"BB{i+1}", "Column": c} for i, c in enumerate(bb_cols)]
        if label_choice != "(none)":
            summary_rows.append({"Role": "label", "Column": label_choice})
        st.table(pd.DataFrame(summary_rows))

    st.subheader("Pipeline options")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric = st.selectbox(
            "Metric", ["pbind", "enrichment"],
            help="enrichment requires a label column.",
        )
    with c2:
        top_n = st.number_input("Top-N to plot", min_value=1, max_value=50, value=10, step=1)
    with c3:
        min_occ_bb = st.number_input("Min BB occurrences", min_value=0, value=30, step=1)
    with c4:
        min_occ_dis = st.number_input("Min disynthon occurrences", min_value=0, value=1, step=1)

    exclude_bb1 = st.checkbox(
        "Exclude BB1 from top-N (recommended if BB1 is a scaffold/linker position)",
        value=True,
    )

    if metric == "enrichment" and label_choice == "(none)":
        st.error("enrichment requires a label column. Pick one or switch to pbind.")
        ready = False
    elif issues:
        ready = st.checkbox("Run anyway (I know what I'm doing)", value=False)
    else:
        ready = True

    st.caption(
        "The first run on a new CSV converts it to a parquet cache; "
        "subsequent runs on the same file reuse the cache and are much faster."
    )

    if st.button("Run pipeline", type="primary", disabled=not (ready and bb_cols)):
        csv_path = _persist_uploaded(file_bytes, upload.name)
        try:
            with st.status("Running DEL-iver pipeline...", expanded=True) as status:
                t_total = time.perf_counter()
                current = {"name": None}

                def _on_step(name, dt):
                    if dt is None:
                        current["name"] = name
                        status.update(label=f"Running: {name}")
                        status.write(f"... {name}")
                    else:
                        status.write(f"  done in {dt:.2f}s")

                result = run_pipeline(
                    csv_path=str(csv_path),
                    bb_cols=tuple(bb_cols),
                    label=(None if label_choice == "(none)" else label_choice),
                    metric=metric,
                    top_n=int(top_n),
                    min_occ_bb=int(min_occ_bb),
                    min_occ_dis=int(min_occ_dis),
                    exclude_bb1=exclude_bb1,
                    _on_step=_on_step,
                )
                # Pre-render BB thumbnails inside the same status so the user sees what's left.
                status.update(label="Rendering structure thumbnails...")
                status.write("... rendering structure thumbnails for interactive plots")
                t0 = time.perf_counter()
                all_smiles = sorted({s for s in result["id_to_smiles"].values() if s})
                thumb_cache = {smi: smiles_to_png_b64(smi) for smi in all_smiles}
                status.write(f"  rendered {len(thumb_cache)} structures in {time.perf_counter() - t0:.2f}s")

                result["_total"] = time.perf_counter() - t_total
                st.session_state["last_result"] = result
                st.session_state["thumb_cache"] = thumb_cache
                st.session_state["thumb_cache_key"] = (result["stats_csv"],)
                status.update(label=f"Pipeline finished in {result['_total']:.1f}s.",
                              state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {exc}")
            st.code(traceback.format_exc())

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.header("Results")
        st.caption(f"Metric used: **{result['metric']}**")

        if result.get("top_bb") is not None:
            st.subheader("Top building blocks")
            st.dataframe(result["top_bb"], use_container_width=True)
        if result.get("top_disynthons") is not None:
            st.subheader("Top disynthons")
            st.dataframe(result["top_disynthons"], use_container_width=True)

        stats_path = Path(result["stats_csv"])
        if stats_path.exists():
            st.subheader("Dataset statistics")
            stats_df = pd.read_csv(stats_path)
            st.dataframe(stats_df, use_container_width=True)

        # Bundle everything into a single zip
        figures = result.get("figures", {})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if stats_path.exists():
                zf.write(stats_path, arcname="dataset_statistics.csv")
            for name, path in figures.items():
                p = Path(path)
                if p.exists():
                    zf.write(p, arcname=name)
            if result.get("top_bb") is not None:
                zf.writestr("top_bb.csv", result["top_bb"].to_csv(index=False))
            if result.get("top_disynthons") is not None:
                zf.writestr("top_disynthons.csv", result["top_disynthons"].to_csv(index=False))

        # Visuals-only zip
        vbuf = io.BytesIO()
        if figures:
            with zipfile.ZipFile(vbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, path in figures.items():
                    p = Path(path)
                    if p.exists():
                        zf.write(p, arcname=name)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download all results (.zip)",
                data=buf.getvalue(),
                file_name="deliver_results.zip",
                mime="application/zip",
                type="primary",
                key="dl_all",
            )
        with dl2:
            if figures:
                st.download_button(
                    "Download all visuals only (.zip)",
                    data=vbuf.getvalue(),
                    file_name="deliver_visuals.zip",
                    mime="application/zip",
                    key="dl_visuals_only",
                )

        # ---- Static figures ----
        st.subheader("Static figures")
        try:
            bb_unique_per_pos = (result["bb_enrichment"]
                                 .query("type == 'building_block'")
                                 .groupby("origin")["chemical_id"].nunique())
            scaffold_positions = [o.replace("_smiles_positional_id", "").upper()
                                  for o in bb_unique_per_pos.index if bb_unique_per_pos[o] <= 1]
            if scaffold_positions:
                _grey_info(
                    f"Note: {', '.join(scaffold_positions)} has only one unique SMILES in this dataset "
                    "(scaffold/linker position). 3D disynthon subplots involving it will look like a "
                    "single line; the interactive pair view below lets you pick non-scaffold pairs."
                )
        except Exception:
            pass
        if not figures:
            _grey_info("No figures were rendered.")
        else:
            captions = {
                "bb_structures.png": "Top building-block structures",
                "disynthon_structures.png": "Top disynthon structures",
                "bbs.png": "Building-block pbind by position (always pbind, per upstream package)",
                "disynthons.png": f"Disynthon landscape ({result['metric']})",
            }
            cols2 = st.columns(2)
            for i, (name, path) in enumerate(figures.items()):
                with cols2[i % 2]:
                    st.image(path, caption=captions.get(name, name), use_container_width=True)
                    with open(path, "rb") as fh:
                        st.download_button(
                            f"Download {name}",
                            data=fh.read(),
                            file_name=name,
                            mime="image/png",
                            key=f"dl_{name}",
                        )

        # ---- Interactive plots ----
        st.subheader("Interactive plots (hover for structure)")

        n_bb_positions = len(result.get("bb_cols", ())) or 3

        # Thumbnails were pre-rendered inside the Run-pipeline status panel.
        # Recompute on cache miss (e.g., user reopened the page on a stale session).
        thumb_cache = st.session_state.get("thumb_cache")
        if not thumb_cache or st.session_state.get("thumb_cache_key") != (result["stats_csv"],):
            all_smiles = sorted({s for s in result["id_to_smiles"].values() if s})
            thumb_cache = {smi: smiles_to_png_b64(smi) for smi in all_smiles}
            st.session_state["thumb_cache"] = thumb_cache
            st.session_state["thumb_cache_key"] = (result["stats_csv"],)

        import streamlit.components.v1 as components

        # ---- BB position scatter (uses chosen metric on Y) ----
        try:
            bb_fig = build_bb_bokeh(result["bb_enrichment"], result["id_to_smiles"],
                                    metric=result["metric"], thumb_cache=thumb_cache)
            if bb_fig is not None:
                st.markdown(f"**Building blocks: positional ID vs {result['metric']}** (one subplot per BB position; color = ntotal)")
                components.html(_render_bokeh(bb_fig), height=1220, scrolling=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not render BB scatter: {exc}")
            st.code(traceback.format_exc())

        # ---- BB pair heatmap-style scatter ----
        st.markdown("**Disynthon pair view**")
        st.caption(
            "Pick two BB positions. The plot scatters every observed disynthon for that pair "
            "with x and y as their positional IDs and color as the chosen metric. "
            "Hover any point for both component structures plus the metric value."
        )
        pair_cols = st.columns([1, 1, 1])
        with pair_cols[0]:
            i_sel = st.selectbox("X axis BB position", options=list(range(1, n_bb_positions + 1)),
                                 index=min(1, n_bb_positions - 1), key="pair_i")
        with pair_cols[1]:
            j_sel = st.selectbox("Y axis BB position", options=list(range(1, n_bb_positions + 1)),
                                 index=min(2, n_bb_positions - 1), key="pair_j")

        # Smart default for top-N: if the metric column for this pair has exactly two
        # distinct values (the common "hits at max, everyone else low" case), default
        # to the count of rows at the max. Otherwise fall back to 100.
        def _smart_default_top_n():
            if i_sel == j_sel:
                return 100
            lo, hi = sorted((i_sel, j_sel))
            origin = f"disynthon_{lo}_{hi}_id"
            sub = result["disynthon_enrichment"]
            sub = sub[sub["origin"] == origin]
            if "ntotal" in sub.columns:
                sub = sub[sub["ntotal"] > 0]
            col = sub[result["metric"]].dropna()
            if col.empty:
                return 100
            vals = col.unique()
            if len(vals) == 2:
                return int((col == col.max()).sum())
            return 100

        default_top = _smart_default_top_n()
        with pair_cols[2]:
            # Including pair in the widget key so changing pair re-applies the smart default.
            top_n_pair = st.number_input(
                f"Show top X by {result['metric']}",
                min_value=1, max_value=50000, value=default_top, step=10,
                key=f"pair_top_n_{i_sel}_{j_sel}_{result['metric']}",
                help="Default is auto-set: count of rows at the max metric value when only "
                     "two distinct values exist, else 100. Edit freely.",
            )

        if i_sel == j_sel:
            _grey_info("Pick two different BB positions.")
        else:
            try:
                pair_fig, pair_msg = build_disynthon_pair_bokeh(
                    result["disynthon_enrichment"], result["id_to_smiles"],
                    metric=result["metric"], pair=(i_sel, j_sel), thumb_cache=thumb_cache,
                    top_n=int(top_n_pair),
                )
                if pair_fig is not None:
                    if pair_msg:
                        st.caption(pair_msg)
                    components.html(_render_bokeh(pair_fig), height=420, scrolling=True)
                else:
                    _grey_info(pair_msg or "No data for this pair.")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not render pair plot: {exc}")
                st.code(traceback.format_exc())



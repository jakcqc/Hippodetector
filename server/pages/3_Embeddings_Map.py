from __future__ import annotations

import importlib
import json
from pathlib import Path
import random
import sys
import textwrap
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

np: Any = None
pd: Any = None
px = None
go = None
umap = None
sk_cluster = None
sk_neighbors = None
sk_metrics = None

try:
    from app import HIPPO_ICON_PATH, inject_css, normalize_text
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import HIPPO_ICON_PATH, inject_css, normalize_text


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_FILE = DATA_DIR / "press_release_embeddings_member_index.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def render_hero(show_loader: bool = False) -> None:
    loader_html = ""
    if show_loader:
        loader_html = """
        <div class="embeddings-loader-overlay">
          <div class="embeddings-loader-card">
            <div class="embeddings-loader-title">Loading Embeddings Map...</div>
            <div class="embeddings-loader-track"><div class="embeddings-loader-bar"></div></div>
          </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="embeddings-hero-wrap">
          <div class="hero">
            <p class="hero-title">Embeddings Map</p>
            <p class="hero-sub">Project and compare press release embeddings across multiple members. Ctrl + Click on a point to pull up the original press release. </p>
          </div>
          {loader_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_embeddings_page_css() -> None:
    st.markdown(
        """
        <style>
          .embeddings-hero-wrap {
            position: relative;
          }
          .embeddings-loader-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 6;
            pointer-events: none;
          }
          .embeddings-loader-card {
            min-width: 340px;
            max-width: 560px;
            background: rgba(20, 20, 20, 0.86);
            border: 1px solid #5a5a5a;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38);
          }
          .embeddings-loader-title {
            color: #ffffff;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
          }
          .embeddings-loader-track {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: #3a3a3a;
            overflow: hidden;
          }
          .embeddings-loader-bar {
            height: 100%;
            width: 45%;
            border-radius: 999px;
            background: linear-gradient(90deg, #d9d9d9 0%, #ffffff 100%);
            animation: embeddings-loader-sweep 1.25s ease-in-out infinite;
          }
          @keyframes embeddings-loader-sweep {
            0% { transform: translateX(-120%); }
            100% { transform: translateX(230%); }
          }
          [data-testid="stTextInput"] label,
          [data-testid="stNumberInput"] label,
          [data-testid="stSelectbox"] label,
          [data-testid="stMultiselect"] label,
          [data-testid="stMultiSelect"] label {
            color: #ffffff !important;
          }
          div[data-baseweb="select"] > div {
            background: #111111 !important;
            border-color: #444444 !important;
          }
          div[data-baseweb="select"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          div[data-baseweb="popover"],
          div[data-baseweb="menu"],
          ul[role="listbox"],
          div[role="option"] {
            background: #111111 !important;
            color: #ffffff !important;
            border-color: #444444 !important;
          }
          [role="listbox"],
          [role="listbox"] *,
          [role="option"],
          [role="option"] * {
            background: #111111 !important;
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          [data-baseweb="popover"] > div,
          [data-baseweb="popover"] > div > div,
          [data-baseweb="menu"] > div,
          [data-baseweb="menu"] > div > div {
            background: #111111 !important;
            color: #ffffff !important;
          }
          ul[role="listbox"] li,
          ul[role="listbox"] li > div,
          div[role="option"],
          div[role="option"] > div {
            background: #111111 !important;
            color: #ffffff !important;
          }
          ul[role="listbox"] li[aria-selected="true"],
          ul[role="listbox"] li:hover,
          div[role="option"][aria-selected="true"],
          div[role="option"]:hover {
            background: #1f1f1f !important;
            color: #ffffff !important;
          }
          div[data-baseweb="menu"] [role="button"],
          div[data-baseweb="menu"] [role="button"] * {
            background: #111111 !important;
            color: #ffffff !important;
            fill: #ffffff !important;
          }
          div[data-baseweb="menu"] [role="button"]:hover,
          div[data-baseweb="menu"] [role="button"][aria-selected="true"] {
            background: #1f1f1f !important;
            color: #ffffff !important;
          }
          div[role="option"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
          }
          [data-baseweb="tag"] {
            background: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
          }
          [data-baseweb="tag"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
          }
          [data-baseweb="input"] > div,
          [data-baseweb="textarea"] > div {
            background: #111111 !important;
            border: 1px solid #444444 !important;
          }
          [data-baseweb="input"] input,
          [data-baseweb="textarea"] textarea {
            background: #111111 !important;
            color: #ffffff !important;
            caret-color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
          }
          [data-baseweb="input"] input::placeholder,
          [data-baseweb="textarea"] textarea::placeholder {
            color: #bdbdbd !important;
            opacity: 1 !important;
          }
          [data-baseweb="input"] > div:hover,
          [data-baseweb="input"] > div:focus-within,
          [data-baseweb="textarea"] > div:hover,
          [data-baseweb="textarea"] > div:focus-within,
          div[data-baseweb="select"] > div:hover,
          div[data-baseweb="select"] > div:focus-within {
            border-color: #7a7a7a !important;
            box-shadow: 0 0 0 1px #7a7a7a55 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_optional_modules() -> dict[str, Any]:
    modules: dict[str, Any] = {
        "px": None,
        "go": None,
        "umap": None,
        "sk_cluster": None,
        "sk_neighbors": None,
        "sk_metrics": None,
    }
    try:
        modules["px"] = importlib.import_module("plotly.express")
    except ModuleNotFoundError:
        pass
    try:
        modules["go"] = importlib.import_module("plotly.graph_objects")
    except ModuleNotFoundError:
        pass
    try:
        modules["umap"] = importlib.import_module("umap")
    except ModuleNotFoundError:
        pass
    try:
        modules["sk_cluster"] = importlib.import_module("sklearn.cluster")
        modules["sk_neighbors"] = importlib.import_module("sklearn.neighbors")
        modules["sk_metrics"] = importlib.import_module("sklearn.metrics")
    except ModuleNotFoundError:
        pass
    return modules


@st.cache_resource(show_spinner=False)
def load_core_data_modules() -> dict[str, Any]:
    modules: dict[str, Any] = {"np": None, "pd": None}
    try:
        modules["np"] = importlib.import_module("numpy")
    except ModuleNotFoundError:
        pass
    try:
        modules["pd"] = importlib.import_module("pandas")
    except ModuleNotFoundError:
        pass
    return modules


def member_label(member: dict[str, Any]) -> str:
    name = str(member.get("name") or member.get("bioguideId") or "-")
    bioguide_id = str(member.get("bioguideId") or "-")
    party = str(member.get("partyName") or "-")
    state = str(member.get("state") or "-")
    count = int(member.get("totalEmbeddingCount") or 0)
    return f"{name} ({bioguide_id}) · {party} · {state} · {count} embeds"


@st.cache_data(show_spinner=False)
def load_embeddings_index(index_path: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    payload = load_json(Path(index_path))
    if not isinstance(payload, dict):
        return {"members": []}
    members = payload.get("members")
    if not isinstance(members, list):
        members = []
    normalized_members = [m for m in members if isinstance(m, dict)]
    return {"members": normalized_members, "payload": payload}


@st.cache_data(show_spinner=False)
def load_member_file(file_path: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    payload = load_json(Path(file_path))
    if not isinstance(payload, dict):
        return {"rows": [], "vectors": np.zeros((0, 2), dtype=float)}
    releases = payload.get("pressReleases")
    if not isinstance(releases, list):
        releases = []

    rows: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    for idx, item in enumerate(releases):
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        try:
            vec = [float(v) for v in emb]
        except Exception:
            continue
        vectors.append(vec)
        rows.append(
            {
                "idx": idx,
                "title": str(item.get("title") or f"Press release {idx + 1}"),
                "date": str(item.get("date") or item.get("publishedTime") or "-"),
                "url": str(item.get("url") or ""),
                "textSource": str(item.get("textSource") or ""),
                "bodyCharCount": int(item.get("bodyCharCount") or 0),
            }
        )

    if not vectors:
        return {"rows": rows, "vectors": np.zeros((0, 2), dtype=float)}
    return {"rows": rows, "vectors": np.asarray(vectors, dtype=np.float32)}


def choose_file_for_member(member: dict[str, Any], source_filter: str) -> dict[str, Any] | None:
    files = member.get("files")
    if not isinstance(files, list):
        return None
    candidates = [f for f in files if isinstance(f, dict)]
    if source_filter != "Any":
        candidates = [f for f in candidates if str(f.get("sourceFolder") or "") == source_filter]
    if not candidates:
        return None
    candidates = sorted(
        candidates,
        key=lambda f: (int(f.get("embeddingCount") or 0), int(f.get("mtimeNs") or 0)),
        reverse=True,
    )
    return candidates[0]


def project_vectors(
    vectors: np.ndarray,
    method: str,
    *,
    random_state: int | None,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if vectors.shape[0] == 1:
        return np.asarray([[0.0, 0.0]], dtype=float)

    if method == "UMAP" and umap is not None:
        n_neighbors = min(max(2, int(umap_n_neighbors)), max(2, vectors.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            metric="cosine",
            random_state=random_state,
            n_jobs=-1,
        )
        return reducer.fit_transform(vectors)

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(float)


def fuzzy_c_means(
    x: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    n_init: int = 1,
    max_iter: int = 150,
    error: float = 1e-5,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n_samples = int(x.shape[0])
    n_clusters = int(n_clusters)
    if n_samples == 0:
        empty = np.zeros((0, n_clusters), dtype=float)
        return np.zeros((n_clusters, int(x.shape[1])), dtype=float), np.zeros((0,), dtype=int), 0.0, empty

    rng = np.random.default_rng(random_state)
    m = float(m)
    if m <= 1.0:
        raise ValueError("Fuzzy C-Means requires m > 1.0")
    p = 2.0 / (m - 1.0)

    def init_centers_kmeanspp() -> np.ndarray:
        n_features = int(x.shape[1])
        centers = np.empty((n_clusters, n_features), dtype=float)
        first_idx = int(rng.integers(0, n_samples))
        centers[0] = x[first_idx]
        min_dist_sq = np.sum((x - centers[0]) ** 2, axis=1).astype(float, copy=False)
        for c in range(1, n_clusters):
            total = float(min_dist_sq.sum())
            if not np.isfinite(total) or total <= 1e-18:
                next_idx = int(rng.integers(0, n_samples))
            else:
                probs = min_dist_sq / total
                next_idx = int(rng.choice(n_samples, p=probs))
            centers[c] = x[next_idx]
            dist_sq = np.sum((x - centers[c]) ** 2, axis=1).astype(float, copy=False)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        return centers

    def update_membership(dist: np.ndarray) -> np.ndarray:
        dist = np.clip(dist, 1e-12, None)
        inv = dist ** (-p)
        denom = np.clip(inv.sum(axis=1, keepdims=True), 1e-12, None)
        return inv / denom

    best_obj = float("inf")
    best_centers = np.zeros((n_clusters, int(x.shape[1])), dtype=float)
    best_u = np.zeros((n_samples, n_clusters), dtype=float)

    for _ in range(max(1, int(n_init))):
        centers = init_centers_kmeanspp()
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        u = update_membership(dist)
        prev = u.copy()

        for _ in range(max_iter):
            um = np.power(np.clip(u, 1e-12, None), m)
            centers = (um.T @ x) / np.clip(um.sum(axis=0, keepdims=True).T, 1e-12, None)
            dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            u = update_membership(dist)
            if np.linalg.norm(u - prev) < error:
                break
            prev = u.copy()

        dist_sq = np.square(np.clip(dist, 1e-12, None))
        obj = float((np.power(np.clip(u, 1e-12, None), m) * dist_sq).sum())
        if obj < best_obj:
            best_obj = obj
            best_centers = centers.astype(float, copy=True)
            best_u = u.astype(float, copy=True)

    fpc = float((best_u * best_u).sum() / max(1, n_samples))
    labels = np.argmax(best_u, axis=1)
    return best_centers, labels, fpc, best_u


def predict_fuzzy_membership(x: np.ndarray, centers: np.ndarray, m: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    dist = np.clip(dist, 1e-12, None)
    p = 2.0 / (m - 1.0)
    ratio = (dist[:, :, None] / dist[:, None, :]) ** p
    membership = 1.0 / np.clip(ratio.sum(axis=2), 1e-12, None)
    labels = np.argmax(membership, axis=1)
    return membership, labels


def xie_beni_index(x: np.ndarray, centers: np.ndarray, membership: np.ndarray, m: float = 2.0) -> float:
    if centers.shape[0] < 2:
        return float("inf")
    um = np.power(np.clip(membership, 1e-12, None), m)
    dist_sq = np.square(np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2))
    numerator = float((um * dist_sq).sum())
    center_dist_sq = np.square(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2))
    center_dist_sq[center_dist_sq == 0.0] = np.inf
    min_center_sep_sq = float(center_dist_sq.min())
    if min_center_sep_sq <= 0.0 or not np.isfinite(min_center_sep_sq):
        return float("inf")
    return numerator / max(1.0, float(x.shape[0])) / min_center_sep_sq


def estimate_best_fcm_clusters(
    x: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int | None,
    criterion: str,
    m: float,
) -> tuple[int, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    best_k = min_k
    best_fpc = -1.0
    best_mpc = -1.0
    best_xb = float("inf")
    for k in range(min_k, max_k + 1):
        if k >= x.shape[0]:
            break
        centers, _, fpc, membership = fuzzy_c_means(x, n_clusters=k, random_state=random_state, m=m)
        xb = xie_beni_index(x, centers, membership, m=m)
        mpc = float((fpc - (1.0 / k)) / (1.0 - (1.0 / k))) if k > 1 else float("nan")
        results.append({"k": k, "fpc": fpc, "mpc": mpc, "xie_beni": xb})
        if criterion == "Xie-Beni (min)":
            if xb < best_xb:
                best_xb = xb
                best_k = k
        elif criterion == "MPC (max)":
            if mpc > best_mpc:
                best_mpc = mpc
                best_k = k
        else:
            if fpc > best_fpc:
                best_fpc = fpc
                best_k = k
    return best_k, results


def kmeans_cluster(
    x: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if sk_cluster is None:
        raise RuntimeError("scikit-learn is required for KMeans clustering.")
    model = sk_cluster.KMeans(
        n_clusters=int(n_clusters),
        init="k-means++",
        n_init="auto",
        max_iter=300,
        random_state=random_state,
    )
    labels = model.fit_predict(x)
    centers = model.cluster_centers_
    inertia = float(model.inertia_)
    return labels, centers, inertia


def estimate_best_kmeans_clusters(
    x: np.ndarray,
    min_k: int,
    max_k: int,
    *,
    random_state: int | None,
    criterion: str,
    silhouette_sample_size: int,
) -> tuple[int, list[dict[str, Any]]]:
    if sk_metrics is None:
        raise RuntimeError("scikit-learn is required for KMeans clustering metrics.")

    results: list[dict[str, Any]] = []
    best_k = min_k
    best_score = -float("inf")
    best_min_score = float("inf")
    for k in range(min_k, max_k + 1):
        if k >= x.shape[0]:
            break
        labels, _, inertia = kmeans_cluster(x, k, random_state=random_state)
        row: dict[str, Any] = {"k": int(k), "inertia": float(inertia)}

        try:
            row["calinski_harabasz"] = float(sk_metrics.calinski_harabasz_score(x, labels))
        except Exception:
            row["calinski_harabasz"] = float("nan")
        try:
            row["davies_bouldin"] = float(sk_metrics.davies_bouldin_score(x, labels))
        except Exception:
            row["davies_bouldin"] = float("nan")

        try:
            sample_size = min(int(silhouette_sample_size), int(x.shape[0]))
            row["silhouette_cosine"] = float(
                sk_metrics.silhouette_score(
                    x,
                    labels,
                    metric="cosine",
                    sample_size=sample_size,
                    random_state=random_state,
                )
            )
        except Exception:
            row["silhouette_cosine"] = float("nan")

        results.append(row)

        if criterion == "Davies-Bouldin (min)":
            score = row["davies_bouldin"]
            if score == score and score < best_min_score:
                best_min_score = score
                best_k = k
        elif criterion == "Calinski-Harabasz (max)":
            score = row["calinski_harabasz"]
            if score == score and score > best_score:
                best_score = score
                best_k = k
        else:
            score = row["silhouette_cosine"]
            if score == score and score > best_score:
                best_score = score
                best_k = k

    return best_k, results


def knn_graph_clusters(
    x: np.ndarray,
    n_neighbors: int,
    *,
    mutual: bool,
    max_distance: float | None,
    min_cluster_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if sk_neighbors is None:
        raise RuntimeError("scikit-learn is required for KNN clustering.")

    n_samples = int(x.shape[0])
    if n_samples == 0:
        return np.zeros((0,), dtype=int), {"n_clusters": 0, "noise_points": 0, "max_distance": max_distance}
    if n_samples == 1:
        return np.asarray([0], dtype=int), {"n_clusters": 1, "noise_points": 0, "max_distance": max_distance}

    k = min(max(2, int(n_neighbors)), max(2, n_samples - 1))
    nn = sk_neighbors.NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    if max_distance is None:
        mask = np.ones_like(distances, dtype=bool)
        used_max_distance = None
    else:
        used_max_distance = float(max_distance)
        mask = distances <= used_max_distance

    neighbor_sets: list[set[int]] = []
    for i in range(n_samples):
        neighbors = indices[i][mask[i]]
        neighbor_sets.append(set(int(v) for v in neighbors))

    parent = list(range(n_samples))
    rank = [0] * n_samples

    def find_root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find_root(a), find_root(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n_samples):
        for j in neighbor_sets[i]:
            if j <= i:
                continue
            if mutual and i not in neighbor_sets[j]:
                continue
            union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n_samples):
        r = find_root(i)
        components.setdefault(r, []).append(i)

    large_components = [members for members in components.values() if len(members) >= int(min_cluster_size)]
    large_components.sort(key=len, reverse=True)

    labels = np.full((n_samples,), -1, dtype=int)
    for cluster_idx, members in enumerate(large_components):
        for i in members:
            labels[i] = int(cluster_idx)

    meta = {
        "n_clusters": int(len(large_components)),
        "noise_points": int((labels < 0).sum()),
        "max_distance": used_max_distance,
        "k_neighbors": int(k),
        "mutual": bool(mutual),
        "min_cluster_size": int(min_cluster_size),
    }
    return labels, meta


def assign_member_styles(member_ids: list[str], style_seed: int | None) -> dict[str, dict[str, Any]]:
    symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "star",
        "hexagon",
        "pentagon",
    ]
    rng = random.Random(style_seed) if style_seed is not None else random.Random()
    shuffled = symbols[:]
    rng.shuffle(shuffled)

    styles: dict[str, dict[str, Any]] = {}
    for idx, member_id in enumerate(sorted(set(member_ids))):
        symbol = shuffled[idx % len(shuffled)]
        size = rng.randint(7, 13)
        styles[member_id] = {"symbol": symbol, "size": size}
    return styles


def normalize_rows_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def member_last_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return "-"
    if "," in raw:
        return raw.split(",", 1)[0].strip()
    return raw.split()[-1]


def wrap_hover_title(title: str, width: int = 56, max_lines: int = 3) -> str:
    clean = " ".join((title or "").split())
    if not clean:
        return "-"
    lines = textwrap.wrap(clean, width=width, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return "-"
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,:;") + "..."
    return "<br>".join(lines)


def format_hover_float(value: Any, decimals: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if number != number or number in (float("inf"), float("-inf")):
        return "-"
    return f"{number:.{decimals}f}"


def format_hover_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def compact_hover_url(url: str, max_chars: int = 72) -> str:
    clean = " ".join((url or "").split())
    if not clean:
        return "-"
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def build_embedding_hover_template(*, expanded: bool, show_cluster_stats: bool, show_membership: bool) -> str:
    lines = [
        "<b>%{customdata[0]}</b>",
        "Member: %{customdata[1]}",
        "Date: %{customdata[2]}",
    ]
    if expanded:
        if show_cluster_stats:
            lines.extend(
                [
                    "Cluster: %{customdata[3]}",
                    "Confidence: %{customdata[4]}",
                    "Uncertainty: %{customdata[5]}",
                ]
            )
        lines.extend(
            [
                "Source: %{customdata[6]}",
                "Characters: %{customdata[7]}",
            ]
        )
        if show_membership:
            lines.append("Top memberships: %{customdata[8]}")
        lines.extend(
            [
                "URL: %{customdata[10]}",
                "<i>Ctrl+click this point to open the URL in a new tab.</i>",
            ]
        )
    return "<br>".join(lines) + "<extra></extra>"


def render_ctrl_click_plotly_chart(fig: Any, *, height: int, key: str, url_customdata_index: int = 9) -> None:
    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )
    container_id = f"embeddings-map-{key}"
    host_id_json = json.dumps(container_id)
    component_html = f"""
    <div id={host_id_json}>
      {chart_html}
    </div>
    <script>
      (function() {{
        const host = document.getElementById({host_id_json});
        if (!host) return;

        const bindClickHandler = () => {{
          const graphDiv = host.querySelector(".plotly-graph-div");
          if (!graphDiv || graphDiv.__hippoCtrlOpenBound) {{
            return Boolean(graphDiv && graphDiv.__hippoCtrlOpenBound);
          }}

          graphDiv.__hippoCtrlOpenBound = true;
          graphDiv.on("plotly_click", function(evt) {{
            if (!evt || !evt.event) return;
            const withModifier = Boolean(evt.event.ctrlKey || evt.event.metaKey);
            if (!withModifier) return;

            const firstPoint = (evt.points && evt.points[0]) ? evt.points[0] : null;
            if (!firstPoint || !Array.isArray(firstPoint.customdata)) return;
            const url = firstPoint.customdata[{int(url_customdata_index)}];
            if (!url || typeof url !== "string") return;
            window.open(url, "_blank", "noopener,noreferrer");
          }});
          return true;
        }};

        if (bindClickHandler()) return;

        let tries = 0;
        const timer = setInterval(() => {{
          tries += 1;
          if (bindClickHandler() || tries > 80) {{
            clearInterval(timer);
          }}
        }}, 60);
      }})();
    </script>
    """
    components.html(component_html, height=height, scrolling=False)


def build_membership_summary(row: pd.Series, k: int) -> str:
    parts: list[tuple[str, float]] = []
    for idx in range(k):
        col = f"membership_C{idx}"
        value = float(row.get(col) or 0.0)
        parts.append((f"C{idx}", value))
    parts.sort(key=lambda item: item[1], reverse=True)
    top = parts[:3]
    return ", ".join(f"{cluster}:{value:.3f}" for cluster, value in top)


def main() -> None:
    page_icon: str | Path = "🦛"
    if HIPPO_ICON_PATH.exists():
        page_icon = HIPPO_ICON_PATH
    st.set_page_config(page_title="Hippodetector Embeddings Map", page_icon=page_icon, layout="wide")
    inject_css()
    inject_embeddings_page_css()
    hero_slot = st.empty()
    with hero_slot.container():
        render_hero(show_loader=True)
    core_modules = load_core_data_modules()
    global np, pd
    np = core_modules["np"]
    pd = core_modules["pd"]
    if np is None or pd is None:
        with hero_slot.container():
            render_hero(show_loader=False)
        st.error("`numpy` and `pandas` are required for this page. Install with: `uv pip install numpy pandas`")
        return

    modules = load_optional_modules()
    global px, go, umap, sk_cluster, sk_neighbors, sk_metrics
    px = modules["px"]
    go = modules["go"]
    umap = modules["umap"]
    sk_cluster = modules["sk_cluster"]
    sk_neighbors = modules["sk_neighbors"]
    sk_metrics = modules["sk_metrics"]

    if px is None:
        with hero_slot.container():
            render_hero(show_loader=False)
        st.error("`plotly` is required for this page. Install with: `uv pip install plotly`")
        return

    if not INDEX_FILE.exists():
        with hero_slot.container():
            render_hero(show_loader=False)
        st.error(
            f"Missing embeddings index: {INDEX_FILE}. "
            "Build it first with `uv run python dataset/build_press_release_embeddings_index.py`."
        )
        return

    index_data = load_embeddings_index(str(INDEX_FILE), INDEX_FILE.stat().st_mtime_ns)
    with hero_slot.container():
        render_hero(show_loader=False)

    members = index_data["members"]
    if not members:
        st.info("No members with embeddings in index.")
        return

    members_by_id = {str(m.get("bioguideId") or "").upper(): m for m in members}
    all_parties = sorted({str(m.get("partyName") or "Unknown") for m in members})
    all_states = sorted({str(m.get("state") or "Unknown") for m in members})

    st.subheader("Navigator")
    nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([1.8, 1.1, 1.0, 1.2, 1.0, 1.0])
    with nav1:
        member_query = st.text_input("Find member", "", placeholder="Try: B001316 or Burlison")
    with nav2:
        selected_party = st.selectbox("Party", ["All parties"] + all_parties, index=0)
    with nav3:
        selected_state = st.selectbox("State", ["All states"] + all_states, index=0)
    with nav4:
        source_filter = st.selectbox("Source folder", ["Any", "press_release_embeddings_1", "press_release_embeddings_2"], index=0)
    with nav5:
        method = st.selectbox("Projection", ["UMAP", "PCA fallback"], index=0)
    with nav6:
        cluster_mode = st.selectbox("Cluster", ["None", "Fuzzy C-Means", "KMeans", "KNN (graph)"], index=1)

    nav7, nav8, nav9 = st.columns([1.0, 0.9, 1.4])
    with nav7:
        max_points = int(st.number_input("Max points", min_value=100, max_value=50000, value=6000, step=100))
    with nav8:
        use_seed = st.toggle("Use seed", value=False, help="Off keeps UMAP fully parallel and unseeded.")
    with nav9:
        random_state: int | None = None
        if use_seed:
            random_state = int(st.number_input("Sampling/clustering seed", min_value=0, max_value=9999, value=42, step=1))
        else:
            st.caption("UMAP runs without a seed for parallelism.")

    umap_col1, umap_col2, umap_col3 = st.columns([1.0, 1.0, 1.2])
    with umap_col1:
        umap_neighbors = int(st.slider("UMAP n_neighbors", min_value=5, max_value=80, value=15, step=1, disabled=(method != "UMAP")))
    with umap_col2:
        umap_min_dist = float(st.slider("UMAP min_dist", min_value=0.0, max_value=0.9, value=0.1, step=0.05, disabled=(method != "UMAP")))
    with umap_col3:
        st.caption("Best practice: cluster in embedding space; projection is for visualization.")

    cluster_k_mode = "Auto"
    cluster_criterion = "Xie-Beni (min)"
    manual_k = 4
    fuzzifier_m = 1.5
    kmeans_criterion = "Silhouette (cosine, max)"
    silhouette_sample_size = 1000
    knn_neighbors = 20
    knn_mutual = True
    knn_dist_threshold: float | dict[str, float] | None = {"percentile": 90.0}
    knn_min_cluster_size = 8

    if cluster_mode == "Fuzzy C-Means":
        cluster_col1, cluster_col2, cluster_col3, cluster_col4 = st.columns([1.0, 1.0, 1.1, 1.2])
        with cluster_col1:
            cluster_k_mode = st.selectbox("Cluster k", ["Auto", "Manual"], index=0)
        with cluster_col2:
            cluster_criterion = st.selectbox("Auto criterion", ["Xie-Beni (min)", "MPC (max)", "FPC (max)"], index=0)
            if cluster_criterion == "FPC (max)":
                st.caption("FPC is biased toward smaller k; MPC/Xie-Beni usually pick more informative k.")
        with cluster_col3:
            manual_k = int(
                st.number_input(
                    "Manual k",
                    min_value=2,
                    max_value=20,
                    value=4,
                    step=1,
                    disabled=(cluster_k_mode != "Manual"),
                )
            )
        with cluster_col4:
            fuzzifier_m = float(
                st.slider(
                    "Fuzziness (m)",
                    min_value=1.1,
                    max_value=2.4,
                    value=1.1,
                    step=0.05,
                    help="Lower values (≈1.1–1.3) behave more like KMeans; higher values can collapse to uniform memberships on dense embedding clouds.",
                )
            )
    elif cluster_mode == "KMeans":
        km1, km2, km3, km4 = st.columns([1.0, 1.1, 1.1, 1.0])
        with km1:
            cluster_k_mode = st.selectbox("Cluster k", ["Auto", "Manual"], index=0, key="kmeans_k_mode")
        with km2:
            kmeans_criterion = st.selectbox(
                "Auto criterion",
                ["Silhouette (cosine, max)", "Calinski-Harabasz (max)", "Davies-Bouldin (min)"],
                index=0,
                key="kmeans_criterion",
            )
        with km3:
            manual_k = int(
                st.number_input(
                    "Manual k",
                    min_value=2,
                    max_value=50,
                    value=8,
                    step=1,
                    disabled=(cluster_k_mode != "Manual"),
                    key="kmeans_manual_k",
                )
            )
        with km4:
            silhouette_sample_size = int(
                st.number_input(
                    "Silhouette sample",
                    min_value=200,
                    max_value=5000,
                    value=1000,
                    step=100,
                    help="Caps silhouette scoring cost for Auto k.",
                )
            )
    elif cluster_mode == "KNN (graph)":
        kn1, kn2, kn3, kn4 = st.columns([1.0, 1.1, 1.1, 1.1])
        with kn1:
            knn_neighbors = int(st.number_input("k neighbors", min_value=3, max_value=80, value=20, step=1))
        with kn2:
            knn_mutual = st.toggle("Mutual edges", value=True, help="Best practice: mutual kNN reduces spurious links.")
        with kn3:
            knn_max_dist_mode = st.selectbox("Max cosine distance", ["None", "Percentile", "Manual"], index=1)
        with kn4:
            knn_min_cluster_size = int(st.number_input("Min cluster size", min_value=2, max_value=200, value=8, step=1))

        if knn_max_dist_mode == "Manual":
            knn_dist_threshold = float(st.slider("Distance threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01))
        elif knn_max_dist_mode == "Percentile":
            knn_dist_percentile = float(st.slider("Distance percentile", min_value=50.0, max_value=99.5, value=90.0, step=0.5))
            knn_dist_threshold = {"percentile": knn_dist_percentile}
        else:
            knn_dist_threshold = None

    hover_col1, hover_col2 = st.columns([1.0, 1.5])
    with hover_col1:
        expanded_hover = st.toggle("Expanded hover", value=False)
    with hover_col2:
        include_membership_hover = st.toggle(
            "Membership context in hover",
            value=True,
            disabled=(cluster_mode != "Fuzzy C-Means"),
            help="Show top cluster-membership values in hover.",
        )
    uncertainty_col1, uncertainty_col2 = st.columns([1.0, 1.8])
    with uncertainty_col1:
        uncertainty_filter_enabled = st.toggle(
            "Filter by uncertainty",
            value=False,
            disabled=(cluster_mode != "Fuzzy C-Means"),
            help="Keep only points at or above the selected uncertainty threshold.",
        )
    with uncertainty_col2:
        uncertainty_min_threshold = float(
            st.slider(
                "Min uncertainty (>=)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                disabled=(cluster_mode != "Fuzzy C-Means" or not uncertainty_filter_enabled),
            )
        )

    filtered_members: list[dict[str, Any]] = []
    for member in members:
        party_ok = selected_party == "All parties" or str(member.get("partyName") or "Unknown") == selected_party
        state_ok = selected_state == "All states" or str(member.get("state") or "Unknown") == selected_state
        search_blob = normalize_text(
            " ".join(
                [
                    str(member.get("bioguideId") or ""),
                    str(member.get("name") or ""),
                    str(member.get("partyName") or ""),
                    str(member.get("state") or ""),
                ]
            )
        )
        query_ok = True
        if member_query.strip():
            query_ok = normalize_text(member_query) in search_blob
        if party_ok and state_ok and query_ok:
            filtered_members.append(member)

    filtered_labels = [member_label(m) for m in filtered_members]
    label_to_id = {member_label(m): str(m.get("bioguideId") or "").upper() for m in filtered_members}
    all_labels = [member_label(m) for m in members]
    all_label_to_id = {member_label(m): str(m.get("bioguideId") or "").upper() for m in members}

    selector_col, pin_col = st.columns([2.2, 1.8])
    with selector_col:
        selected_labels = st.multiselect(
            "Members in context (filtered)",
            filtered_labels,
            default=[],
            placeholder="Select one or more members",
        )
    with pin_col:
        pin_add_label = st.selectbox("Pin member (manual add)", all_labels, index=0)
        pin_add_clicked = st.button("Pin Member")

    if "embeddings_pinned_member_ids" not in st.session_state:
        st.session_state["embeddings_pinned_member_ids"] = []

    pinned_ids: list[str] = list(st.session_state["embeddings_pinned_member_ids"])
    if pin_add_clicked:
        pin_id = all_label_to_id.get(pin_add_label)
        if pin_id and pin_id not in pinned_ids:
            pinned_ids.append(pin_id)
            st.session_state["embeddings_pinned_member_ids"] = pinned_ids

    pinned_options = [m for m in members if str(m.get("bioguideId") or "").upper() in pinned_ids]
    pinned_labels = [member_label(m) for m in pinned_options]
    pinned_keep_labels = st.multiselect(
        "Pinned members (keep/remove)",
        pinned_labels,
        default=pinned_labels,
        placeholder="Pinned set",
    )
    st.session_state["embeddings_pinned_member_ids"] = [all_label_to_id[label] for label in pinned_keep_labels if label in all_label_to_id]
    if st.button("Clear pins"):
        st.session_state["embeddings_pinned_member_ids"] = []

    selected_ids = {label_to_id[label] for label in selected_labels if label in label_to_id}
    selected_ids.update(st.session_state["embeddings_pinned_member_ids"])
    selected_ids = {s for s in selected_ids if s in members_by_id}

    if not selected_ids:
        st.info("Select at least one member from context or pin a member to visualize embeddings.")
        return

    process_bar = st.progress(0, text="Preparing embedding visualization...")
    all_rows: list[dict[str, Any]] = []
    vectors_list: list[np.ndarray] = []
    skipped_members: list[str] = []
    source_files_used: list[str] = []
    source_folders_used: list[str] = []
    selected_id_list = sorted(selected_ids)
    total_selected = max(1, len(selected_id_list))

    for idx, member_id in enumerate(selected_id_list, start=1):
        pct = int(5 + (idx / total_selected) * 45)
        process_bar.progress(min(50, pct), text=f"Loading embeddings for {member_id} ({idx}/{total_selected})...")
        member = members_by_id[member_id]
        file_info = choose_file_for_member(member, source_filter)
        if file_info is None:
            skipped_members.append(member_id)
            continue
        rel_path = str(file_info.get("path") or "")
        source_folder = str(file_info.get("sourceFolder") or "")
        file_path = DATA_DIR / rel_path
        if not file_path.exists():
            skipped_members.append(member_id)
            continue
        loaded = load_member_file(str(file_path), file_path.stat().st_mtime_ns)
        rows = loaded["rows"]
        vectors = loaded["vectors"]
        if vectors.shape[0] == 0:
            skipped_members.append(member_id)
            continue
        source_files_used.append(rel_path)
        if source_folder:
            source_folders_used.append(source_folder)
        for row in rows:
            row["memberId"] = member_id
            row["memberName"] = str(member.get("name") or member_id)
            row["partyName"] = str(member.get("partyName") or "Unknown")
            row["state"] = str(member.get("state") or "Unknown")
            row["memberLabel"] = f"{row['memberName']} ({member_id})"
        all_rows.extend(rows)
        vectors_list.append(vectors)

    if not vectors_list:
        st.info("No embeddings available for selected members with the current source filter.")
        return

    unique_dims = sorted({int(v.shape[1]) for v in vectors_list if int(v.ndim) == 2})
    if len(unique_dims) > 1:
        st.error(
            "Embedding dimensions differ across selected members/files. "
            "This typically means mixed embedding models. "
            f"Detected dimensions: {unique_dims}. "
            "Select a single `Source folder` so all members use the same embedding run."
        )
        return

    unique_source_folders = sorted({s for s in source_folders_used if s})
    if source_filter == "Any" and len(unique_source_folders) > 1:
        st.warning(
            "Mixed `Source folder` values detected across selected members. "
            "Best practice: use a single embedding run/model for meaningful clustering. "
            f"Used: {', '.join(unique_source_folders)}"
        )

    vectors = np.vstack(vectors_list)
    rows = all_rows
    if vectors.shape[0] > max_points:
        process_bar.progress(58, text="Sampling points for faster visualization...")
        sample_rng = random.Random(random_state) if random_state is not None else random.Random()
        sample_indices = sorted(sample_rng.sample(range(vectors.shape[0]), max_points))
        vectors = vectors[sample_indices]
        rows = [rows[i] for i in sample_indices]

    analysis_vectors = normalize_rows_l2(vectors)

    used_method = method
    if method == "UMAP" and umap is None:
        used_method = "PCA fallback"
        st.warning("UMAP is not installed in this environment; using PCA fallback.")

    process_bar.progress(68, text=f"Projecting embeddings ({used_method})...")
    projected = project_vectors(
        analysis_vectors,
        used_method,
        random_state=(random_state if use_seed else None),
        umap_n_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
    )
    df = pd.DataFrame(rows)
    df["x"] = projected[:, 0]
    df["y"] = projected[:, 1]

    style_map = assign_member_styles(df["memberId"].astype(str).tolist(), random_state)
    df["memberStyleSymbol"] = df["memberId"].astype(str).map(lambda mid: style_map[mid]["symbol"])
    df["memberStyleSize"] = df["memberId"].astype(str).map(lambda mid: style_map[mid]["size"])

    fcm_best_k = None
    fcm_centers: np.ndarray | None = None
    membership_matrix: np.ndarray | None = None
    fcm_results: list[dict[str, Any]] = []
    kmeans_best_k = None
    kmeans_results: list[dict[str, Any]] = []
    knn_results: dict[str, Any] | None = None
    df["cluster_confidence"] = np.nan
    df["cluster_uncertainty"] = np.nan
    if cluster_mode == "Fuzzy C-Means" and len(df) >= 4:
        process_bar.progress(82, text="Running Fuzzy C-Means clustering...")
        cluster_space = analysis_vectors
        max_k = min(12, max(3, int(np.sqrt(len(df)))))
        if cluster_k_mode == "Manual":
            fcm_best_k = min(manual_k, max(2, len(df) - 1))
        else:
            fcm_best_k, fcm_results = estimate_best_fcm_clusters(
                cluster_space,
                2,
                max_k,
                random_state,
                cluster_criterion,
                fuzzifier_m,
            )
        fcm_centers, labels, _, membership = fuzzy_c_means(
            cluster_space,
            n_clusters=fcm_best_k,
            random_state=random_state,
            m=fuzzifier_m,
        )
        membership_matrix = membership
        df["cluster"] = [f"C{int(v)}" for v in labels]
        confidence = membership.max(axis=1)
        df["cluster_confidence"] = confidence
        df["cluster_uncertainty"] = 1.0 - confidence
        for cluster_idx in range(membership.shape[1]):
            df[f"membership_C{cluster_idx}"] = membership[:, cluster_idx]
        df["membership_top3"] = df.apply(lambda row: build_membership_summary(row, fcm_best_k), axis=1)

        center_dists = np.linalg.norm(fcm_centers[:, None, :] - fcm_centers[None, :, :], axis=2).astype(float)
        center_dists[np.eye(int(fcm_best_k), dtype=bool)] = np.inf
        min_center_sep = float(center_dists.min()) if center_dists.size else float("inf")
        uniform_floor = 1.0 / float(fcm_best_k)
        mean_conf = float(confidence.mean())
        if min_center_sep < 1e-6 or mean_conf <= uniform_floor + 0.02:
            st.warning(
                "FCM collapsed to near-uniform memberships (cluster centers became coincident). "
                "Try lowering `m` (≈1.1–1.3), using `Xie-Beni` or `MPC` for Auto k, or switching to `KMeans`."
            )
    elif cluster_mode == "KMeans":
        if sk_cluster is None or sk_metrics is None:
            st.error("`scikit-learn` is required for KMeans clustering. Install with: `uv pip install scikit-learn`")
            return
        process_bar.progress(82, text="Running KMeans clustering...")
        cluster_space = analysis_vectors
        max_k = min(30, max(3, int(np.sqrt(len(df))) + 8))
        if cluster_k_mode == "Manual":
            kmeans_best_k = min(int(manual_k), max(2, int(len(df) - 1)))
        else:
            kmeans_best_k, kmeans_results = estimate_best_kmeans_clusters(
                cluster_space,
                2,
                max_k,
                random_state=random_state,
                criterion=str(kmeans_criterion),
                silhouette_sample_size=int(silhouette_sample_size),
            )
        labels, _, inertia = kmeans_cluster(cluster_space, kmeans_best_k, random_state=random_state)
        df["cluster"] = [f"C{int(v)}" for v in labels]
        df["membership_top3"] = "-"
        st.caption(f"KMeans inertia: {inertia:.2f}")
    elif cluster_mode == "KNN (graph)":
        if sk_neighbors is None:
            st.error("`scikit-learn` is required for KNN clustering. Install with: `uv pip install scikit-learn`")
            return
        process_bar.progress(82, text="Running KNN graph clustering...")
        cluster_space = analysis_vectors
        max_dist_value: float | None = None
        if isinstance(knn_dist_threshold, dict) and "percentile" in knn_dist_threshold:
            pct = float(knn_dist_threshold["percentile"])
            k = min(max(2, int(knn_neighbors)), max(2, int(cluster_space.shape[0] - 1)))
            nn = sk_neighbors.NearestNeighbors(n_neighbors=k + 1, metric="cosine")
            nn.fit(cluster_space)
            dists, _ = nn.kneighbors(cluster_space, return_distance=True)
            dists = dists[:, 1:].ravel()
            max_dist_value = float(np.quantile(dists, pct / 100.0))
        elif isinstance(knn_dist_threshold, (int, float)):
            max_dist_value = float(knn_dist_threshold)

        labels, knn_results = knn_graph_clusters(
            cluster_space,
            int(knn_neighbors),
            mutual=bool(knn_mutual),
            max_distance=max_dist_value,
            min_cluster_size=int(knn_min_cluster_size),
        )
        df["cluster"] = ["noise" if int(v) < 0 else f"C{int(v)}" for v in labels]
        df["membership_top3"] = "-"
    else:
        df["cluster"] = "unclustered"
        df["membership_top3"] = "-"

    process_bar.progress(95, text="Building charts...")

    member_visibility_map = {
        f"{member_last_name(str(name))} ({mid})": str(mid)
        for mid, name in sorted(df[["memberId", "memberName"]].drop_duplicates().itertuples(index=False), key=lambda t: str(t[1]))
    }
    member_visibility_options = list(member_visibility_map.keys())
    visible_members_key = "embedding_visible_members"
    existing_visible_members = st.session_state.get(visible_members_key)
    if not isinstance(existing_visible_members, list):
        st.session_state[visible_members_key] = member_visibility_options.copy()
    else:
        retained = [label for label in existing_visible_members if label in member_visibility_options]
        newly_available = [label for label in member_visibility_options if label not in retained]
        st.session_state[visible_members_key] = retained + newly_available

    selected_member_visibility = st.multiselect(
        "Show members",
        member_visibility_options,
        default=member_visibility_options,
        key=visible_members_key,
    )
    visible_member_ids = {member_visibility_map[label] for label in selected_member_visibility if label in member_visibility_map}
    df_view = df[df["memberId"].astype(str).isin(visible_member_ids)] if visible_member_ids else df.iloc[0:0]

    visible_clusters: list[str] = []
    if cluster_mode != "None":
        cluster_options = sorted(df["cluster"].astype(str).unique().tolist())
        visible_clusters_key = "embedding_visible_clusters"
        existing_visible_clusters = st.session_state.get(visible_clusters_key)
        if not isinstance(existing_visible_clusters, list):
            st.session_state[visible_clusters_key] = cluster_options.copy()
        else:
            retained_clusters = [label for label in existing_visible_clusters if label in cluster_options]
            new_clusters = [label for label in cluster_options if label not in retained_clusters]
            st.session_state[visible_clusters_key] = retained_clusters + new_clusters

        visible_clusters = st.multiselect(
            "Show clusters",
            cluster_options,
            default=cluster_options,
            key=visible_clusters_key,
        )
        if visible_clusters:
            df_view = df_view[df_view["cluster"].astype(str).isin(visible_clusters)]
        else:
            df_view = df_view.iloc[0:0]
    if uncertainty_filter_enabled and cluster_mode == "Fuzzy C-Means":
        df_view = df_view[
            df_view["cluster_uncertainty"].notna()
            & (df_view["cluster_uncertainty"].astype(float) >= float(uncertainty_min_threshold))
        ]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Members selected", str(len(selected_ids)))
    c2.metric("Points shown", str(len(df_view)))
    c3.metric("Projection", used_method)
    if cluster_mode == "Fuzzy C-Means":
        c4.metric("FCM best k", str(fcm_best_k) if fcm_best_k is not None else "-")
    elif cluster_mode == "KMeans":
        c4.metric("KMeans k", str(kmeans_best_k) if kmeans_best_k is not None else "-")
    elif cluster_mode == "KNN (graph)":
        c4.metric("KNN clusters", str(knn_results["n_clusters"]) if knn_results else "-")
    else:
        c4.metric("Clusters", "-")
    c5.metric("Files used", str(len(source_files_used)))

    if df_view.empty:
        process_bar.progress(100, text="Embeddings and clustering ready.")
        process_bar.empty()
        st.info("No points visible with current member/cluster visibility filters.")
        return

    color_field = "cluster" if cluster_mode != "None" else "memberLabel"
    df_view = df_view.copy()
    df_view["hover_member_party"] = df_view["memberName"].astype(str) + " (" + df_view["partyName"].astype(str) + ")"
    df_view["hover_title_wrapped"] = df_view["title"].astype(str).map(wrap_hover_title)
    df_view["hover_date"] = df_view["date"].astype(str).replace("", "-")
    df_view["hover_cluster"] = df_view["cluster"].astype(str).replace("", "-")
    df_view["hover_confidence"] = df_view["cluster_confidence"].map(format_hover_float)
    df_view["hover_uncertainty"] = df_view["cluster_uncertainty"].map(format_hover_float)
    df_view["hover_source"] = df_view["textSource"].astype(str).replace("", "-")
    df_view["hover_chars"] = df_view["bodyCharCount"].map(format_hover_int)
    df_view["hover_membership"] = df_view["membership_top3"].astype(str).replace("", "-")
    df_view["hover_url_display"] = df_view["url"].astype(str).map(compact_hover_url)

    hover_customdata_cols = [
        "hover_title_wrapped",
        "hover_member_party",
        "hover_date",
        "hover_cluster",
        "hover_confidence",
        "hover_uncertainty",
        "hover_source",
        "hover_chars",
        "hover_membership",
        "url",
        "hover_url_display",
        "title",
    ]
    show_cluster_stats = cluster_mode != "None"
    show_membership_hover = bool(include_membership_hover and cluster_mode == "Fuzzy C-Means" and fcm_best_k is not None)
    hover_template = build_embedding_hover_template(
        expanded=bool(expanded_hover),
        show_cluster_stats=show_cluster_stats,
        show_membership=show_membership_hover,
    )

    member_counts = (
        df_view.groupby(["memberId", "memberName", "memberLabel"], as_index=False)
        .size()
        .rename(columns={"size": "points"})
    )
    legend_rows: list[dict[str, Any]] = []
    for row in member_counts.to_dict(orient="records"):
        member_id = str(row["memberId"])
        style = style_map.get(member_id, {"symbol": "circle", "size": 8})
        legend_rows.append(
            {
                "last_name": member_last_name(str(row["memberName"])),
                "shape": str(style["symbol"]),
                "size": int(style["size"]),
                "points": int(row["points"]),
            }
        )
    legend_rows = sorted(legend_rows, key=lambda r: str(r["last_name"]))
    legend_text = " | ".join(
        f"{row['last_name']} ({row['shape']}, {row['points']})"
        for row in legend_rows
    )
    st.caption(f"Member shape legend: {legend_text}")

    fig = px.scatter(
        df_view,
        x="x",
        y="y",
        color=color_field,
        symbol="memberStyleSymbol",
        size="memberStyleSize",
        size_max=14,
        custom_data=hover_customdata_cols,
        title="Press Release Embeddings Projection",
        template="plotly_white",
    )
    fig.update_traces(
        marker={"opacity": 0.85, "line": {"width": 0.6, "color": "#1a1a1a"}},
        hovertemplate=hover_template,
    )
    fig.update_layout(
        height=700,
        legend_title_text=color_field,
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.98)",
            "bordercolor": "#1f2937",
            "font": {"color": "#111827", "size": 12},
        },
    )
    render_ctrl_click_plotly_chart(fig, height=730, key="projection", url_customdata_index=9)
    st.caption("Ctrl+click (or Cmd+click on macOS) a point to open the press release in a new tab.")
    process_bar.progress(100, text="Embeddings and clustering ready.")
    process_bar.empty()

    show_uncertainty = st.toggle(
        "Show cluster uncertainty plot",
        value=False,
        help="Color encodes uncertainty (0=confident cluster assignment, 1=uncertain).",
    )
    if show_uncertainty:
        if cluster_mode != "Fuzzy C-Means" or fcm_best_k is None:
            st.info("Enable `Fuzzy C-Means` clustering to view uncertainty.")
        elif go is None:
            st.info("`plotly.graph_objects` is unavailable; cannot render boundary uncertainty plot.")
        else:
            if membership_matrix is not None and fcm_best_k is not None:
                visible_mask = df.index.isin(df_view.index)
                membership_view = membership_matrix[visible_mask]
                if cluster_mode == "Fuzzy C-Means" and visible_clusters:
                    visible_cluster_indices = [int(c.replace("C", "")) for c in visible_clusters if c.startswith("C")]
                    if visible_cluster_indices:
                        keep = np.zeros(membership_view.shape[1], dtype=bool)
                        for idx in visible_cluster_indices:
                            if 0 <= idx < keep.shape[0]:
                                keep[idx] = True
                        membership_view = membership_view.copy()
                        membership_view[:, ~keep] = 0.0
                        denom = np.clip(membership_view.sum(axis=1, keepdims=True), 1e-12, None)
                        membership_view = membership_view / denom
                um = np.power(np.clip(membership_view, 1e-12, None), fuzzifier_m)
                projected_points = df_view[["x", "y"]].to_numpy(dtype=float)
                projected_centers = (um.T @ projected_points) / np.clip(um.sum(axis=0, keepdims=True).T, 1e-12, None)
            else:
                projected_centers = np.zeros((0, 2), dtype=float)
            x_min, x_max = float(df_view["x"].min()), float(df_view["x"].max())
            y_min, y_max = float(df_view["y"].min()), float(df_view["y"].max())
            x_pad = (x_max - x_min) * 0.08 + 1e-6
            y_pad = (y_max - y_min) * 0.08 + 1e-6
            gx = np.linspace(x_min - x_pad, x_max + x_pad, 180)
            gy = np.linspace(y_min - y_pad, y_max + y_pad, 180)
            grid_x, grid_y = np.meshgrid(gx, gy)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            if projected_centers.shape[0] >= 2:
                grid_membership, grid_labels = predict_fuzzy_membership(grid_points, projected_centers, m=fuzzifier_m)
                grid_uncertainty = (1.0 - grid_membership.max(axis=1)).reshape(grid_x.shape)
                grid_labels = grid_labels.reshape(grid_x.shape).astype(float)
            else:
                grid_uncertainty = np.zeros(grid_x.shape, dtype=float)
                grid_labels = np.zeros(grid_x.shape, dtype=float)

            uncertainty_fig = go.Figure()
            uncertainty_fig.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=grid_uncertainty,
                    colorscale=[(0.0, "#1a9850"), (0.5, "#fee08b"), (1.0, "#d73027")],
                    zmin=0.0,
                    zmax=1.0,
                    opacity=0.35,
                    colorbar={"title": "Uncertainty"},
                    hoverinfo="skip",
                )
            )
            uncertainty_fig.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=grid_labels,
                    showscale=False,
                    opacity=0.75,
                    line={"color": "#2f2f2f", "width": 1},
                    contours={"coloring": "none"},
                    hoverinfo="skip",
                )
            )
            scattergl_kwargs = {
                "x": df_view["x"],
                "y": df_view["y"],
                "mode": "markers",
                "customdata": df_view[hover_customdata_cols].to_numpy(),
                "hovertemplate": hover_template,
                "marker": {
                    "size": df_view["memberStyleSize"],
                    "symbol": df_view["memberStyleSymbol"],
                    "color": df_view["cluster_uncertainty"],
                    "colorscale": [(0.0, "#1a9850"), (0.5, "#fee08b"), (1.0, "#d73027")],
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "line": {"width": 0.6, "color": "#111111"},
                    "opacity": 0.92,
                },
                "showlegend": False,
            }
            uncertainty_fig.add_trace(go.Scattergl(**scattergl_kwargs))
            uncertainty_fig.update_layout(
                title="FCM Cluster Uncertainty Map (with boundaries)",
                template="plotly_white",
                height=700,
                xaxis_title="x",
                yaxis_title="y",
            )
            st.plotly_chart(uncertainty_fig, width="stretch")

    if skipped_members:
        st.caption(f"Skipped members (no file for source filter): {', '.join(skipped_members)}")

    if fcm_results:
        st.subheader("FCM Cluster Scan")
        st.dataframe(fcm_results, width="stretch", height=220, hide_index=True)

    if kmeans_results:
        st.subheader("KMeans k Scan")
        st.dataframe(kmeans_results, width="stretch", height=240, hide_index=True)

    if knn_results:
        st.subheader("KNN Graph Summary")
        st.dataframe([knn_results], width="stretch", height=120, hide_index=True)

    st.subheader("Points")
    points_table = df_view[["memberLabel", "date", "title", "cluster", "textSource", "bodyCharCount", "url"]].copy()
    st.dataframe(
        points_table,
        width="stretch",
        height=340,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Press release URL", display_text="Open release"),
        },
    )


if __name__ == "__main__":
    main()

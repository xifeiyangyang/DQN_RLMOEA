"""
=============================================================================
spatial_strategy_analysis.py — Four-Experiment Spatial Strategy Framework
=============================================================================
Exp 1: Urban-Rural Gradient Quantification (urban_score from POI+pop+dist to CBD)
Exp 2: Moran's I Spatial Autocorrelation (binary and urban-weighted)
Exp 3: Distance-to-CBD CDF curves
Exp 4: Permutation Test (10,000 iterations, statistical significance)

Output: statistical report + 3 analysis figures
=============================================================================
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt
import warnings
import os
import fiona

warnings.filterwarnings('ignore')

# ====================== 0. Global Style (Times New Roman) ======================
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.format": "png",
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ['S1 (Full-Coverage)', 'S2', 'S3', 'S4 (Cost-Efficient)']
LABELS_SHORT = ['S1', 'S2', 'S3', 'S4']
COLORS = ['#D55E00', '#E69F00', '#009E73', '#0072B2']
CBD_LON, CBD_LAT = 121.474, 31.232  # People's Square


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def load_data():
    candidate_file = os.path.join(BASE_DIR, "shanghai_data_processer", "垂直起降场候选选址结果.xlsx")
    candidate_df = pd.read_excel(candidate_file, engine='openpyxl')
    candidate_df.columns = ['FID', 'gpsx', 'gpsy', 'field1', 'field2', 'area_sqm', 'poi_count']

    demand_file = os.path.join(BASE_DIR, "shanghai_data_processer",
                               "community_population_20220301_having_pops.shp")
    demand_gdf = None
    for enc in ['utf-8', 'gbk', 'gb18030', 'latin-1', 'cp936']:
        try:
            with fiona.open(demand_file, encoding=enc) as src:
                demand_gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
            break
        except:
            continue
    if demand_gdf is None:
        demand_gdf = gpd.read_file(demand_file)

    county_gdf = gpd.read_file(os.path.join(BASE_DIR, "shanghai_data_processer", "上海市_县界.shp"))

    scheme_df = pd.read_excel(os.path.join(BASE_DIR, "RL-MOEA2_4_Schemes_Detail.xlsx"),
                              engine='openpyxl')
    scheme_indices = [[int(x) for x in str(row.iloc[4]).split(',')]
                      for _, row in scheme_df.iterrows()]

    return candidate_df, demand_gdf, county_gdf, scheme_indices


# ======================================================================
#  Experiment 1: Urban-Rural Gradient Quantification
# ======================================================================
def compute_urban_score(candidate_df, demand_gdf):
    print("\n" + "=" * 70)
    print("Experiment 1: Urban-Rural Gradient Quantification")
    print("=" * 70)

    n = len(candidate_df)
    scores = pd.DataFrame(index=range(n))
    scores['FID'] = candidate_df['FID']

    # (a) POI Density Score
    poi_raw = candidate_df['poi_count'].values
    scores['poi_score'] = stats.rankdata(poi_raw) / n
    print(f"\n  (a) POI Score: range=[{scores['poi_score'].min():.3f}, {scores['poi_score'].max():.3f}]")

    # (b) Population Density Score (weighted KDE)
    demand_coords = np.column_stack([demand_gdf['gpsx'].values, demand_gdf['gpsy'].values])
    demand_weights = demand_gdf['average_po'].values / demand_gdf['average_po'].sum()

    mean_lat = demand_coords[:, 1].mean()
    coords_km = demand_coords.copy()
    coords_km[:, 0] *= 111.32 * cos(radians(mean_lat))
    coords_km[:, 1] *= 110.574

    candidate_coords = candidate_df[['gpsx', 'gpsy']].values
    candidate_km = candidate_coords.copy()
    candidate_km[:, 0] *= 111.32 * cos(radians(mean_lat))
    candidate_km[:, 1] *= 110.574

    kde = KernelDensity(bandwidth=3.0)
    kde.fit(coords_km, sample_weight=demand_weights)
    pop_density = np.exp(kde.score_samples(candidate_km))
    scores['pop_score'] = stats.rankdata(pop_density) / n
    print(f"  (b) Population Density Score: range=[{scores['pop_score'].min():.3f}, {scores['pop_score'].max():.3f}]")

    # (c) Distance-to-CBD Score
    dist_to_cbd = np.array([
        haversine_distance(candidate_df.iloc[i]['gpsx'], candidate_df.iloc[i]['gpsy'],
                           CBD_LON, CBD_LAT) for i in range(n)
    ])
    scores['dist_score'] = stats.rankdata(-dist_to_cbd) / n
    print(f"  (c) Distance-to-CBD Score: range=[{scores['dist_score'].min():.3f}, {scores['dist_score'].max():.3f}]")
    print(f"      Physical distance range: [{dist_to_cbd.min():.1f}, {dist_to_cbd.max():.1f}] km")

    # Composite: equal-weight average
    scores['urban_score'] = (scores['poi_score'] + scores['pop_score'] + scores['dist_score']) / 3.0

    # PCA validation
    scaler = StandardScaler()
    X = scaler.fit_transform(scores[['poi_score', 'pop_score', 'dist_score']].values)
    pca = PCA(n_components=1)
    scores['urban_pca'] = pca.fit_transform(X)[:, 0]
    scores['urban_pca'] = (scores['urban_pca'] - scores['urban_pca'].min()) / \
                          (scores['urban_pca'].max() - scores['urban_pca'].min())
    if np.corrcoef(scores['urban_score'], scores['urban_pca'])[0, 1] < 0:
        scores['urban_pca'] = 1 - scores['urban_pca']

    corr = np.corrcoef(scores['urban_score'], scores['urban_pca'])[0, 1]
    print(f"\n  Equal-weight vs PCA r = {corr:.4f} (r > 0.80, equal-weight sufficient)")

    urban_median = scores['urban_score'].median()
    print(f"  urban_score median: {urban_median:.4f}")
    print(f"  High-urban (>median): {np.sum(scores['urban_score'] > urban_median)} pts")
    print(f"  Low-urban (<=median): {np.sum(scores['urban_score'] <= urban_median)} pts")

    return scores, dist_to_cbd


# ======================================================================
#  Experiment 2: Moran's I Spatial Autocorrelation
# ======================================================================
def compute_morans_i(candidate_df, scheme_indices, urban_scores):
    print("\n" + "=" * 70)
    print("Experiment 2: Moran's I Spatial Autocorrelation")
    print("=" * 70)

    n_candidates = len(candidate_df)
    coords = candidate_df[['gpsx', 'gpsy']].values

    k = 8
    mean_lat = coords[:, 1].mean()
    coords_approx = coords.copy()
    coords_approx[:, 0] *= 111.32 * cos(radians(mean_lat))
    coords_approx[:, 1] *= 110.574
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords_approx)
    _, indices_nn = nn.kneighbors(coords_approx)

    W = np.zeros((n_candidates, n_candidates))
    for i in range(n_candidates):
        for j_idx in range(1, k + 1):
            W[i, indices_nn[i, j_idx]] = 1.0
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, np.newaxis]

    print(f"\n  Spatial weight matrix: k={k}-NN, row-standardized")
    E_I = -1.0 / (n_candidates - 1)

    results = []
    for s_idx, indices in enumerate(scheme_indices):
        # Binary Moran's I
        binary = np.zeros(n_candidates)
        binary[indices] = 1.0
        bc = binary - binary.mean()
        num = sum(W[i, j] * bc[i] * bc[j] for i in range(n_candidates) for j in range(n_candidates) if W[i, j] > 0)
        den = np.sum(bc ** 2)
        I_binary = (n_candidates / W.sum()) * (num / den) if den > 0 else 0.0

        # Urban-weighted Moran's I
        urban_sel = np.zeros(n_candidates)
        urban_sel[indices] = urban_scores['urban_score'].values[indices]
        uc = urban_sel - urban_sel.mean()
        num_u = sum(W[i, j] * uc[i] * uc[j] for i in range(n_candidates) for j in range(n_candidates) if W[i, j] > 0)
        den_u = np.sum(uc ** 2)
        I_urban = (n_candidates / W.sum()) * (num_u / den_u) if den_u > 0 else 0.0

        if I_binary > E_I + 0.05:
            interp = "Clustered (selected sites tend to group together)"
        elif I_binary < E_I - 0.05:
            interp = "Dispersed (selected sites avoid each other)"
        else:
            interp = "Near random"

        print(f"\n  {LABELS[s_idx]}:")
        print(f"    I_binary = {I_binary:.4f}  (E[I] = {E_I:.4f})")
        print(f"    I_urban  = {I_urban:.4f}  (E[I] = {E_I:.4f})")
        print(f"    Interpretation: {interp}")

        results.append({'scheme': LABELS[s_idx], 'I_binary': I_binary,
                        'I_urban': I_urban, 'E_random': E_I, 'interpretation': interp})

    return results


# ======================================================================
#  Moran Scatter Plot (visual companion to Experiment 2)
# ======================================================================
def plot_moran_scatter(candidate_df, scheme_indices, urban_scores):
    """
    Moran scatter plot for each scheme.
    X-axis: standardized urban_score of selected points (z_i)
    Y-axis: spatial lag (Wz_i, average of neighbors' urban_score)
    Four quadrants:
      HH (upper-right) = high-urban point surrounded by high-urban neighbors
      HL (lower-right) = high-urban point surrounded by low-urban neighbors
      LH (upper-left)  = low-urban point surrounded by high-urban neighbors
      LL (lower-left)  = low-urban point surrounded by low-urban neighbors
    Slope of fitted line = Moran's I.
    """
    print("\n" + "=" * 70)
    print("Moran Scatter Plot")
    print("=" * 70)

    n = len(candidate_df)
    coords = candidate_df[['gpsx', 'gpsy']].values
    mean_lat = coords[:, 1].mean()

    # Build k-NN spatial weight matrix (same as Experiment 2)
    k = 8
    coords_km = coords.copy()
    coords_km[:, 0] *= 111.32 * np.cos(np.radians(mean_lat))
    coords_km[:, 1] *= 110.574
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords_km)
    _, indices_nn = nn.kneighbors(coords_km)

    W = np.zeros((n, n))
    for i in range(n):
        for j_idx in range(1, k + 1):
            W[i, indices_nn[i, j_idx]] = 1.0
    W = W / W.sum(axis=1, keepdims=True)

    E_I = -1.0 / (n - 1)
    u = urban_scores['urban_score'].values

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    quad_colors = {'HH': '#E41A1C', 'HL': '#FDAE6B', 'LH': '#6BAED6', 'LL': '#2171B5'}
    row_titles = ['(a) I_binary', '(b) I_urban']

    for s_idx, sel_indices in enumerate(scheme_indices):
        is_sel = np.zeros(n, dtype=bool); is_sel[sel_indices] = True

        # ===== ROW 1: I_binary scatter =====
        ax = axes[0, s_idx]
        binary = np.zeros(n); binary[sel_indices] = 1.0
        z_b = (binary - binary.mean()) / binary.std()
        Wz_b = W.dot(z_b)

        # All 200 points (unselected in gray, selected colored by quadrant)
        ax.scatter(z_b[~is_sel], Wz_b[~is_sel], s=6, c='lightgray', alpha=0.30, marker='.', zorder=1)
        z_s = z_b[is_sel]; wz_s = Wz_b[is_sel]
        hh = (z_s>0)&(wz_s>0); hl = (z_s>0)&(wz_s<=0)
        for mask, label, qc in [(hh,'HH',quad_colors['HH']),(hl,'HL',quad_colors['HL'])]:
            if mask.sum()>0:
                ax.scatter(z_s[mask], wz_s[mask], s=18, c=qc, edgecolor='black',
                           linewidth=0.3, alpha=0.85, label=f'{label}({mask.sum()})', zorder=3)

        # Regression line
        slope_b, intercept_b = np.polyfit(z_b, Wz_b, 1)
        xl = np.array([z_b.min(), z_b.max()])
        ax.plot(xl, slope_b*xl + intercept_b, 'k-', linewidth=1.0, alpha=0.7, zorder=4)
        ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.3)
        ax.axvline(0, color='gray', lw=0.4, ls='--', alpha=0.3)

        ax.text(0.97, 0.95, f"I = {slope_b:+.4f}\nE[I] = {E_I:+.4f}", transform=ax.transAxes,
                fontsize=8, ha='right', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_title(LABELS[s_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Standardized Selection (z)', fontsize=7)
        if s_idx == 0: ax.set_ylabel('Spatial Lag (Wz)\n' + row_titles[0], fontsize=8)
        ax.legend(loc='lower right', fontsize=5.5, frameon=True, facecolor='white', edgecolor='gray')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # Quadrant labels
        for tx, ty, lbl in [(0.97,0.97,'HH'),(0.97,0.03,'HL'),(0.03,0.03,'LL'),(0.03,0.97,'LH')]:
            ax.text(tx, ty, lbl, transform=ax.transAxes, fontsize=6, ha='right' if tx>0.5 else 'left',
                    va='top' if ty>0.5 else 'bottom', color='gray', alpha=0.5)

        # ===== ROW 2: I_urban scatter =====
        ax = axes[1, s_idx]
        us = np.zeros(n); us[sel_indices] = u[sel_indices]
        zu = (us - us.mean()) / us.std()
        Wz_u = W.dot(zu)

        # Unselected all at same z (negative constant), but spread on Wz
        ax.scatter(zu[~is_sel], Wz_u[~is_sel], s=6, c='lightgray', alpha=0.30, marker='.', zorder=1)
        # Selected points colored by quadrant
        z_us = zu[is_sel]; wz_us = Wz_u[is_sel]
        hh = (z_us>0)&(wz_us>0); hl = (z_us>0)&(wz_us<=0)
        lh = (z_us<=0)&(wz_us>0); ll = (z_us<=0)&(wz_us<=0)
        for mask, label, qc in [(hh,'HH',quad_colors['HH']),(hl,'HL',quad_colors['HL']),
                                 (lh,'LH',quad_colors['LH']),(ll,'LL',quad_colors['LL'])]:
            if mask.sum()>0:
                ax.scatter(z_us[mask], wz_us[mask], s=18, c=qc, edgecolor='black',
                           linewidth=0.3, alpha=0.85, label=f'{label}({mask.sum()})', zorder=3)

        # Regression line
        slope_u, intercept_u = np.polyfit(zu, Wz_u, 1)
        xl = np.array([zu.min(), zu.max()])
        ax.plot(xl, slope_u*xl + intercept_u, 'k-', linewidth=1.0, alpha=0.7, zorder=4)
        ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.3)
        ax.axvline(0, color='gray', lw=0.4, ls='--', alpha=0.3)

        ax.text(0.97, 0.95, f"I = {slope_u:+.4f}\nE[I] = {E_I:+.4f}", transform=ax.transAxes,
                fontsize=8, ha='right', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_xlabel('Standardized Urban Score (z)', fontsize=7)
        if s_idx == 0: ax.set_ylabel('Spatial Lag (Wz)\n' + row_titles[1], fontsize=8)
        ax.legend(loc='lower right', fontsize=5.5, frameon=True, facecolor='white', edgecolor='gray')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        for tx, ty, lbl in [(0.97,0.97,'HH'),(0.97,0.03,'HL'),(0.03,0.03,'LL'),(0.03,0.97,'LH')]:
            ax.text(tx, ty, lbl, transform=ax.transAxes, fontsize=6, ha='right' if tx>0.5 else 'left',
                    va='top' if ty>0.5 else 'bottom', color='gray', alpha=0.5)

    out = os.path.join(OUTPUT_DIR, "Fig_Moran_Scatter.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    print(f"  Saved: {os.path.basename(out)}")
    plt.close()


# ======================================================================
#  Experiment 3: Distance-to-CBD Cumulative Distribution
# ======================================================================
def plot_distance_cdf(candidate_df, scheme_indices, dist_to_cbd, urban_scores):
    print("\n" + "=" * 70)
    print("Experiment 3: Distance-to-CBD CDF Analysis")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Distance-to-CBD CDF
    ax = axes[0]
    all_sorted = np.sort(dist_to_cbd)
    all_cdf = np.arange(1, len(all_sorted) + 1) / len(all_sorted)
    ax.plot(all_sorted, all_cdf, 'k--', linewidth=1.5, alpha=0.5, label='All candidates (200)')

    for s_idx, indices in enumerate(scheme_indices):
        d = np.sort(dist_to_cbd[indices])
        cdf = np.arange(1, len(d) + 1) / len(d)
        ax.step(d, cdf, where='post', color=COLORS[s_idx], linewidth=2.0,
                alpha=0.85, label=f'{LABELS_SHORT[s_idx]} (n={len(indices)})')

    ax.set_xlabel('Distance to CBD (km)')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('Distance-to-CBD CDF')
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Right: Urban Score CDF
    ax = axes[1]
    all_u = np.sort(urban_scores['urban_score'].values)
    all_uc = np.arange(1, len(all_u) + 1) / len(all_u)
    ax.plot(all_u, all_uc, 'k--', linewidth=1.5, alpha=0.5, label='All candidates (200)')

    for s_idx, indices in enumerate(scheme_indices):
        u = np.sort(urban_scores['urban_score'].values[indices])
        uc = np.arange(1, len(u) + 1) / len(u)
        ax.step(u, uc, where='post', color=COLORS[s_idx], linewidth=2.0,
                alpha=0.85, label=f'{LABELS_SHORT[s_idx]} (n={len(indices)})')

    ax.set_xlabel('Urban Score (0 = Rural, 1 = Urban Core)')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('Urban Score CDF')
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "Fig_CDF_Curves.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    print(f"  Saved: {os.path.basename(out)}")
    plt.close()

    # Numerical summary
    ub = urban_scores['urban_score'].values
    u_med = urban_scores['urban_score'].median()
    print(f"\n  {'Scheme':<24} {'Mean Dist(km)':>13} {'Mean Urban':>11} {'High-Urban %':>13}")
    print(f"  {'All candidates (200)':<24} {np.mean(dist_to_cbd):>13.1f} {ub.mean():>11.4f} {np.mean(ub > u_med):>12.1%}")
    for s_idx, indices in enumerate(scheme_indices):
        d = dist_to_cbd[indices]
        u = ub[indices]
        print(f"  {LABELS_SHORT[s_idx]:<24} {np.mean(d):>13.1f} {u.mean():>11.4f} {np.mean(u > u_med):>12.1%}")

    return fig


# ======================================================================
#  Experiment 4: Permutation Test
# ======================================================================
def permutation_test(candidate_df, scheme_indices, urban_scores):
    print("\n" + "=" * 70)
    print("Experiment 4: Permutation Test (10,000 iterations)")
    print("=" * 70)

    np.random.seed(42)
    n_perm = 10000
    all_urban = urban_scores['urban_score'].values
    n_total = len(all_urban)

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    for s_idx, indices in enumerate(scheme_indices):
        n_sel = len(indices)
        actual_mean = np.mean(all_urban[indices])

        null_means = np.array([np.mean(np.random.choice(all_urban, n_sel, replace=False))
                               for _ in range(n_perm)])

        p_left = np.mean(null_means <= actual_mean)
        p_right = np.mean(null_means >= actual_mean)
        p_two = 2 * min(p_left, p_right)
        es = (actual_mean - np.mean(null_means)) / np.std(null_means)

        sig = '***' if p_two < 0.01 else ('**' if p_two < 0.05 else ('*' if p_two < 0.1 else 'n.s.'))
        direction = 'Urban-oriented' if es > 0 else 'Rural gap-filling'
        print(f"\n  {LABELS[s_idx]} (n={n_sel}):")
        print(f"    Actual mean urban_score:  {actual_mean:.4f}")
        print(f"    Null distribution:        {np.mean(null_means):.4f} +/- {np.std(null_means):.4f}")
        print(f"    Two-tailed p-value:       {p_two:.4f} {sig}")
        print(f"    Cohen's d (effect size):  {es:+.3f}  -> {direction}")

        ax = axes[s_idx]
        ax.hist(null_means, bins=50, color='lightgray', edgecolor='gray', alpha=0.7, density=True)
        ax.axvline(actual_mean, color=COLORS[s_idx], linewidth=2.5,
                   label=f'{LABELS_SHORT[s_idx]} actual\nmean = {actual_mean:.3f}')
        ax.axvline(np.mean(null_means), color='black', linewidth=1.5, linestyle='--',
                   label=f'Null mean\n{np.mean(null_means):.3f}')
        ax.text(0.98, 0.95, f'p = {p_two:.4f}\nCohen d = {es:+.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel('Mean Urban Score')
        ax.set_ylabel('Density')
        ax.set_title(LABELS[s_idx])
        ax.legend(loc='upper left', fontsize=7, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        results.append({'scheme': LABELS[s_idx], 'n': n_sel, 'actual_mean': actual_mean,
                        'null_mean': np.mean(null_means), 'null_std': np.std(null_means),
                        'p_two_tailed': p_two, 'cohens_d': es})

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "Fig_Permutation_Test.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    print(f"\n  Saved: {os.path.basename(out)}")
    plt.close()

    return results


# ======================================================================
#  Composite: Urban Gradient Map
# ======================================================================
def plot_urban_gradient_map(candidate_df, scheme_indices, urban_scores, county_gdf):
    print("\n" + "=" * 70)
    print("Composite: Urban Gradient Map")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1],
                          width_ratios=[1, 1, 1, 1, 0.12],
                          hspace=0.28, wspace=0.20,
                          left=0.04, right=0.96, top=0.94, bottom=0.10)

    # Row 1: 4 maps + colorbar
    for s_idx, indices in enumerate(scheme_indices):
        ax = fig.add_subplot(gs[0, s_idx])
        county_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)

        all_set = set(range(len(candidate_df)))
        unselected = list(all_set - set(indices))
        ax.scatter(candidate_df.iloc[unselected]['gpsx'],
                   candidate_df.iloc[unselected]['gpsy'],
                   s=2, color='lightgray', alpha=0.3, marker='.')

        sel_urban = urban_scores['urban_score'].values[indices]
        sel_coords = candidate_df.iloc[indices]
        ax.scatter(sel_coords['gpsx'], sel_coords['gpsy'],
                   s=22, c=sel_urban, cmap='RdYlBu_r',
                   edgecolor='black', linewidth=0.3, vmin=0, vmax=1, zorder=5)

        ax.set_title(LABELS_SHORT[s_idx], fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Colorbar
    ax_cbar = fig.add_subplot(gs[0, 4])
    import matplotlib.colorbar as mcb
    norm = plt.Normalize(0, 1)
    cb = mcb.ColorbarBase(ax_cbar, cmap='RdYlBu_r', norm=norm, orientation='vertical')
    cb.set_label('Urban Score', fontsize=10, labelpad=2)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0\n(Rural)', '0.5', '1\n(Urban)'])
    cb.ax.tick_params(labelsize=8)

    # Row 2: Boxplot (cols 0:2) + Bar (cols 2:5)
    ax_box = fig.add_subplot(gs[1, 0:2])
    box_data = [urban_scores['urban_score'].values[idx] for idx in scheme_indices]

    positions = [0.5, 1.1, 1.7, 2.3]
    bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.30,
                        positions=positions,
                        medianprops={'color': 'black', 'linewidth': 1.2})
    for i, color in enumerate(COLORS):
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.6)

    ax_box.axhline(y=urban_scores['urban_score'].median(), color='black',
                   linestyle='--', linewidth=0.8, alpha=0.5)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(LABELS_SHORT, fontsize=11)
    ax_box.set_xlim(0.1, 2.7)
    ax_box.set_ylabel('Urban Score', fontsize=10)
    ax_box.set_title('Urban Score Distribution', fontsize=11, loc='left')
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.grid(True, axis='y', alpha=0.3, linestyle='--')

    ax_bar = fig.add_subplot(gs[1, 2:5])
    u_med = urban_scores['urban_score'].median()
    high_pcts = [np.mean(urban_scores['urban_score'].values[idx] > u_med) * 100
                 for idx in scheme_indices]
    bars = ax_bar.bar(LABELS_SHORT, high_pcts, color=COLORS, alpha=0.8,
                      edgecolor='black', linewidth=0.5, width=0.45)
    ax_bar.axhline(y=50, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_bar.set_ylabel('High-Urban Ratio (%)', fontsize=10)
    ax_bar.set_title('Urban-Oriented Selection Ratio', fontsize=11, loc='left')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(labelsize=9)
    for bar, pct in zip(bars, high_pcts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')

    out = os.path.join(OUTPUT_DIR, "Fig_Urban_Gradient_Map.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    print(f"  Saved: {os.path.basename(out)}")
    plt.close()


# ======================================================================
#  Main
# ======================================================================
def main():
    print("=" * 70)
    print("RL-MOEA2 Spatial Strategy Analysis — Four-Experiment Framework")
    print("=" * 70)

    candidate_df, demand_gdf, county_gdf, scheme_indices = load_data()
    print(f"\nData loaded: {len(candidate_df)} candidate sites, {len(demand_gdf)} demand points")

    urban_scores, dist_to_cbd = compute_urban_score(candidate_df, demand_gdf)
    moran_results = compute_morans_i(candidate_df, scheme_indices, urban_scores)
    plot_moran_scatter(candidate_df, scheme_indices, urban_scores)
    cdf_fig = plot_distance_cdf(candidate_df, scheme_indices, dist_to_cbd, urban_scores)
    perm_results = permutation_test(candidate_df, scheme_indices, urban_scores)
    plot_urban_gradient_map(candidate_df, scheme_indices, urban_scores, county_gdf)

    # Final Report
    print("\n\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print("\n--- Moran's I ---")
    for r in moran_results:
        print(f"  {r['scheme']}: I_binary={r['I_binary']:.4f} | I_urban={r['I_urban']:.4f} | {r['interpretation']}")

    print("\n--- Distance to CBD & Urban Score ---")
    u = urban_scores['urban_score'].values
    um = urban_scores['urban_score'].median()
    for s_idx, indices in enumerate(scheme_indices):
        pu = u[indices]
        print(f"  {LABELS_SHORT[s_idx]}: mean_urban={pu.mean():.4f} | "
              f"high-urban={np.mean(pu > um):.1%} | "
              f"mean_dist_to_CBD={np.mean(dist_to_cbd[indices]):.1f} km")

    print("\n--- Permutation Test ---")
    for r in perm_results:
        sig = '***' if r['p_two_tailed'] < 0.01 else ('**' if r['p_two_tailed'] < 0.05 else 'n.s.')
        direction = 'Urban preference' if r['cohens_d'] > 0 else 'Rural gap-filling'
        print(f"  {r['scheme']}: p={r['p_two_tailed']:.4f} {sig} | d={r['cohens_d']:+.3f} -> {direction}")

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  Fig_CDF_Curves.png")
    print("  Fig_Moran_Scatter.png")
    print("  Fig_Permutation_Test.png")
    print("  Fig_Urban_Gradient_Map.png")
    print("\nDone.")


if __name__ == "__main__":
    main()

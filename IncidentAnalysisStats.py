import warnings

warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu, poisson
from lifelines import KaplanMeierFitter

WORKBOOK = "XO_incidents_RCA_merchants.xlsx"
SHEET = "Data - Updated"
SAMPLE_WINDOW = ("2025-10-01", "2026-06-01")
# colors used
BLUE = "#008CFF"
GRAY = "#595959"
LTGRAY = "#BFBFBF"
RED = "#E60000"
BLACK = "#222222"
LTBLUE = "#60CDFF"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "axes.edgecolor": "#CFCFCF",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def onlyx(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)


# who owns each root cause -- PayPal, Merchant, or Process
owner_map = {
    "Unclassified / Other": "Process",
    "Merchant-side Misconfiguration / Account Issue": "Merchant",
    "Configuration Regression": "PayPal",
    "Code Deployment Regression": "PayPal",
    "Merchant-side Vendor Failure": "Merchant",
    "Payment Partner Failure": "Merchant",
    "Compute / Network / Service Outage": "Merchant",
    "Conversion / Volume Variance": "Process",
    "API Contract / Validation Bug": "PayPal",  # shown as "API Contact" downstream, same field
    "Security / Abuse / Bot Attack": "Process",
    "Internal Merchant Misconfiguration / Account Issue": "Merchant",
    "3DS / Authentication / Login Failure": "PayPal",
    "Planned Maintenance / Migration Impact": "Merchant",
    "Risk / Fraud Engine Over-trigger": "PayPal",
    "JS SDK / Button Rendering Issue": "PayPal",
    "Webhook / Event Notification Bug": "PayPal",
    "Merchant Misconfiguration / Account Issue": "Merchant",
    "Third-party / External Dependency Failure": "Merchant",
    "Unexplained Conversion / UX Drop-off": "Process",
    "Code / Config Deployment Regression": "PayPal",
    "Infrastructure / Service Availability": "PayPal",
}
bucket_name = {
    "PayPal": "PayPal-controllable",
    "Merchant": "Merchant / External",
    "Process": "Process / No-defect",
}


def load_data(path=WORKBOOK):
    raw = pd.read_excel(path, sheet_name=SHEET)
    raw = raw.rename(
        columns={
            "Root Cause Category": "cat",
            "Reassigned Merchant / Partner": "merch_reassigned",
            "Merchant / Partner": "merch_orig",
            "ower": "prio",
            "opened_at": "opened",
            "sys_updated_on": "updated",
            "Change relationship": "change",
        }
    )
    raw["cat"] = raw["cat"].astype(str).str.strip()
    raw["opened"] = pd.to_datetime(raw["opened"])
    raw["updated"] = pd.to_datetime(raw["updated"])
    raw["merch"] = raw["merch_reassigned"].fillna(raw["merch_orig"])
    raw["split"] = np.where(
        raw["merch"].astype(str).str.strip().eq("Shopify"), "Shopify", "Non-Shopify"
    )
    raw["mttr"] = (raw["updated"] - raw["opened"]).dt.total_seconds() / 3600
    raw["month"] = raw["opened"].dt.to_period("M")
    chg_txt = raw["change"].astype(str).str.lower()
    raw["chg"] = np.where(
        chg_txt.str.contains("caused by change|external change|undocumented"),
        "Change-caused",
        "No change",
    )
    raw["owner"] = raw["cat"].map(owner_map)
    is_p2 = raw["prio"].astype(str).str.contains("P2")
    full = raw[is_p2].copy()
    in_window = (raw["opened"] >= SAMPLE_WINDOW[0]) & (raw["opened"] < SAMPLE_WINDOW[1])
    not_test = ~raw["cat"].str.contains("Test", case=False, na=False)
    samp = raw[is_p2 & in_window & not_test].copy()

    return full, samp


full, samp = load_data()
print(f"full={len(full)}, samp={len(samp)}")


# P2 Incident Volume Over Time; limits = mean +/- 3*sqrt(mean); Assumes monthly counts behave like a Poisson process (variance ~ mean) which is the normal assumption for count data like this
def control_chart(full):
    counts = full.groupby("month").size()
    months = counts.index
    vals = counts.values
    mean = vals.mean()
    ucl = mean + 3 * math.sqrt(mean)
    lcl = max(0, mean - 3 * math.sqrt(mean))
    bad = [i for i, v in enumerate(vals) if v > ucl or v < lcl]
    print(
        f"mean={mean:.1f} ucl={ucl:.1f} lcl={lcl:.1f}, {len(bad)}/{len(vals)} months out of range"
    )
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.plot(range(len(vals)), vals, "-", color=BLUE, lw=2, zorder=2)
    ok = [i for i in range(len(vals)) if i not in bad]
    ax.plot(ok, vals[ok], "o", color=BLUE, ms=5, zorder=3)
    ax.plot(bad, vals[bad], "o", color=RED, ms=11, zorder=4)
    ax.axhline(ucl, ls="--", color=LTGRAY, lw=1.6)
    ax.axhline(lcl, ls="--", color=LTGRAY, lw=1.6)
    ax.axhline(mean, ls="--", color=GRAY, lw=1.6)
    ax.set_xticks(range(0, len(months), 3))
    ax.set_xticklabels([str(months[i]) for i in range(0, len(months), 3)], fontsize=14)
    ax.set_ylabel("P2 incidents per month", fontsize=15)
    onlyx(ax)
    plt.tight_layout()
    plt.savefig("chart1.png", dpi=200)
    plt.close()


# Root-Cause Concentration; Pareto chart -- top 8 categories in the sample, ranked by count
def concentration(samp):
    cnt = samp.cat.value_counts().head(8)
    order = list(cnt.index)[::-1]
    colors = [BLUE if owner_map.get(k) == "PayPal" else LTGRAY for k in order]
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    ax.barh(range(len(order)), [cnt[k] for k in order], color=colors, height=0.72)
    for i, k in enumerate(order):
        ax.text(cnt[k] + 0.4, i, f"{cnt[k]}", va="center", fontsize=16, color=BLACK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=15)
    onlyx(ax)
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("chart2.png", dpi=200)
    plt.close()
    print(f"top 8 = {100 * cnt.sum() / len(samp):.0f}% of the sample")


# Root Cause Resolution -- Median days to resolve, per category, full population, resolved tickets only;  Use median as resolution time isskewed; also drop categories with fewer than 6 tickets
def resolution_top6(full):
    done = full[(full.state != "In-Progress") & (full.mttr >= 0)]
    med = done.groupby("cat").mttr.median()
    cnts = done.groupby("cat").size()
    med = med[cnts >= 6].sort_values(ascending=False).head(6).sort_values()
    ext = {"Merchant Misconfiguration / Account Issue"}
    proc = {"Conversion / Volume Variance"}
    colors = [GRAY if k in ext else (LTBLUE if k in proc else BLUE) for k in med.index]
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    ax.barh(range(len(med)), med.values / 24, color=colors, height=0.66)
    for i, k in enumerate(med.index):
        ax.text(
            med.values[i] / 24 + 0.4,
            i,
            f"{med.values[i] / 24:.1f} days",
            va="center",
            fontsize=18,
            color=colors[i],
        )
    ax.set_yticks(range(len(med)))
    ax.set_yticklabels(med.index, fontsize=18)
    onlyx(ax)
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("chart3.png", dpi=200)
    plt.close()
    print("slowest 6 categories, days:")
    print((med / 24).round(1).to_string())


# Incident Ownership --  % split across the 3 owner buckets, sample only. No test
def ownership(samp):
    pct = (samp.owner.map(bucket_name).value_counts(normalize=True) * 100).round(1)
    print("ownership split:")
    print(pct.to_string())
    return pct


# Change Governance as a Control Lever -- Fisher's exact test on a 2x2 table: owner (PayPal/Merchant) x change-caused (yes/no); Use Fisher instead of chi-square as some of the cell counts are small (merchant + change-caused = 5),
def change_governance(samp):
    sub = samp[samp.owner.isin(["PayPal", "Merchant"])]
    tab = pd.crosstab(sub.owner, sub.chg)
    odds, p = fisher_exact(tab.values)
    pp_pct = 100 * tab.loc["PayPal", "Change-caused"] / tab.loc["PayPal"].sum()
    mp_pct = 100 * tab.loc["Merchant", "Change-caused"] / tab.loc["Merchant"].sum()
    print(
        f"paypal change-caused={pp_pct:.0f}% merchant change-caused={mp_pct:.0f}% "
        f"fisher odds={odds:.3f} p={p:.5f}"
    )
    return tab, odds, p


# Resolution Performance by Service Model -- Kaplan-Meier survival curve (time to resolve), Shopify vs everyone else, sample only; still-open tickets get censored instead of dropped; Mann-Whitney to check if the two groups differ, since resolution time is skewed and a t-test would need normal data
def km_resolution(samp):
    d = samp[samp.mttr >= 0].copy()
    d["event"] = (d.state != "In-Progress").astype(int)

    fig, ax = plt.subplots(figsize=(13, 5.4))
    meds = {}
    for grp, col in [("Shopify", BLUE), ("Non-Shopify", GRAY)]:
        km = KaplanMeierFitter()
        sub = d[d.split == grp]
        km.fit(sub.mttr / 24, sub.event, label=grp)
        km.plot_survival_function(ax=ax, color=col, linewidth=3, ci_show=False)
        meds[grp] = km.median_survival_time_
    ax.set_xlabel("Days to resolution", fontsize=17)
    ax.set_ylabel("Share still open", fontsize=17)
    ax.legend(fontsize=16, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig("chart4.png", dpi=200)
    plt.close()

    done = d[d.event == 1]
    u, p = mannwhitneyu(
        done[done.split == "Shopify"].mttr, done[done.split == "Non-Shopify"].mttr
    )
    print(f"km medians (days)={meds}, mann-whitney u={u:.0f} p={p:.4f}")
    return meds, p


# Shopify Share of P2 Incidents -- Plotting Shopify's monthly share against the overall average -- Exploratory trend chart
def shopify_share(full):
    months = sorted(full.month.unique())
    tot = full.groupby("month").size().reindex(months).values
    shop = (
        full[full.split == "Shopify"]
        .groupby("month")
        .size()
        .reindex(months, fill_value=0)
        .values
    )
    share = 100 * shop / tot
    post = [i for i, m in enumerate(months) if m >= pd.Period("2024-11", "M")]
    recent_avg = 100 * shop[post].sum() / tot[post].sum()

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(range(len(months)), share, "-o", color=BLUE, lw=2.2, ms=5)
    ax.axhline(recent_avg, ls="--", color=LTGRAY, lw=2)
    ax.set_xticks(range(0, len(months), 3))
    ax.set_xticklabels([str(months[i]) for i in range(0, len(months), 3)], fontsize=14)
    ax.set_ylabel("Shopify share of P2 incidents (%)", fontsize=15)
    ax.set_ylim(0, 100)
    onlyx(ax)
    plt.tight_layout()
    plt.savefig("chart5.png", dpi=200)
    plt.close()
    print(f"recent avg share = {recent_avg:.0f}%")


# Drivers of the Feb-Apr 2025 Spike -- Assumes the baseline rate outside the spike is roughly constant
def spike(full):
    spike_mo = [pd.Period(x, "M") for x in ["2025-02", "2025-03", "2025-04"]]
    spk = full[full.month.isin(spike_mo)]
    base = full[~full.month.isin(spike_mo)]
    nmo = full.month.nunique() - 3
    rate = len(base) / nmo
    expected = rate * 3
    p = 1 - poisson.cdf(len(spk) - 1, expected)
    print(
        f"spike total={len(spk)} expected={expected:.0f} ratio={len(spk) / expected:.2f}x poisson p={p:.2e}"
    )

    top6 = spk.cat.value_counts().head(6)
    base_cnt = base.cat.value_counts() / nmo * 3
    cats = list(top6.index)[::-1]
    y = np.arange(len(cats))
    h = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.barh(y + h / 2, [top6[k] for k in cats], h, color=BLUE)
    ax.barh(y - h / 2, [base_cnt.get(k, 0) for k in cats], h, color=LTGRAY)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=15)
    bare(ax)
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("chart6.png", dpi=200)
    plt.close()
    return p


# Only for PayPal-owned tickets, split by whether a change caused it
def change_resolution(samp):
    pp = samp[
        (samp.owner == "PayPal") & (samp.state != "In-Progress") & (samp.mttr >= 0)
    ]
    grp = pp.groupby("chg").mttr.median() / 24
    n = pp.groupby("chg").size()
    print(
        f"change-caused median={grp.get('Change-caused', float('nan')):.1f}d (n={n.get('Change-caused', 0)}), "
        f"no-change median={grp.get('No change', float('nan')):.1f}d (n={n.get('No change', 0)})"
    )
    u, p = mannwhitneyu(
        pp[pp.chg == "Change-caused"].mttr, pp[pp.chg == "No change"].mttr
    )
    print(f"mann-whitney u={u:.0f} p={p:.4f} (small groups, read directionally)")
    return grp, p


# Internal Change-Related Root Causess; Just a headcount of root causes within the PayPal-owned + change-caused group
def internal_causes(samp):
    chg_df = samp[(samp.owner == "PayPal") & (samp.chg == "Change-caused")]
    cnt = chg_df.cat.value_counts()
    print("paypal-owned change-caused, by category:")
    print(cnt.to_string())

    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = np.arange(len(cnt))[::-1]
    ax.barh(y, cnt.values, color=BLUE, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(cnt.index, fontsize=14)
    bare(ax)
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("chart7.png", dpi=200)
    plt.close()
    return cnt


# Drivers of Extended Resolution Time Showing the 50th/75th/90th percentiles for shopify vs non-shopify
def percentiles(full):
    done = full[(full.state != "In-Progress") & (full.mttr >= 0)]
    sh = done[done.split == "Shopify"].mttr
    ns = done[done.split == "Non-Shopify"].mttr
    rows = []
    for label, q in [
        ("50th percentile", 0.50),
        ("75th percentile", 0.75),
        ("90th percentile", 0.90),
    ]:
        rows.append((label, sh.quantile(q) / 24, ns.quantile(q) / 24))
    print("percentiles (days):", rows)

    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    h = 0.34
    for i, (name, sv, nv) in enumerate(rows):
        yy = len(rows) - 1 - i
        ax.barh(yy + h / 2, sv, h, color=BLUE)
        ax.barh(yy - h / 2, nv, h, color=GRAY)
        ax.text(
            sv + 1.5, yy + h / 2, f"{sv:.1f} days", va="center", fontsize=17, color=BLUE
        )
        ax.text(
            nv + 1.5, yy - h / 2, f"{nv:.1f} days", va="center", fontsize=17, color=GRAY
        )
        ax.text(-3, yy, name, va="center", ha="right", fontsize=18, color=BLACK)
    bare(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("chart8.png", dpi=200)
    plt.close()
    return rows


# Risk Concentration ; Full 16-category Pareto + a normalized Herfindahl-Hirschman Index
def pareto_hhi(samp):
    cnt = samp.cat.value_counts()
    total = len(samp)
    cum_pct = 100 * cnt.cumsum() / total

    shares = cnt / total
    hhi = (shares**2).sum()
    ncat = len(cnt)
    hhi_norm = (hhi - 1 / ncat) / (1 - 1 / ncat)
    print(f"normalized hhi={hhi_norm:.3f} across {ncat} categories")

    x = range(len(cnt))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax2 = ax.twinx()
    ax.bar(x, cnt.values, color=BLUE, width=0.62, zorder=3)
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 0.6, str(v), ha="center", va="bottom", color=BLUE, fontsize=13)
    ax2.plot(x, cum_pct.values, "-o", color="#111111", ms=5, lw=1.6, zorder=4)
    ax2.axhline(80, ls="--", color="#B5B5B5", lw=1.4, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cnt.index, rotation=38, ha="right", fontsize=12.5)
    ax.set_ylabel("Count", fontsize=15)
    ax2.set_ylabel("Cumulative %", fontsize=15)
    plt.tight_layout()
    plt.savefig("chart9.png", dpi=200)
    plt.close()
    return hhi_norm


# helper
def get_merchant_col(df):
    m = df.merch.astype(str).str.strip()
    return m.where(m != "No specific merchant")


# Concentrated merchant risk -- % of merchants that show up 2+ times, and how much of the total incident volume those repeat merchants make up
def merchant_concentration(full, samp):
    def get_stats(df):
        mb = df[df.owner == "Merchant"].copy()
        named = get_merchant_col(mb).dropna()
        cnt = named.value_counts()
        rep = cnt[cnt >= 2]
        pct_rep_merchants = 100 * len(rep) / len(cnt)
        pct_rep_incidents = 100 * rep.sum() / cnt.sum()
        return len(cnt), pct_rep_merchants, pct_rep_incidents

    n_full, full_m_pct, full_i_pct = get_stats(full)
    n_samp, samp_m_pct, samp_i_pct = get_stats(samp)
    print(
        f"full: {n_full} merchants, {full_m_pct:.1f}% repeat, {full_i_pct:.1f}% of incidents from repeats"
    )
    print(
        f"samp: {n_samp} merchants, {samp_m_pct:.1f}% repeat, {samp_i_pct:.1f}% of incidents from repeats"
    )
    groups = [
        "Merchants with 2+ incidents\n(% of all named merchants)",
        "Incidents caused by those\nrepeat merchants (% of total)",
    ]
    full_vals = [full_m_pct, full_i_pct]
    samp_vals = [samp_m_pct, samp_i_pct]
    x = np.arange(len(groups))
    w = 0.32
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.bar(
        x - w / 2,
        full_vals,
        w,
        color=BLUE,
        label=f"Full population (n={n_full} merchants)",
    )
    ax.bar(
        x + w / 2,
        samp_vals,
        w,
        color=LTBLUE,
        label=f"137-incident sample (n={n_samp} merchants)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)
    onlyx(ax)
    ax.set_yticks([])
    ax.legend(
        frameon=False, fontsize=13, loc="upper center", bbox_to_anchor=(0.5, 1.20)
    )
    ax.set_title(
        "Merchant Concentration: A Few Repeat Merchants\nDrive Most Incidents",
        fontsize=20,
        color=BLACK,
        pad=68,
    )
    plt.tight_layout()
    plt.savefig("chart10.png", dpi=200)
    plt.close()


# Shopify's Exposure -- % of Shopify's merchant-caused incidents that are misconfiguration
def shopify_misconfig(full, samp):
    misconfig_cats = {
        "Merchant-side Misconfiguration / Account Issue",
        "Merchant Misconfiguration / Account Issue",
    }

    def get_pct(df):
        mb = df[df.owner == "Merchant"].copy()
        mb["merch_clean"] = get_merchant_col(mb)
        shop = mb[mb.merch_clean == "Shopify"]
        n = len(shop)
        nm = shop.cat.isin(misconfig_cats).sum()
        return n, nm, 100 * nm / n

    n_full, nm_full, pct_full = get_pct(full)
    n_samp, nm_samp, pct_samp = get_pct(samp)
    print(
        f"full: {nm_full}/{n_full} = {pct_full:.1f}% misconfig, samp: {nm_samp}/{n_samp} = {pct_samp:.1f}%"
    )

    table = [[nm_full, n_full - nm_full], [nm_samp, n_samp - nm_samp]]
    odds, p = fisher_exact(table)
    print(f"fisher exact odds={odds:.3f} p={p:.4f} (not different)")

    labels = [f"Full population\n(n={n_full})", f"137-incident sample\n(n={n_samp})"]
    vals = [pct_full, pct_samp]
    colors = [BLUE, LTBLUE]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.bar(labels, vals, color=colors, width=0.5)
    for i, v in enumerate(vals):
        ax.text(
            i,
            v + 1.2,
            f"{v:.1f}%",
            ha="center",
            color=colors[i] if colors[i] != LTBLUE else "#2E9AC7",
            fontsize=19,
        )
    onlyx(ax)
    ax.set_yticks([])
    ax.set_title(
        "Shopify: Share of Merchant-Caused Incidents\nThat Are Misconfiguration",
        fontsize=20,
        color=BLACK,
        pad=14,
    )
    plt.tight_layout()
    plt.savefig("chart11.png", dpi=200)
    plt.close()
    return p


if __name__ == "__main__":
    control_chart(full)
    concentration(samp)
    resolution_top6(full)
    ownership(samp)
    change_governance(samp)
    km_resolution(samp)
    shopify_share(full)
    spike(full)
    change_resolution(samp)
    internal_causes(samp)
    percentiles(full)
    pareto_hhi(samp)
    merchant_concentration(full, samp)
    shopify_misconfig(full, samp)
    print("done, charts saved")

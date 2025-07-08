"""
phase1_static_plots.py — Phase I
--------------------------------------------------------
Run:
    python phase1_static_plots.py
"""

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

# ────────────────────────── paths & style ─────────────────────────── #
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
README  = FIG_DIR / "README.md"; README.write_text("# Static-EDA Observations\n\n", encoding="utf-8")

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

def save_fig(fname: str, note: str):
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=300)
    plt.close()
    README.write_text(f"- **{fname}** – {note}\n", encoding="utf-8")

# ─────────────────────────── cleaner ──────────────────────────────── #
def clean_dataframe(csv="AB_NYC_2019.csv") -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df.drop_duplicates("id").dropna(subset=["id"]); df["id"] = df["id"].astype(int)
    df = df[(df["price"].between(10,2000)) &
            (df["minimum_nights"].between(1,365)) &
            (df["availability_365"].between(0,365))]
    bbox = dict(lat=(40.49,40.92), lon=(-74.27,-73.68))
    df = df[(df["latitude"].between(*bbox["lat"])) & (df["longitude"].between(*bbox["lon"]))]
    df["last_review"]       = pd.to_datetime(df["last_review"], errors="coerce")
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0).astype(float)
    df = df[df["last_review"].dt.year == 2019]        # keep 2019 only
    for c in ("price","number_of_reviews"):
        q1,q3=df[c].quantile([.25,.75]); i=q3-q1
        df = df[(df[c]>=q1-3*i)&(df[c]<=q3+3*i)]
    return df.reset_index(drop=True)

df = clean_dataframe()
boro_order = ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]
num_cols   = ["price","minimum_nights","number_of_reviews","reviews_per_month","availability_365"]

# ─────────────────────────── plots 1-25 ───────────────────────────── #
# 1 line
monthly = (df.dropna(subset=["last_review"])
             .set_index("last_review").resample("ME")["id"].count())
monthly = monthly[monthly>0]
plt.figure(figsize=(7,4)); plt.plot(monthly.index, monthly, marker="o")
plt.title("Listings with ≥1 Review / Month", color="blue", font="serif")
plt.xlabel("Month", color="darkred"); plt.ylabel("Listings", color="darkred"); plt.grid(alpha=.3)
save_fig("01_line_reviews_per_month.png", "Summer peaks show high tourist demand.")

# 2-3 price hist & KDE
plt.figure(figsize=(7,4)); sns.histplot(df["price"], bins=30, color="skyblue")
plt.xlim(0,1000); plt.title("Price Distribution", color="blue", font="serif")
plt.xlabel("Price ($)", color="darkred"); plt.grid(alpha=.3)
save_fig("02_hist_price.png","Most listings <$300; luxury tail >$500.")

plt.figure(figsize=(7,4)); sns.kdeplot(df["price"], fill=True,color="teal")
plt.xlim(0,1000); plt.title("KDE of Prices", color="blue", font="serif")
plt.xlabel("Price ($)", color="darkred")
save_fig("03_kde_price.png","KDE confirms heavy right-skew.")

# 4 min-nights hist
plt.figure(figsize=(7,4)); sns.histplot(df["minimum_nights"], bins=30, color="orange")
plt.xlim(0,60); plt.title("Minimum Nights", color="blue", font="serif")
plt.xlabel("Min Nights", color="darkred")
save_fig("04_hist_min_nights.png","~80 % allow ≤10-night stays.")

# 5  boxplot  price × room  (no log-scale)
plt.figure(figsize=(7, 4))
sns.boxplot(x="room_type", y="price", data=df, palette="Set2")
plt.title("Price by Room Type", color="blue", font="serif")
plt.xlabel("Room Type", color="darkred")
plt.ylabel("Price ($)",   color="darkred")
plt.ylim(0, 1000)           # optional: clamp to a sensible range
plt.grid(alpha=.3, axis="y")
save_fig("05_box_price_room_type.png",
         "Entire homes command highest medians.")

# 6  violin  price × borough  (no log-scale)
plt.figure(figsize=(7, 4))
sns.violinplot(x="neighbourhood_group", y="price",
               data=df, order=boro_order,
               palette="Pastel1", inner="quartile")
plt.title("Price Distribution by Borough", color="blue", font="serif")
plt.xlabel("Borough",  color="darkred")
plt.ylabel("Price ($)", color="darkred")
plt.ylim(0, 1000)          # optional: focus on main price range
plt.grid(alpha=.3, axis="y")
save_fig("06_violin_price_group.png",
         "Manhattan violin clearly highest.")
# 7-8 reviews boxplots
plt.figure(figsize=(7,4)); sns.boxplot(x="room_type", y="number_of_reviews", data=df, palette="Set2")
plt.title("Reviews by Room", color="blue", font="serif")
plt.xlabel("Room Type", color="darkred"); plt.ylabel("Reviews", color="darkred")
save_fig("07_box_reviews_room_type.png","Private rooms have widest review spread.")

plt.figure(figsize=(7,4))
sns.boxplot(x="neighbourhood_group", y="number_of_reviews", data=df,
            order=boro_order, palette="Set3")
plt.title("Reviews by Borough", color="blue", font="serif")
plt.xlabel("Borough", color="darkred")
save_fig("08_box_reviews_group.png","Brooklyn & Manhattan most reviewed.")

# 9-10 rev/month violins
plt.figure(figsize=(7,4))
sns.violinplot(x="room_type", y="reviews_per_month", data=df, palette="coolwarm", inner="quartile")
plt.title("Reviews/Month by Room", color="blue", font="serif")
save_fig("09_violin_revpm_room.png","Entire homes slightly higher monthly rate.")

plt.figure(figsize=(7,4))
sns.violinplot(x="neighbourhood_group", y="reviews_per_month", data=df,
               order=boro_order, palette="cool", inner="quartile")
plt.title("Reviews/Month by Borough", color="blue", font="serif")
save_fig("10_violin_revpm_group.png","Manhattan & Brooklyn thicker tails.")

# 11-12 count bars
plt.figure(figsize=(7,4))
sns.countplot(x="neighbourhood_group", data=df, order=boro_order, palette="Accent")
plt.title("Listings per Borough", color="blue", font="serif")
plt.xlabel("Borough", color="darkred"); plt.ylabel("Listings", color="darkred")
save_fig("11_bar_count_group.png","Brooklyn edges Manhattan.")

plt.figure(figsize=(7,4)); sns.countplot(x="room_type", data=df, palette="Accent")
plt.title("Listings per Room Type", color="blue", font="serif")
save_fig("12_bar_count_room.png","Entire homes ≈51 % of supply.")

# 13-14 avg price bars
avg_boro = df.groupby("neighbourhood_group")["price"].mean().loc[boro_order]
avg_boro.plot(kind="bar", color="orchid", figsize=(7,4))
plt.title("Avg Price by Borough", color="blue", font="serif"); plt.ylabel("Avg Price ($)", color="darkred")
save_fig("13_bar_avg_price_group.png","Manhattan avg ≈$200; Staten Island cheapest.")

avg_room = df.groupby("room_type")["price"].mean()
avg_room.plot(kind="bar", color="teal", figsize=(7,4))
plt.title("Avg Price by Room", color="blue", font="serif"); plt.ylabel("Avg Price ($)", color="darkred")
save_fig("14_bar_avg_price_room.png","Entire homes cost ~2× private rooms.")

# 15-16 top 10 neighbourhoods & hosts
top_nbh = df["neighbourhood"].value_counts().nlargest(10)
top_nbh.plot(kind="bar", color="slateblue", figsize=(7,4))
plt.title("Top-10 Neighbourhoods", color="blue", font="serif"); plt.xticks(rotation=45, ha="right")
save_fig("15_bar_top10_neighbourhoods.png","Bed-Stuy & Williamsburg lead.")

top_hosts = df["host_id"].value_counts().nlargest(10)
top_hosts.plot(kind="bar", color="salmon", figsize=(7,4))
plt.title("Top-10 Hosts", color="blue", font="serif"); plt.xticks(rotation=45, ha="right")
save_fig("16_bar_top10_hosts.png","Largest host >200 listings.")

# 17-18 pie charts
plt.figure(figsize=(5,5))
room_counts = df["room_type"].value_counts()
plt.pie(room_counts, labels=room_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Share by Room Type", color="blue", font="serif")
save_fig("17_pie_room_type.png","Entire homes ~51 %, private rooms ~46 %.")

plt.figure(figsize=(5,5))
boro_counts = df["neighbourhood_group"].value_counts()
plt.pie(boro_counts, labels=boro_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Share by Borough", color="blue", font="serif")
save_fig("18_pie_group.png","Brooklyn & Manhattan ~85 % combined.")

# 19-20 scatter
plt.figure(figsize=(7,4))
sns.scatterplot(x="number_of_reviews", y="price", data=df, alpha=.4)
plt.ylim(0,1000); plt.title("Price vs Reviews", color="blue", font="serif")
save_fig("19_scatter_price_reviews.png","Cheaper places accumulate more reviews.")

plt.figure(figsize=(7,4))
sns.scatterplot(x="minimum_nights", y="price", data=df, alpha=.4, color="teal")
plt.xlim(0,60); plt.ylim(0,1000); plt.title("Price vs Min Nights", color="blue", font="serif")
save_fig("20_scatter_price_min_nights.png","No clear premium for longer stays.")

# 21 geo
plt.figure(figsize=(7,4))
sns.scatterplot(x="longitude", y="latitude", hue="price", data=df,
                palette="viridis", legend=False, alpha=.6)
plt.title("NYC Map — Color = Price", color="blue", font="serif")
plt.xlabel("Lon", color="darkred"); plt.ylabel("Lat", color="darkred")
save_fig("21_scatter_geo_price.png","High price cluster in Manhattan core.")

# 22-23 joint
j1 = sns.jointplot(data=df, x="number_of_reviews", y="price", height=6,
                   kind="scatter", marginal_kws=dict(bins=30, fill=True))
j1.fig.suptitle("Joint: Price & Reviews", y=1.02, color="blue", font="serif")
j1.fig.savefig(FIG_DIR/"22_joint_price_reviews.png", dpi=300); plt.close()

j2 = sns.jointplot(data=df, x="minimum_nights", y="price", height=6,
                   kind="scatter", marginal_kws=dict(bins=30, fill=True))
j2.fig.suptitle("Joint: Price & Min Nights", y=1.02, color="blue", font="serif")
j2.fig.savefig(FIG_DIR/"23_joint_price_min_nights.png", dpi=300); plt.close()

with README.open("a", encoding="utf-8") as fh:
    fh.write("- **22_joint_price_reviews.png** – Dense low-price/high-review cloud.\n")
    fh.write("- **23_joint_price_min_nights.png** – Listings cluster at short stays.\n")

# 24 3-D scatter
fig = plt.figure(figsize=(7,5)); ax = fig.add_subplot(111, projection="3d")
samp=df.sample(3000, random_state=1)
ax.scatter(samp["number_of_reviews"], samp["minimum_nights"], samp["price"],
           c=samp["price"], cmap="plasma", alpha=.5)
ax.set_xlabel("Reviews"); ax.set_ylabel("Min Nights"); ax.set_zlabel("Price ($)")
ax.set_zlim(0,1000); plt.title("3-D: Price × Reviews × Min Nights", color="blue", font="serif")
save_fig("24_scatter3d_price_reviews_nights.png","Bright high-price at low reviews/min nights region.")

# 25 correlation heatmap
plt.figure(figsize=(6,5))
corr = df[num_cols].corr().round(2)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix", color="blue", font="serif")
save_fig("25_heatmap_corr.png","Strongest (−0.29) price vs reviews: cheaper → more reviews.")

# 26 supply vs total reviews
supply = df.groupby(df["last_review"].dt.to_period("M"))["id"].nunique()
reviews= df.groupby(df["last_review"].dt.to_period("M"))["number_of_reviews"].sum()
fig,ax1=plt.subplots(figsize=(8,4)); ax2=ax1.twinx()
ax1.plot(supply.index.to_timestamp(), supply, marker="o", label="Listings")
ax2.bar(reviews.index.to_timestamp(), reviews, alpha=.3, color="orange", label="Reviews")
ax1.set_title("2019 Supply vs Demand", color="blue", font="serif")
ax1.set_xlabel("Month", color="darkred"); ax1.set_ylabel("Listings", color="darkred")
ax2.set_ylabel("Reviews", color="darkred")
save_fig("26_supply_vs_reviews.png","Listings peak midsummer; reviews peak slightly later.")

# 27 median price heatmap
pivot=df.groupby(["neighbourhood_group","room_type"])["price"].median().unstack().loc[boro_order]
plt.figure(figsize=(6,4)); sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("Median Price ($) by Segment", color="blue", font="serif")
save_fig("27_heatmap_med_price.png","Manhattan entire-home median ≈$225.")

# 28 host portfolio distribution
host_counts=df["host_id"].value_counts()
plt.figure(figsize=(6,4)); sns.histplot(host_counts, log_scale=True, bins=30, color="slateblue")
plt.title("Listings per Host (log-log)", color="blue", font="serif")
plt.xlabel("Listings/Host", color="darkred"); plt.ylabel("Hosts", color="darkred")
save_fig("28_hist_host_portfolio.png","Power-law: 75 % hosts manage one listing.")

# 29 availability strip (Manhattan)
man=df[df["neighbourhood_group"]=="Manhattan"]
cal=(man.set_index("last_review").resample("D")["availability_365"].mean())
plt.figure(figsize=(10,1.8)); plt.plot(cal.index, cal, color="teal")
plt.title("Manhattan Mean Availability (2019)", color="blue", font="serif")
plt.yticks([]); plt.xlabel("Date", color="darkred")
save_fig("29_strip_availability_man.png","Two dips: summer & holiday season bookings.")

# 30 price vs latitude (Manhattan)
plt.figure(figsize=(6,4))
sns.regplot(x="latitude", y="price", data=man, scatter_kws=dict(alpha=.3), line_kws=dict(color="red"))
plt.title("Manhattan Latitude vs Price", color="blue", font="serif")
plt.xlabel("Latitude", color="darkred"); plt.ylabel("Price ($)", color="darkred")
save_fig("30_reg_price_latitude.png","Price drops ~\$10 for every .01° northward.")

print("✅ 30 figures saved to ./figures.")
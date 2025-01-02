# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

from trendfilterpy._estimator import TrendFilterRegression

# %%
df = pl.read_csv("../tests/french_AL_data.csv", infer_schema_length=25000)
model = TrendFilterRegression(lam=1000)
model.fit(
    df.select(
        "Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"
    ),
    df.select(pl.col("ClaimAmount").log1p()),
    categorical_features=[1, 6, 7, 9],
)

# %%
test = model.vars_[0]

# %%
plt.plot(test.unique_vals, test.beta.value)

# %%
rng = np.random.default_rng()
n = 10000
X = np.hstack([rng.integers(-100, 100, size=(n, 1)) / 10, rng.integers(-250, 250, size=(n, 1)) / 250])
true_y = 5 + np.sin(X[:, 0]) + np.exp(X[:, 1])
obs_y = true_y + 0.5 * rng.standard_normal(n)

estimator = TrendFilterRegression(lam=100)
estimator.fit(X, obs_y)

# %%
test = estimator.vars_[0]
plt.plot(test.unique_vals, test.beta.value)

# %%
df = pl.read_csv("../tests/Housing.csv").with_columns(cs.string().cast(pl.Categorical))
model = TrendFilterRegression(lam=2)
model.fit(
    df.select(pl.col("area", "bedrooms", "bathrooms", "stories", "parking"), cs.categorical().to_physical()),
    df.select(pl.col("price").log()),
    categorical_features=[5, 6, 7, 8, 9, 10, 11],
)

# %%
test = model.vars_[0]
plt.plot(test.unique_vals, test.beta)

# %%
plt.scatter(df.select(pl.col("price").log()), model.mu_)

# %%
model.vars_[0].predict([-1000000, 14000, 15000, 1000, 5000, 1000000])

# %%
model.vars_[5].predict([0, 1])

# %%
model.score(
    df.select(pl.col("area", "bedrooms", "bathrooms", "stories", "parking"), cs.categorical().to_physical()),
    df.select(pl.col("price").log()),
)

# %%
model.predict(
    df.select(pl.col("area", "bedrooms", "bathrooms", "stories", "parking"), cs.categorical().to_physical())[:10]
)

# %%

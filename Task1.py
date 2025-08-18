import matplotlib.pyplot as plt

# 1. Global Fast-Food Chains (Real Data)
chains = ["McDonald's", "Starbucks", "Subway", "KFC"]
locations = [41822, 36171, 37000, 30000]  # real approximations

# 2. Favorite Fast-Food Chains (Survey Example)
favorites = ["McDonald's", "Subway", "Starbucks", "KFC"]
votes = [55, 35, 30, 25]  # hypothetical survey data

# 3. Daily Calorie Availability by Region (Real Data)
regions = ["High-Income Countries", "Low-Income (e.g., Nigeria)"]
calories_availability = [3914, 2469]  # kcal/person/day

# 4. Personal Daily Calorie Consumption (Histogram Sample)
calories_consumed = [
    2200, 2400, 2600, 2000, 2300, 2500, 2700, 2100,
    2800, 3000, 2500, 2400, 2600, 3100, 2900, 2700
]

# Create 2x2 plot grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Food-Related Insights with Real Data", fontsize=18, fontweight="bold")

# Chart 1: Global Fast-Food Chains by Location Count
axs[0,0].bar(chains, locations, color="#FF8C00", edgecolor="black")
axs[0,0].set_title("Global Fast-Food Chains (by Outlet Count)")
axs[0,0].set_ylabel("Number of Locations")
axs[0,0].grid(axis="y", linestyle="--", alpha=0.5)

# Chart 2: Favorite Fast-Food Chains (Survey)
axs[0,1].bar(favorites, votes, color="#4682B4", edgecolor="black")
axs[0,1].set_title("Favorite Fast-Food Chains (Survey Sample)")
axs[0,1].set_ylabel("Votes")
axs[0,1].grid(axis="y", linestyle="--", alpha=0.5)

# Chart 3: Calorie Availability by Region
axs[1,0].bar(regions, calories_availability, color="#3CB371", edgecolor="black")
axs[1,0].set_title("Daily Calorie Availability by Region")
axs[1,0].set_ylabel("kcal per Day")
axs[1,0].grid(axis="y", linestyle="--", alpha=0.5)

# Chart 4: Daily Calorie Consumption Distribution
axs[1,1].hist(calories_consumed, bins=6, color="#D2691E", edgecolor="black", alpha=0.8)
axs[1,1].set_title("Personal Daily Calorie Consumption")
axs[1,1].set_xlabel("Calories per Day")
axs[1,1].set_ylabel("Number of People")
axs[1,1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

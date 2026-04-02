#Importing the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#using this to make the background cleaner
sns.set_theme()

# The path for the cleaned stock prices data that was made in task 1
input_file = "Data/cleaned_stock_prices.csv"

# Loading dataset
df = pd.read_csv(input_file)

# Converting date column once again to avoid issues and match the csv file in excel
df["date"] = pd.to_datetime(df["date"])

print("Cleaned dataset loaded successfully.\n")

#Line chart creation for showing Stock Closing Price Over Time

plt.figure()

#Choosing the x and y for this graph, graph title and axis titles
plt.plot(df["date"], df["close"])

plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")

#fixing issue with graph looking messy and hard to read
plt.xticks(rotation=45)

#saving an image of the graph
plt.savefig("Level1/plots/price_trend.png")
plt.close()

print("Saved: price_trend.png")


# Making the bar chart to showcase the Average closing price per month

plt.figure()

# Extracting the months from the date column and converting numbers for months into words
df["month"] = df["date"].dt.strftime("%b")

# Making the order of months since it was showing in alphabetical order
month_order = ["Jan", "Feb", "Mar", 
               "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", 
               "Oct", "Nov", "Dec"]

df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

# Group by the months of the year and getting the average
monthly_avg = df.groupby("month")["close"].mean()

#Choosing the x and y for this graph, graph title and axis titles
plt.bar(monthly_avg.index, monthly_avg.values)

plt.title("Average Monthly Closing Price")
plt.xlabel("Month")
plt.ylabel("Average Closing Price")

# Saving an image
plt.savefig("Level1/plots/monthly_avg_price.png")
plt.close()

print("Saved: monthly_avg_price.png")

# Making the scatter plot graph to show Open vs Close Price

plt.figure()

#Choosing the x and y for this graph, graph title and axis titles
plt.scatter(df["open"], df["close"])

plt.title("Open vs Close Price")
plt.xlabel("Open Price")
plt.ylabel("Close Price")

#saving an image
plt.savefig("Level1/plots/open_vs_close.png")
plt.close()

print("Saved: open_vs_close.png")
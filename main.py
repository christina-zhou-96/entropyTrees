import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import tree
from tabulate import tabulate

# The question is this: given some physical properties of the diamond, can we classify it as luxury?
# https://bookdown.org/yih_huynh/Guide-to-R-Book/diamonds.html

# 1. Load in diamonds
diamonds_full = sns.load_dataset('diamonds')

# Define luxury column
diamonds_full['luxury'] = diamonds_full['price'].apply(lambda x: 1 if x > 5324 else 0)

# Calculate the proportions of luxury and non-luxury diamonds
luxury_count = diamonds_full['luxury'].sum()
non_luxury_count = diamonds_full.shape[0] - luxury_count
luxury_proportion = luxury_count / diamonds_full.shape[0]
non_luxury_proportion = 1 - luxury_proportion

# Set the number of samples you want to select
n_samples = 100

# Calculate the number of luxury and non-luxury samples to select
n_luxury_samples = int(n_samples * luxury_proportion)
n_non_luxury_samples = n_samples - n_luxury_samples

# Select luxury and non-luxury samples separately
luxury_samples = diamonds_full[diamonds_full['luxury'] == 1].sample(n_luxury_samples, random_state=42)
non_luxury_samples = diamonds_full[diamonds_full['luxury'] == 0].sample(n_non_luxury_samples, random_state=42)

# Concatenate luxury and non-luxury samples to get the final dataset
diamonds = pd.concat([luxury_samples, non_luxury_samples], axis=0)

# Shuffle the selected samples
diamonds = diamonds.sample(frac=1, random_state=42).reset_index(drop=True)

print(tabulate(diamonds[:10], headers='keys', tablefmt='psql'))

# 2. Change the categoricals to numericals
# Create a dictionary to map the categorical values to numerical values
# cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
# color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
# clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
#
# # Replace the categorical values with the corresponding numerical values
# diamonds['cut'] = diamonds['cut'].map(cut_mapping).astype('int')
# diamonds['color'] = diamonds['color'].map(color_mapping).astype('int')
# diamonds['clarity'] = diamonds['clarity'].map(clarity_mapping).astype('int')

# Remove categoricals for now
diamonds = diamonds.drop(columns=['cut','color','clarity'])

# Remove price - it's embedded in luxury
diamonds = diamonds.drop(columns=['price'])

# sns.pairplot(diamonds, hue='luxury')
# # plt.show()
# plt.savefig('diamonds.png')

# 3. Graph a decision tree surface


# Calculations for my blog post

# (A) Entropy score
# import math
# (-3/4)*(math.log2(3/4))
# .311
# (-1/2)*(math.log2(1/2))
# .5
# Confirm that .311 + .5 = .81 as shown

# (B) Information gain
# information gain = entropy (parent) - entropy (children)
# carat information gain (chosen: .49)
# .81 - (((.29 * 77)+(.43*23))/100)

# depth information gain (not chosen: .04)
# .81 - (((.89 * 71)+(.48*29))/100)

# There are other ways to measure information gain and entropy besides the ones shown here!

# Define the function for the decision tree model
def decision_tree_model(X, y, xx, yy):
    clf = DecisionTreeClassifier().fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return Z


# Plot pair plot
g = sns.pairplot(diamonds, hue='luxury')
# plt.show()
plt.savefig('diamonds.png')

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(diamonds.drop('luxury', axis=1))
y = diamonds['luxury']

# Create a DataFrame with standardized features
diamonds_std = diamonds.copy()
diamonds_std.iloc[:, :-1] = X

# Plot the pairplot
g = sns.pairplot(diamonds_std, hue='luxury')
g.fig.suptitle('Decision surface for diamonds pair plot',  y=1.001)
# plt.show()

# # For my blog post, I created three extra trees with limited information
# two_var_diamonds = diamonds[['carat','depth','luxury']]
# carat_diamonds = diamonds[['carat','luxury']]
# depth_diamonds = diamonds[['depth','luxury']]
#
# # Standardize the datasets
# scaler = StandardScaler()
# two_var_X = scaler.fit_transform(two_var_diamonds.drop('luxury', axis=1))
# carat_X = scaler.fit_transform(carat_diamonds.drop('luxury', axis=1))
# depth_X = scaler.fit_transform(depth_diamonds.drop('luxury', axis=1))
#
# y = two_var_diamonds['luxury']
#
# # Create DataFrames with standardized features
# two_var_diamonds_std = two_var_diamonds.copy()
# two_var_diamonds_std.iloc[:, :-1] = two_var_X
#
# carat_diamonds_std = carat_diamonds.copy()
# carat_diamonds_std.iloc[:, :-1] = carat_X
#
# depth_diamonds_std = depth_diamonds.copy()
# depth_diamonds_std.iloc[:, :-1] = depth_X

# # View the decision tree sklearn decided given the criterion of entropy
# two_var_clf = DecisionTreeClassifier(criterion='entropy',max_depth=1).fit(two_var_X, y)
# tree.plot_tree(two_var_clf, precision=2)
# # plt.show()
# plt.savefig('tree plot two var.png')
#
# # View single decision tree, carat only
# carat_clf = DecisionTreeClassifier(criterion='entropy',max_depth=1).fit(carat_X, y)
# tree.plot_tree(carat_clf, precision=2)
# plt.show()
#
# # View single decision tree, depth only
# depth_clf = DecisionTreeClassifier(criterion='entropy',max_depth=1).fit(depth_X, y)
# tree.plot_tree(depth_clf, precision=2)
# # plt.show()
# plt.savefig('tree plot depth.png')

# Iterate through the pairplot axes
n_features = len(diamonds_std.columns) - 1
for i, ax in enumerate(g.axes.flat):
    x_idx = i % n_features
    y_idx = i // n_features

    # Skip the diagonal plots
    if x_idx == y_idx:
        continue

    # Get the limits for the x and y axes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Create a meshgrid
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # Get the decision tree model and predict on the meshgrid
    X_subset = X[:, [x_idx, y_idx]]
    Z = decision_tree_model(X_subset, y, xx, yy)

    # Plot the decision surface
    ax.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Paired)

# Show the pairplot
# plt.show()

# Save the pairplot
plt.savefig('diamonds decision tree surface pairplot.png', bbox_inches='tight')
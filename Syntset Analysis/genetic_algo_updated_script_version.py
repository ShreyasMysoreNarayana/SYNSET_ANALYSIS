# -*- coding: utf-8 -*-
"""Genetic Algo Updated-Script_version.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1C5jOjKv286IDqKm1yFGCFWkiu38Ci-Iz
"""

import pandas as pd
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and prepare data
data = pd.read_csv("synth_seg.csv")  # Make sure the dataset is available in this path Use the desired dataset - ORIGINAL,RESAMPLED,UNDERSAMPLED
y = data['decision'].astype(bool)
X = data.drop(columns=['Subject', 'decision', 'neuropsych_score'], axis=1)
n_features = X.shape[1]

# Doctor's preferred features (to be handled differently)
preferred_features = [
    'general grey matter', 'general csf', 'cerebellum', 'hippocampus+amygdala',
    'left hippocampus', 'right hippocampus', 'ctx-lh-entorhinal', 'ctx-rh-entorhinal'
]

# Mapping feature names to their indices in the dataset
preferred_feature_indices = [X.columns.get_loc(feature) for feature in preferred_features]

# User input to decide the option (Option 1 or Option 2)
print("Choose feature handling option for doctor's preferred features:")
print("1: Keep doctor's preferred features as non-mutable (always ON)")
print("2: Initialize doctor's preferred features as ON but allow mutation")
user_choice = int(input("Enter your choice (1 or 2): "))

# Genetic Algorithm Functions From Scratch
def create_individual(n_features, preferred_feature_indices, user_choice):
    # Creates a random individual (feature subset).
    individual = [random.randint(0, 1) for _ in range(n_features)]

    if user_choice == 1:  # Option 1: Set preferred features to 1 and non-mutable
        for idx in preferred_feature_indices:
            individual[idx] = 1

    elif user_choice == 2:  # Option 2: Set preferred features to 1 initially, but allow mutation later
        for idx in preferred_feature_indices:
            individual[idx] = 1

    return individual

def apply_lda(numeric_df, y):
    n_components = 1
    lda = LDA(n_components=n_components)
    classifier = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=5)
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for train_index, test_index in skf.split(numeric_df, y):
        X_train_fold, X_test_fold = numeric_df.iloc[train_index], numeric_df.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        X_lda = lda.fit_transform(X_train_fold, y_train_fold)
        lda_df = pd.DataFrame(data=X_lda, columns=[f'LD{i+1}' for i in range(n_components)])
        classifier.fit(lda_df, y_train_fold)

        X_test_lda = lda.transform(X_test_fold)
        X_test_df = pd.DataFrame(data=X_test_lda, columns=[f'LD{i+1}' for i in range(n_components)])
        y_pred_fold = classifier.predict(X_test_df)

        precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='macro'))
        recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='macro'))
        accuracy_scores.append(classifier.score(X_test_df, y_test_fold))

    return np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)

def evaluate(individual, X, y):
    # Evaluates the fitness of an individual.
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    num_selected_features = len(selected_features)

    if num_selected_features == 0:
        return 0  # Avoid selecting no features

    X_selected = X.iloc[:, selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Apply LDA using the apply_lda function
    accuracy, precision, recall = apply_lda(pd.DataFrame(X_scaled), y)

    # Calculate fitness with penalty for more features
    fitness = accuracy + precision + recall + (1 / num_selected_features)
    return fitness

def crossover(parent1, parent2):
    # Performs crossover at a random point.
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutation(individual, indpb=0.05, preferred_feature_indices=None, user_choice=None):
    # Performs flip-bit mutation at a single random point.
    if random.random() < indpb:  # Check if mutation should occur
        point = random.randint(0, len(individual) - 1)  # Select a random point

        # Option 1: Keep doctor's preferred features as non-mutable
        if user_choice == 1 and point in preferred_feature_indices:
            return individual  # Skip mutation for preferred features

        # Option 2: Allow mutation on all features, including the preferred ones
        individual[point] = 1 - individual[point]  # Flip the bit at the selected point

    return (individual,)

# Main Genetic Algorithm Call Function
def main(X, y, pop_size=100, ngen=10, cxpb=0.5, mutpb=0.2, top_percent=0.10, tolerance=1e-4):
    random.seed(42)

    # Initialize population
    population = [create_individual(n_features, preferred_feature_indices, user_choice) for _ in range(pop_size)]

    # Evaluate initial population
    fitnesses = [evaluate(ind, X, y) for ind in population]

    # Store best individuals, their generation, and their fitness values
    best_individuals = []
    best_generations = []
    best_fitnesses = []

    for gen in range(ngen):
        # Selection (Select top individuals directly)
        num_top = int(pop_size * top_percent)
        top_indices = np.argsort(fitnesses)[-num_top:]
        offspring = [population[i] for i in top_indices]

        # Crossover to create the rest of the offspring
        num_crossovers = pop_size - num_top  # Number of crossovers needed
        for _ in range(num_crossovers // 2):  # // 2 because each crossover creates 2 offspring
            parent1, parent2 = random.sample(offspring, 2)  # Select parents from the top individuals
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # Mutation
        for i in range(num_top, len(offspring)):  # Mutate only the new offspring
            if random.random() < mutpb:
                offspring[i], = mutation(offspring[i], preferred_feature_indices=preferred_feature_indices, user_choice=user_choice)

        # Evaluate offspring
        fitnesses = [evaluate(ind, X, y) for ind in offspring]

        # Replace population with offspring (Elitism: Keep top individuals)
        population[:] = offspring

        # Store the top individuals and their generation
        top_indices = np.argsort(fitnesses)[-num_top:]
        for i in top_indices:
            best_individuals.append(population[i])
            best_generations.append(gen)  # Store the generation

        # Store fitness of the best individual in the current generation
        best_fitnesses.append(max(fitnesses))

        # Print generation information for the top individual
        best_individual_current_gen = max(population, key=lambda ind: evaluate(ind, X, y))

        # Calculate X_scaled for the best individual
        selected_features = [i for i, bit in enumerate(best_individual_current_gen) if bit == 1]
        X_selected = X.iloc[:, selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        accuracy_top = apply_lda(pd.DataFrame(X_scaled), y)[0]  # Get accuracy for the top individual
        print(f"Generation {gen}: Fitness = {best_fitnesses[-1]}, Accuracy = {accuracy_top}")

        # Convergence check
        if gen > 1 and abs(best_fitnesses[-1] - best_fitnesses[-2]) < tolerance:
            print(f"Converged at generation {gen}")
            break

    return best_individuals, best_generations, best_fitnesses  # Return all three lists

# Run the GA
best_individuals, best_generations, best_fitnesses = main(X, y, pop_size=100, top_percent=0.10)  # Use top 10%

# Print the final best solution and its generation
best_individual = max(best_individuals, key=lambda ind: evaluate(ind, X, y))
best_index = best_individuals.index(best_individual)
best_generation = best_generations[best_index]

selected_features_lda = [i for i, bit in enumerate(best_individual) if bit == 1]
print(f"Final Best Solution (found at generation {best_generation}):")
print(f"  Selected Features: {X.columns[selected_features_lda].tolist()}")
print(f"  Number of features selected: {len(selected_features_lda)}")

# Print fitness values across generations
print(f"Best fitness values across generations: {best_fitnesses}")
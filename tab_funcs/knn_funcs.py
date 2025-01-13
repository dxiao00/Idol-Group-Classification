from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import numpy as np 

def n_neighbors_cv(X_train, y_train): 
    neighbors = [5, 10, 15, 20, 25, 30, 35, 40]
    accuracies = []
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        cv_accuracy = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        accuracies.append(cv_accuracy.mean())

    cross_val_plot = go.Figure()
    cross_val_plot.add_trace(go.Scatter(
        x=neighbors, 
        y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='blue')
    ))
    cross_val_plot.update_layout(
        title='KNN: Number of Neighbors vs Cross-Validation Accuracy',
        xaxis_title='Number of Neighbors',
        yaxis_title='Cross-Validation Accuracy',
    )

    opt_k = neighbors[np.argmax(accuracies)]

    return cross_val_plot, opt_k

import numpy as np 
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import pandas as pd 

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def plotly_decision_boundary_with_margins(model, selected_features, test, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test):
    
    # show decision boundary only when 2 features are selected 
    if len(selected_features) != 2:
        decision_fig = {
            'data': [],
            'layout': {
                'title': 'Decision Boundary (requires 2 features)',
                'xaxis': {'title': 'Feature 1'},
                'yaxis': {'title': 'Feature 2'},
                'annotations': [{
                    'text': 'Select exactly 2 features to display the decision boundary.',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            }
        }
        return decision_fig

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # define decision grid 
    x_min, x_max = X_train_selected.iloc[:, 0].min() - 1, X_train_selected.iloc[:, 0].max() + 1
    y_min, y_max = X_train_selected.iloc[:, 1].min() - 1, X_train_selected.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid, columns=selected_features)

    # make predictions on the grid 
    Z = model.decision_function(grid_df).reshape(xx.shape)

    # if test, plot test points
    if test:
        scatter_x = X_test_selected.iloc[:, 0]
        scatter_y = X_test_selected.iloc[:, 1]
        scatter_labels = y_test

        # get test accuracy
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        songs = song_test
        artists = artist_test

    # else, plot train points
    else:
        scatter_x = X_train_selected.iloc[:, 0]
        scatter_y = X_train_selected.iloc[:, 1]
        scatter_labels = y_train

        # plot train accuracy
        y_pred = model.predict(X_train_selected)
        accuracy = accuracy_score(y_train, y_pred)
        songs = song_train
        artists = artist_train

    # final figure
    decision_fig = go.Figure(
        data=[
            # decision boundary
            go.Contour(
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                z=Z,
                colorscale='RdBu',
                showscale=False,
                opacity=0.6, 
                contours=dict(
                    start=-2,
                    end=2,
                    size=0.1,
                    coloring='fill',  
                ),
            ),
            # margins 
            go.Contour(
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                z=Z,
                colorscale='RdBu',
                showscale=False,
                contours=dict(
                    start=-1.0,
                    end=1.0,
                    size=1.0,
                    coloring='lines', 
                ),
                line_width=2,
            ),
            # scatter points
            go.Scatter(
                x=scatter_x,
                y=scatter_y,
                mode='markers',
                marker={
                    'color': scatter_labels,
                    'symbol': 'circle',
                    'line': {'width': 1}
                },
                hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artists), np.array(songs))]
            )
        ],
        layout={
            'title': f'Decision Boundary with Margins (Accuracy: {accuracy * 100:.2f}%)',
            'xaxis': {'title': selected_features[0]},
            'yaxis': {'title': selected_features[1]},
        }
    )
    
    return decision_fig


def svm_grid_search(grid_search, X_train, y_train, X_test, y_test, kernel, selected_features,
                    song_train, song_test, artist_train, artist_test):

    if grid_search == 'no': 
        grid_search_plot =  {
            'data': [],
            'layout': {
                'title': 'No Plot (Grid Search Disabled)',
                'xaxis': {'title': 'Feature 1'},
                'yaxis': {'title': 'Feature 2'}
            }
        }
    
        return grid_search_plot, grid_search_plot 

        
    # grid search for best hyperparameters

    if kernel == 'linear':
        param_grid = {
        'C': [0.01, 0.1, 1, 10],
    }

    elif kernel == 'poly': 
        param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'degree': [2, 3, 4],
        # 'gamma': [0.01, 0.1, 1, 10],
    }
        
    elif kernel == 'rbf':
        param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': [0.01, 0.1, 1, 10],
    }

    grid = GridSearchCV(SVC(kernel=kernel), param_grid, cv=5)
    grid.fit(X_train, y_train)

    # fit the model with best hyperparameters
    best_model = grid.best_estimator_
    best_hyperparameters = best_model.get_params()
    best_model = SVC(**best_hyperparameters)
    best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # only show decision boundary if there are 2 selected features 
    if len(selected_features) == 2:

        # training plot 
        train_fig = plotly_decision_boundary_with_margins(best_model, selected_features, False, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)

        if kernel == 'linear': 
            train_fig.update_layout(
                title=f"SVM Decision Boundary (Accuracy: {round(train_accuracy,3)})<br>Kernel: {kernel}, C: {best_model.C}"
            )

        elif kernel == 'poly': 
            train_fig.update_layout(
                title=f"SVM Decision Boundary (Accuracy: {round(train_accuracy,3)}<br>Kernel: {kernel}, C: {best_model.C}, Degree: {best_model.degree}, Gamma: {best_model.gamma}"
            )   
        
        elif kernel == 'rbf':
            train_fig.update_layout(
                title=f"SVM Decision Boundary (Accuracy: {round(train_accuracy,3)}<br>Kernel: {kernel}, C: {best_model.C}, Gamma: {best_model.gamma}"
            )

        # test plot 
        test_fig = plotly_decision_boundary_with_margins(best_model, selected_features, True, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)

        if kernel == 'linear': 
            test_fig.update_layout(
                title=f"SVM Decision Boundary (Accuracy: {round(test_accuracy,3)})<br>Kernel: {kernel}, C: {best_model.C}"
            )

        elif kernel == 'poly': 
            test_fig.update_layout(
                title=f"SVM Decision Boundary (Accuracy: {round(test_accuracy,3)}<br>Kernel: {kernel}, C: {best_model.C}, Degree: {best_model.degree}, Gamma: {best_model.gamma}"
            )   
        
        elif kernel == 'rbf':
            test_fig.update_layout(
                title=f"SVM Decision Boundary (Accuracy: {round(test_accuracy,3)}<br>Kernel: {kernel}, C: {best_model.C}, Gamma: {best_model.gamma}"
            )

        return train_fig, test_fig 

    
    else: 
        fig =  {
            'data': [],
            'layout': {
                'title': 'No Plot (Not 2 Selected Features)',
                'xaxis': {'title': 'Feature 1'},
                'yaxis': {'title': 'Feature 2'}
            }
        }

        return fig, fig 



    

    
    


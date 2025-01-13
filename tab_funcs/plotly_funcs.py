from sklearn.metrics import confusion_matrix
import numpy as np 
import plotly.graph_objects as go 
import pandas as pd 
from sklearn.metrics import accuracy_score


def plotly_confusion_matrix(y_true, y_pred): 

    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)

    # plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm, 
        x=['Predicted: 0', 'Predicted: 1'],  
        y=['Actual: 0', 'Actual: 1'],      
        colorscale='Blues',                 
        zmin=0, zmax=np.max(cm),           
        colorbar=dict(title="Count"),       
        text=cm,                           
        texttemplate="%{text}",             
        hovertemplate='Count: %{text}'     
    ))

    # accuracy annotation
    fig.add_annotation(
        x=1, y=1, 
        xref="x domain", yref="y domain", 
        text=f"Accuracy: {accuracy*100:.2f}%", 
        showarrow=False, font=dict(size=16, color="black"), 
        align="center", bgcolor="white", borderpad=5
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Labels"),
        yaxis=dict(title="True Labels"),
        autosize=False,
        width=500,
        height=500,
    )

    return fig


def plotly_decision_boundary(model, selected_features, test, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test):
    
    # need 2 selected features to show decision boundary
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

    # define grid for predictions 
    x_min, x_max = X_train_selected.iloc[:, 0].min() - 1, X_train_selected.iloc[:, 0].max() + 1
    y_min, y_max = X_train_selected.iloc[:, 1].min() - 1, X_train_selected.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid, columns=selected_features)

    # make predictions on the grid 
    class_labels = model.predict(grid_df).reshape(xx.shape)

    # plot test points 
    if test:
        scatter_x = X_test_selected.iloc[:, 0]
        scatter_y = X_test_selected.iloc[:, 1]
        scatter_labels = y_test

        # test set accuracy 
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        songs = song_test
        artists = artist_test

    # plot training points 
    else:
        scatter_x = X_train_selected.iloc[:, 0]
        scatter_y = X_train_selected.iloc[:, 1]
        scatter_labels = y_train

        # training accuracy
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
                z=class_labels,
                colorscale='RdBu',
                contours={'showlabels': False},  # Do not show probabilities
            ),
            # plot points
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
            'title': f'Decision Boundary (Accuracy: {accuracy * 100:.2f}%)',
            'xaxis': {'title': selected_features[0]},
            'yaxis': {'title': selected_features[1]},
            
        }
    )
    
    return decision_fig

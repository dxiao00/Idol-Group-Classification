
import plotly.graph_objects as go 
import numpy as np 

def plot_feature_importances(ensemble, model, feature_names):
    # only have feature importance for random forest 
    if ensemble == 'random_forest':
        importances = model.feature_importances_

        # descending order 
        sorted_indices = np.argsort(importances)[::-1]  
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]

        fig = go.Figure(
            go.Bar(x=sorted_feature_names, y=sorted_importances)
        )

        fig.update_layout(
            title="Feature Importances",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            template="plotly_white"
        )
    
    else:
        fig = go.Figure()
        fig.update_layout(
            title="No Feature Importances Available",
            template="plotly_white"
        )

    return fig
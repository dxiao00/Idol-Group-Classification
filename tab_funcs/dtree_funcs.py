import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import base64
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def plot_impurity_metric(metric):
    # proportion class 1
    p1 = np.linspace(0, 1, 100)
    p0 = 1 - p1  

    if metric == 'gini':
        values = 2 * p1 * p0
        metric_name = 'Gini Impurity'

    elif metric == 'entropy':
        # avoid taking log of 0 
        values = -p1[1:-1] * np.log(p1[1:-1]) - p0[1:-1] * np.log(p0[1:-1])
        metric_name = 'Cross Entropy'

    elif metric == 'classification_error':
        values = 1 - np.maximum(p1, p0)
        metric_name = 'Classification Error'

    else:
        raise ValueError("Invalid metric. Choose 'gini', 'cross_entropy', or 'classification_error'.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p1, y=values, mode='lines', name=metric_name,))
    fig.update_layout(
        title=f"{metric_name} as a Function of Class 1 Proportion",
        xaxis_title="Proportion of Class 1 (p1)",
        yaxis_title="Metric Value",
        legend_title="Metric",
        template="plotly_white"
    )

    return fig

def plot_decision_tree(dtree, selected_features): 
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(dtree, filled=True, feature_names=selected_features, class_names=['non idol group', 'idol group'], rounded=True, ax=ax)
    
    # converts plt figure into a base64-encoded png image string 
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    tree_img = f"data:image/png;base64,{img_base64}"

    return tree_img

def prune_tree(pruning_method, dtree_cross_val, impurity, max_depth, node_cost, X_train, y_train):
    if pruning_method == 'max_depth': 
        if dtree_cross_val == 'no':
            dtree = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=impurity)
        elif dtree_cross_val == 'yes': 
            param_grid = {'max_depth': [1, 2, 3, 4, 5]}  
            grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0, criterion=impurity), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_max_depth = grid_search.best_params_['max_depth']
            dtree = DecisionTreeClassifier(random_state=0, max_depth=best_max_depth, criterion=impurity)
    elif pruning_method == 'ccp':
        if dtree_cross_val == 'no':
            dtree = DecisionTreeClassifier(random_state=0, ccp_alpha=node_cost, criterion=impurity)
        elif dtree_cross_val == 'yes': 
            dtree = DecisionTreeClassifier(random_state=0, criterion=impurity)
            dtree.fit(X_train, y_train)

            # calculates a sequence of possible pruning options for the decision tree
            path = dtree.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas = path.ccp_alphas
            param_grid = {'ccp_alpha': ccp_alphas}
            grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0, criterion=impurity), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_alpha = grid_search.best_params_['ccp_alpha']
            dtree = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha, criterion=impurity)

    return dtree 
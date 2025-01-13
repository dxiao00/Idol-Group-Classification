
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

import numpy as np 
import plotly.graph_objects as go 
from scipy.stats import multivariate_normal
from scipy.stats import norm


def lda_3d(means, covariances, priors, features):
    # means: list of mean vectors for each class ([[mean_x1, mean_y1], [mean_x2, mean_y2]])
    # covariances: list of covariance matrices for each class
    # priors: list of prior probabilities for each class
    
    labels = ['non idol group', 'idol group']

    means = np.array(means)
    mean_x = means[:, 0]
    mean_y = means[:, 1]

    # plot the distributions on a range based on the means
    margin = 2
    x_range = [mean_x.min() - margin, mean_x.max() + margin]
    y_range = [mean_y.min() - margin, mean_y.max() + margin]

    # define grid 
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    posteriors = []
    for mean, cov, prior in zip(means, covariances, priors):
        rv = multivariate_normal(mean, cov)
        # multiply the density by the prior to get unnormalized posterior
        posterior = rv.pdf(pos) * prior
        posteriors.append(posterior)
    max_posterior = max([posterior.max() for posterior in posteriors])  # global maximum to have same scale for colorbar for both classes


    fig = go.Figure()
    colors = ['Burg', 'Blugrn'] 
    # plot each class 
    for posterior, mean, prior, label, color in zip(posteriors, means, priors, labels, colors):
        fig.add_trace(go.Surface(
            z=posterior,
            x=X,
            y=Y,
            colorscale=color, 
            cmin=0,
            cmax=max_posterior, # normalize colors to global max
            colorbar=dict(title="Density", len=0.5, y=0.7 if label == 'idol group' else 0.1),  
            opacity=0.7,
            showscale=True,  
            hovertemplate=(
                f"<b>Class:</b> {label}<br>"  
                f"<b>{features[0]}:</b> %{{x:.2f}}<br>" 
                f"<b>{features[1]}:</b> %{{y:.2f}}<br>"
                "<b>Density:</b> %{z:.4f}<extra></extra>" # z coordinate
            ),
            name=f"{label}" 
        ))

    fig.update_layout(
        title="Unnormalized Posteriors P(X=x|Y=k)P(Y=k)",
        scene=dict(
            xaxis_title=f"{features[0]}",
            yaxis_title=f"{features[1]}",
            zaxis_title="Unnormalized Posterior"
        ),
        legend_title="Classes",
        template="plotly_white"
    )
    
    return fig


def lda_2d(means, variances, priors):

    labels = ['non idol', 'idol']

    std_devs = [np.sqrt(var) for var in variances]
    min_mean = min(mean[0] for mean in means)
    max_mean = max(mean[0] for mean in means)
    margin = 3  
    x = np.linspace(min_mean - margin, max_mean + margin, 1000)

    # calculate the unnormalized posteriors P(X=x|Y=k)P(Y=k)
    posteriors = [
        norm.pdf(x, loc=mean[0], scale=std_dev) * prior
        for mean, std_dev, prior in zip(means, std_devs, priors)
    ]

    # intersection points
    diff = posteriors[0] - posteriors[1]
    intersection_indices = np.where(np.diff(np.sign(diff)))[0]
    intersection_points = x[intersection_indices]


    fig = go.Figure()
    for posterior, mean, std_dev, prior, label in zip(posteriors, means, std_devs, priors, labels):
        fig.add_trace(go.Scatter(
            x=x,
            y=posterior,
            mode='lines',
            name=f"{label} mean = {round(mean[0], 3)}, std_dev = {round(std_dev, 3)}, prior = {round(prior, 3)}"
        ))

    # vertical lines at the intersections
    for point in intersection_points:
        fig.add_trace(go.Scatter(
            x=[point, point],
            y=[0, max(posteriors[0].max(), posteriors[1].max())],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f"Intersection at x = {round(point, 3)}"
        ))

    fig.update_layout(
        title="Unnormalized Posteriors P(X=x|Y=k)P(Y=k)",
        xaxis_title="X",
        yaxis_title="Unnormalized Posterior",
        legend_title="Classes",
        template="plotly_white"
    )

    return fig




def posteriors_plot(discriminant, selected_features, priors, X_train, y_train): 

    if len(selected_features) > 2: 
        gauss_plot =  {'data': [], 'layout': {'title': 'Unnormalized Posterior P(X=x|Y=k)P(Y=k) (1 or 2 features)',
                                            'xaxis': {'title': 'Feature 1'}, 'yaxis': {'title': 'Feature 2'},
                                            'annotations': [{'text': 'Select 1 or 2 features to display the plot.',
                                                            'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]}}


    if discriminant == 'linear':
        model = LinearDiscriminantAnalysis(store_covariance=True)
        model.fit(X_train, y_train)
        means = model.means_
        variance = model.covariance_

        if len(selected_features) == 2: 
            # each class has the same covariance matrix for LDA 
            covariance_matrices = [variance] * 2
            gauss_plot = lda_3d(means, covariance_matrices, priors, selected_features)

        elif len(selected_features) == 1: 
            class_vars = [variance[0][0], variance[0][0]]
            gauss_plot = lda_2d(means, class_vars, priors)
        

    elif discriminant == 'quadratic':
        model = QuadraticDiscriminantAnalysis(store_covariance=True)
        model.fit(X_train, y_train)
        means = model.means_
        variance = model.covariance_

        if len(selected_features) == 2:
            covariance_matrices = variance
            gauss_plot = lda_3d(means, covariance_matrices, priors, selected_features)
        
        elif len(selected_features) == 1: 
            class_vars = [variance[0][0][0], variance[1][0][0]]
            gauss_plot = lda_2d(means, class_vars, priors)


    elif discriminant == 'nb':
        model = GaussianNB()
        model.fit(X_train, y_train)
        means = model.theta_  
        variances = model.var_  

        if len(selected_features) == 2: 
            # construct covariance matrices (diagonal matrices for each class)
            covariance_matrices = []
            for i in range(len(means)):
                cov_matrix = np.diag(variances[i])  
                covariance_matrices.append(cov_matrix)
            gauss_plot = lda_3d(means, covariance_matrices, priors, selected_features)
        
        elif len(selected_features) == 1: 
            # each class has its own variance similar to qda 
            class_vars = [variances[0][0], variances[1][0]]
            gauss_plot = lda_2d(means, class_vars, priors)
        
    return model, gauss_plot 
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree


df = pd.read_csv('idol_classification.csv')
features = ['danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'valence', 'tempo', 'duration_ms', 'gender']
X = df[features]
X = pd.get_dummies(X, drop_first=True, dtype=int)  
y = df['idol_group']
song_names = df['title']
artist_names = df['artist/s']

dummies_features = ['danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'valence', 'tempo', 'duration_ms', 'gender_male', 'gender_mixed']

cat_features = ['key', 'mode', 'gender_male', 'gender_mixed']
scaled_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
unscaled_features = ['loudness', 'tempo', 'duration_ms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
_, _, song_train, song_test = train_test_split(X, song_names, test_size=0.2, random_state=0)
_, _, artist_train, artist_test = train_test_split(X, artist_names, test_size=0.2, random_state=0)


# scaled data
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
# X_train_scaled[unscaled_features] = scaler.fit_transform(X_train_scaled[unscaled_features])
# X_test_scaled[unscaled_features] = scaler.transform(X_test_scaled[unscaled_features])
X_train_scaled[scaled_features + unscaled_features] = scaler.fit_transform(X_train_scaled[scaled_features + unscaled_features])
X_test_scaled[scaled_features + unscaled_features] = scaler.transform(X_test_scaled[scaled_features + unscaled_features])

# only numeric columns
X_train_scaled_numeric = X_train_scaled[scaled_features + unscaled_features]
X_test_scaled_numeric = X_test_scaled[scaled_features + unscaled_features]

# pca
pca = PCA()  
X_train_pca = pca.fit_transform(X_train_scaled_numeric)
X_test_pca = pca.transform(X_test_scaled_numeric)
loading_vectors = pca.components_

X_train_pca = pd.DataFrame(X_train_pca).rename(columns={i: f'PCA{i+1}' for i in range(X_train_pca.shape[1])})
X_test_pca = pd.DataFrame(X_test_pca).rename(columns={i: f'PCA{i+1}' for i in range(X_test_pca.shape[1])})

datasets = { 
    'original': {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'features': dummies_features,
    },

    'scaled': {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'features': dummies_features,
    },

    'scaled_numeric': {
        'X_train': X_train_scaled_numeric,
        'y_train': y_train,
        'X_test': X_test_scaled_numeric,
        'y_test': y_test,
        'features': scaled_features + unscaled_features,
    },

    'pca': {
        'X_train': X_train_pca,
        'y_train': y_train.reset_index(drop=True),
        'X_test': X_test_pca,
        'y_test': y_test.reset_index(drop=True),
        'features': X_train_pca.columns.tolist(),
    }
}


from tab_funcs import pca_funcs
from tab_funcs import logistic_funcs
from tab_funcs import plotly_funcs
from tab_funcs import knn_funcs
from tab_funcs import discriminant_funcs
from tab_funcs import svm_funcs
from tab_funcs import dtree_funcs
from tab_funcs import ensemble_funcs




app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    html.H1("Interactive Model Dashboard", style={'textAlign': 'center'}),

    # model tabs
    dcc.Tabs(id="model-tabs", value='pca', children=[
        # unsupervised 
        dcc.Tab(label='PCA', value='pca'),
        # supervised 
        dcc.Tab(label='Logistic Regression', value='logistic'),
        dcc.Tab(label='K-Nearest Neighbors', value='knn'),
        dcc.Tab(label='Discriminant Analysis', value='discriminant'),
        dcc.Tab(label='Support Vector Machine', value='svm'),
        dcc.Tab(label='Decision Tree', value='dtree'),
        dcc.Tab(label='Tree Ensemble Models', value='tree_ensemble')
    ]),
    html.Div(id='tab-content')
])

# update content based on selected tab 
@app.callback(
    Output('tab-content', 'children'),
    [Input('model-tabs', 'value')],
)
def render_tab_content(tab_name):

    if tab_name == 'pca':
        return html.Div([

            # slider for number of PCA components
            html.Label("Number of Components:"),
            dcc.Slider(
                id='pca-components-slider',
                min=2, max=3, step=1, value=2,
                marks={i: str(i) for i in range(2, 4)}
            ),
            
            dcc.Markdown(r"""PCA results in a new set of features called principal components that are linear combinations of the original features and uncorrelated with each other.
                         The idea is to find the directions that explain the most variance in the data.
                         We can take a subset of the principal components (first few principal components) to reduce the dimensionality of our data and retain most of the variance.
                         For the first principal component $Z_1$, we have the optimization problem to maximize the variance of $Z_1 = \phi_{11}X_1 + \phi_{21}X_2 + ... + \phi_{p1} X_p$ 
                         with respect to $\phi_1 = (\phi_{11}, \phi_{21}, ..., \phi_{p1})$ and with the constraint $\sum_{j=1}^p \phi_{j1}^2 = 1$ because otherwise we can choose the $\phi_i$ to make the variance arbitrarily large. 
                         For subsequent principal components $Z_2, Z_3, ...$, we have the additional constraint that they are uncorrelated with all previous components. 
                         This is equivalent to constraining $\phi_1 = (\phi_{11}, \phi_{21}, ..., \phi_{p1})$ to be orthogonal to $\phi_2 = (\phi_{12}, \phi_{22}, ..., \phi_{p2})$, and so on.
                         """, mathjax=True),

            dcc.Markdown("""PCA only makes sense for numerical features. We are trying to find a new smaller set of features that are a linear combination of the original features with maximal variance.
                    We cannot compute variance and covariance for categorical features.
                    We could use the full dataset for PCA. However, if we intend to use a test set for our future models and we plan to use PCA features, it makes sense to split the data into training and test sets now.
                    We can fit the PCA on the training set and transform both the training set and test set.
                    PCA is unsupervised, so the labels are not important for PCA itself. However, they will be useful for the classification models we fit using PCA features.
                """),

            # PCA training set title 
            html.Div('PCA on Training Set', style={'textAlign': 'center', 'padding': '10px'}),

            # PCA training set plot 
            html.Div(
                dcc.Graph(id='pca-plot-train'),
                style={'width': '100%', 'padding': '10px'}
            ),
        
            # PCA test set title 
            html.Div('PCA on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            
            # PCA test set plot 
            html.Div(
                dcc.Graph(id='pca-plot-test'),
                style={'width': '100%', 'padding': '10px'}
            ),

            dcc.Markdown("""We can see some separation between the classes. Idol group songs seem to have higher PCA1 values than non idol group songs.
                         Non idol group songs seem to have a larger spread in PCA2 than idol group songs. In particular, songs with low PCA2 values are mostly non idol group songs."""),

            dcc.Markdown("""The explained variance ratio tells us the percentage of the total variance in the data explained by each principal component.
                          It must be that the earlier principal components explain a larger proportion of the total variance than the later principal components.
                          If we use all principal components (same dimensionality as original set of features), then the cumulative variance explained is 100%."""),

            # scree plot title 
            html.Div('PCA Scree Plot', style={'textAlign': 'center', 'padding': '10px'}),

            # scree plot 
            html.Div(
                    dcc.Graph(id='pca-scree-plot'),
                    style={'width': '100%', 'padding': '10px'}
                ),

            dcc.Markdown("""Using 2 principal components explains around 55% of the variance in the data. Using 3 principal components explains around 67% of the variance in the data.
                          However, they are still useful to look at since we can visualize 2 or 3 dimensions. Using 5 principal components explains a good amount of the variance in the data, around 86%."""),

            dcc.Markdown("""We can get a notion of feature importance by looking at the loading vectors, which tell us the contribution of each of the original features to the respective principal components.
                          The sign of an element of a loading vector tells us whether the feature contributed positively or negatively to the principal component.
                          The magnitude tells us how much the feature contributes to the principal component.
                          Since the earlier principal components are more important than the later principal components, for each feature, we weight the first n_components loadings by the proportion of variance explained,
                          then compute sums of squares to obtain an overall magnitude, which we can use as a feature importance score."""),

            # feature magnitude plot title 
            html.Div('PCA on Test Set', style={'textAlign': 'center', 'padding': '10px'}),

            # feature magnitude plot 
            html.Div(
                    dcc.Graph(id='pca-feature-magnitude-plot'),
                    style={'width': '100%', 'padding': '10px'}
                ),

            # description of feature magnitude plot 
            html.Div(id='pca-feature-mag-desc',),

])

    if tab_name == 'logistic':
        return html.Div([

            dcc.Markdown(r"""The logistic regression model has the form $$\log(\frac{p}{1-p}) = \beta_0 + \beta_1 X_1 + ... + \beta_k X_k$$. 
                         This is equivalent to $$p(X) = Pr(Y=1|X) = \frac{e^{\beta_0 + \beta_1 X_1 + ... + \beta_k X_k}}{1 + e^{\beta_0 + \beta_1 X_1 + ... + \beta_k X_k}}$$. 
                         The parameters $\beta$ are chosen to maximize the likelihood of the observed data.""", mathjax=True),

            dcc.Markdown(r"""We can build a logistic regression model by choosing some variables to include.
                         We want our model to be simple. Often times, there is a subset of useful features that fits the data about as well as the model with all of the coefficients.
                         """, mathjax=True),

            dcc.Markdown(r"""The output shows p-values for z-tests that test for whether the corresponding coefficients $\beta$ are nonzero.
                          If the p-values are small, there is sufficient evidence that the coefficients are nonzero, so they would be useful to include in our model. 
                         However, in the full model, some terms may appear not significant when they actually are.
                          This is because of multicollinearity, where we have variables that are correlated with each other.
                          Since the variables are correlated, if we know information about one variable, then we know some information about the other variables,
                          so it may not be needed to include all of the correlated variables in the model.
                          We would like to avoid collinearity because it can lead to overfitting and high variance in the estimated coefficients.""", mathjax=True),

            dcc.Markdown(r"""When comparing models, there are also other criteria we can look at. We want to try to have significant coefficients, which
                          we can assess by looking at the p-values for the individual coefficients and the p-value for the likelihood ratio test, which tests for whether any of the coefficients are nonzero.
                          One useful criterion to look at is the AIC, which accounts for the likelihood of the data given the model and the number of estimated parameters.
                          The AIC is given by $AIC = 2k - 2\log(L)$. When comparing models,
                          a models with a lower AIC is preferred since we have a better fit to the data (higher $L$) and a less complex model (lower $k$).""", mathjax=True),

            # dropdown for dataset selection
            dcc.Dropdown(id='dataset-dropdown', options=[{'label': name, 'value': name} for name in datasets.keys()],
                         value='original', placeholder="Select Dataset"),

            # dropdown for feature selection
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in datasets['original']['features']],
                value=datasets['original']['features'],  # default to all features of the selected dataset
                multi=True,
                placeholder="Select Features"
            ),

            # p values plot 
            html.Div('P-values for Coefficients in Logistic Model', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='logistic-p-values'),

            # decision boundary training set plot 
            html.Div('Decision Boundary on Training Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='logistic-train-decision'),

            # decision boundary test set plot 
            html.Div('Decision Boundary on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='logistic-test-decision'),

            dcc.Markdown(r"""We can perform a drop in deviance test for the coefficients in the model.
                          We compare the drop in deviance of the full model with $m$ predictors and a reduced model with $p < m$ predictors with a $\chi^2$ distribution with $m-p$ degrees of freedom.
                          If the reduced model is a constant model with only an intercept, then we compare the drop in deviance with a $\chi^2$ distribution with $m$ degrees of freedom.
                          The drop in deviance tests for whether the coefficients in the full model that are not included in the reduced model are 0.
                         For the reduced model, we will take the two predictors with the lowest p-values for simplicity.""", mathjax=True),

            # drop in deviance test comment 
            html.Div(id='logistic-drop-in-dev-text'),

            dcc.Markdown("""One benefit of logistic regression is that the output of the model is log odds,
                          which we can transform into probabilities. Thus, we can obtain a confidence for our predictions.
                         We are more confident in our predictions for probabilities close to 0 or 1.
                         We can convert the probabilities to predictions by checking if they meet some threshold (0.5).
                          Then, we can compute the training and test accuracies."""),

            # training set confusion matrix 
            html.Div('Training Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='logistic-train-conf-mat'),

            # test set confusion matrix 
            html.Div('Test Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='logistic-test-conf-mat'),

            dcc.Markdown("""We can also interpret the coefficients in logistic regression. 
                         We can compute confidence intervals to make inference about coefficients.""" ),

            # confidence intervals plot 
            html.Div('Confidence Intervals for Coefficients', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='logistic-confidence-plot'),
            html.Div(id='logistic-confidence-text'),

            dcc.Markdown("""However, all of these interpretations are in log-odds, which can be quite hard to think about.
                          It makes more intuitive sense to think about odds. We can do this by exponentiating the lower and upper bounds.
                         Additionally, we talk about unit increases in the features, but some of the variables are on a scale from 0 to 1,
                         so these changes are extreme. Thus, for these variables, we can think about how a smaller change in the features (such as 0.1) affects the odds."""),
            
            # convert confidence interval from log odds to odds 
            html.Div(id='logistic-odds-confidence-text'),
            

        ])

    
    elif tab_name == 'knn':
        return html.Div([

            dcc.Markdown("""The idea behind KNN is that observations of the same class will have features that are similar to each other. 
                         In k-Nearest Neighbors, for each point, the distance between that point and all other training points are computed.
                         The k nearest points in terms of distance are the nearest neighbors of that point.
                         We look at the classes of the nearest neighbors and assign the point to the majority class."""),

            dcc.Markdown("""Since KNN is based on distance and some of the features are on different scales, it is best to scale the data.
                         One downside to KNN is that it can be difficult to implement with categorical features since the classification is based on distance between points.
                         It would be best to choose a distance function that works well for our data, especially if we are using the categorical variables (gender_male, gender_mixed, mode, and key).
                         However, for simplicity, we are using euclidean distance."""),

            # dropdown for dataset selection
            dcc.Dropdown(id='dataset-dropdown', options=[{'label': name, 'value': name} for name in datasets.keys()],
                         value='scaled_numeric', placeholder="Select Dataset"),
            
            # dropdown for feature selection 
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in datasets['scaled_numeric']['features']],
                value=datasets['scaled_numeric']['features'],  # default to all features of the selected dataset
                multi=True,
                placeholder="Select Features"
            ),

            # slider for number of neighbors 
            html.Label("Number of Neighbors (K):"),
            dcc.Slider(id='knn-k-slider', min=1, max=40, step=1, value=5,
                       marks={i: str(i) for i in range(1, 41)}),

            dcc.Markdown("""For smaller K, we capture more local variation since the classification depends on a smaller number of points.
                          For larger K, we expect the decision boundary to be smoother since the classification depends on more points that are further away."""),

            # training set decision boundary
            html.Div('Decision Boundary on Training Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='knn-train-decision'),

            # test set decision boundary 
            html.Div('Decision Boundary on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='knn-test-decision'),

            # training set confusion matrix 
            html.Div('Training Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='knn-train-conf-mat'),

            # test set confusion matrix 
            html.Div('Test Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='knn-test-conf-mat'),

            dcc.Markdown("""Typically, the number of neighbors is chosen by cross validation.
                         We choose the value of k that results in the lowest average error on the cross validation sets."""),

            # cross validation plot 
            html.Div('KNN Cross Validation Plot', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='knn-cross-val-plot'),
            html.Div(id='knn-cross-val-text'),

            # cross validation confusion matrix 
            html.Div('Cross Validation k Test Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='knn-cv-conf-mat'),
            

        ])
    
    elif tab_name == 'discriminant': 
        return html.Div([

            dcc.Markdown("""When we have a set of observations and their classes, if we make some assumptions about
                         the probabilities $Pr(X=x|Y=k)$, then we can use Bayes' theorem to obtain $Pr(Y=k|X=x)$. This
                         is the main idea of discriminant analysis classification.""", mathjax=True),

            dcc.Markdown(r"""For a given class k, we have $Pr(Y=k | X=x) =  \frac{\pi_k f_k(x)}{\sum_l^K \pi_l f_l(x)}$ where $\pi_k$ is the prior probability
                        of an observation belonging to class k and $f_k(x) = Pr(X=x | Y=k)$. In the case of LDA and QDA, we assume
                        that $f_k(x)$ is Gaussian (multivariate if there are multiple features).
                        In the case of LDA, we assume the variance or the variance-covariance matrix is the same across all classes.
                        In the case of QDA, we do not make this assumption. For example, in the case of LDA with multiple features, we
                          have $f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} e^{-\frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x-\mu_k)}$ where $\mu_k$ is the mean vector
                        for class $k$ and $\Sigma$ is the shared variance-covariance matrix across all classes.""", mathjax=True),

            dcc.Markdown("""Since we make the assumption that $X$ in each class comes from a multivariate normal distribution, 
                         our model will likely not perform well with encoded categorical variables.""", mathjax=True),

            dcc.Markdown(r"""In LDA, we classify the observation in the class for which $Pr(Y=k|X=x)$ is maximized. This is equivalent to 
                         classifying based on the class for which the discriminant
                          function $\delta_k(x) = x^T \Sigma_k^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma_k^{-1} \mu_k + \log \pi_k$ is maximized.
                         It can be shown that the discriminant function is a linear function in $x$, so LDA results in a linear decision boundary.
                         """, mathjax=True),                        

            dcc.Markdown("""QDA is similar to LDA with the difference that the classes do not share the same variance-covariance matrix. 
                         This results in a quadratic decision boundary, which allows QDA to perform better than LDA if the data is nonlinear.
                          The drawback is that QDA can overfit if the data is actually linear, especially if you have a small dataset.
                          Another concern is that if we do not have enough data, we may not be able to estimate the multiple covariance matrices."""),

            dcc.Markdown("""Naive Bayes assumes that the predictors are independent of each other. As a result, the variance-covariance matrix is diagonal.
                          Additionally, there is no requirement for the features to come from a multivariate normal distribution.
                          This means unlike LDA and QDA, naive Bayes can allow for categorical predictors. 
                         The simple assumptions of naive Bayes are useful when there are few observations and you are unable to estimate many parameters.
                          It also works well when there are multiple features, where QDA and LDA break down because there are too many parameters to estimate.
                          The disadvantage is that the assumptions are often times not met."""),

            # dataset selection dropdown 
            dcc.Dropdown(id='dataset-dropdown', options=[{'label': name, 'value': name} for name in datasets.keys()],
                         value='scaled_numeric', placeholder="Select Dataset"),
            
            # feature selection dropdown 
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in datasets['scaled_numeric']['features']],
                value=datasets['scaled_numeric']['features'],  # Default to all features of the selected dataset
                multi=True,
                placeholder="Select Features"
            ),

            # discriminant selection dropdown 
            html.Label("Discriminant:"),
            dcc.Dropdown(id='discriminant-dropdown', options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Quadratic', 'value': 'quadratic'},
                {'label': 'Naive Bayes', 'value': 'nb'}
            ], value='linear'),
            
            # unnormalized posteriors plot 
            html.Div('Unnormalized Posteriors', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='lda-posteriors'),

            # training decision boundary
            html.Div('Discriminant Analysis Decision Boundary on Training Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='lda-train-decision'),

            # test decision boundary
            html.Div('Discriminant Analysis Decision Boundary on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='lda-test-decision'),

            # training confusion matrix 
            html.Div('Training Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='lda-train-conf-mat'),

            # test confusion matrix 
            html.Div('Test Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='lda-test-conf-mat'),


        ])
    
    elif tab_name == 'svm':
        return html.Div([

            html.Div('Background: Maximal Margin Classifier', style={'textAlign': 'center', 'padding': '10px'}),

            dcc.Markdown("""Say we have two classes, class 0 and class 1. 
                         We want to find a hyperplane that separates the two classes, so that points on one side of the hyperplane 
                         are classified as class 0 and the other side are classified as class 1. But there are many separating 
                         hyperplanes, so how do we know which one to choose? One idea is to choose the hyperplane that maximizes 
                         the margins, the distances from the hyperplane to the support vectors (points closest to the hyperplane)."""),
            
            dcc.Markdown(r"""We have the constrained optimization problem $\max_{\beta_0,\beta_1,...,\beta_p}M$ subject 
                         to $\sum_{j=1}^p \beta_j^2 = 1$ and $y_i(\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip}) \ge M$ for all $i$.""", mathjax=True),
                         
            dcc.Markdown(r"""So all points have to be on the right side of the hyperplane or 
                         else $y_i$ and $\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip}$ will have
                          different signs and their product will be negative. The constraint on the $\beta_j$'s doesn't affect the hyperplane,
                          since we can obtain the same line by scaling all of the coefficients. 
                         However, it helps with interpretation, since with the constraint, we have 
                         that $y_i(\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip})$ is the perpendicular
                          distance of the $i$th observation to the hyperplane.""", mathjax=True),

            html.Div('Support Vector Classifier', style={'textAlign': 'center', 'padding': '10px'}),

            dcc.Markdown("""However, in many cases, we do not have classes that are 
                        perfectly linearly separable, so a maximal margin classifier will
                        fail. If we relax the constraints by allowing some points to violate
                        the margins or even the hyperplane, then we have a support vector classifier."""),

            dcc.Markdown("""Another benefit of SVCs is that even if the data is linearly separable, it is 
                         more robust to individual points. A maximal margin classifier may result in an 
                         unoptimal separating hyperplane with small margins just to make sure no points 
                         violate the hyperplane."""),

            dcc.Markdown(r"""The optimization problem is $\max_{\beta_0,\beta_1,...,\beta_p, \epsilon_1, ...,\epsilon_n} M$ subject 
                         to $\sum_{j=1}^p \beta_j^2 = 1$ and
                         $y_i (\beta_0 + \beta_1 + \beta_2 x_{i2} + ... + \beta_p x_{ip}) \ge M(1-\epsilon_i)$ with
                         $\epsilon_i \ge 0$ for all $i$ and 
                         $\sum_{i=1}^n \epsilon_i \le C$.""", mathjax=True),

            dcc.Markdown(r"""We call $C$ the cost. If we have $\epsilon_i$ = 0 for all $i$ so that $C=0$, then this is the
                        same as the maximal margin classifier. If we have some $0 < \epsilon_i < 1$, then we are allowing 
                         that point to violate the margin. If we have some $\epsilon_i > 1$, then we are allowing 
                         that point to violate the hyperplane. The cost is an upper limit on the number of points that 
                         we allow to violate the hyperplane.""", mathjax=True),


            dcc.Markdown(r"""In our formulation, a smaller $C$ allows for less violations, leading to narrower margins. 
                         However, the definition of $C$ might be different in different frameworks like R and scikit-learn. 
                         In R, $C$ is a penalty for violations, so a larger $C$ leads to narrower margins. 
                         In scikit-learn, the optimization problem is $$\min \sum_{i=1,n} L(f(x_i), y_i) + \Omega (w)$$ where
                         $L$ is a loss function based on how much the data points violates the margins 
                         and $\Omega$ is a penalty function that encourages the model to find a simpler decision
                         boundary with smaller weights. Since $C$ only affects the loss function, a larger $C$ means 
                         we give more weight to boundary violations, resulting in narrower margins.""", mathjax=True),

            html.Div('Support Vector Machine', style={'textAlign': 'center', 'padding': '10px'}),

            dcc.Markdown("""With the support vector classifier, we are limited to a linear decision boundary. 
                         However, if we enlarge the feature space using kernels, we can obtain nonlinear 
                         decision boundaries. This is the idea behind support vector machines."""),

            dcc.Markdown(r"""The support vector machine has the form $f(x)= \beta_0 + \sum_{i \in S} \alpha_i K(x, x_i)$ 
                         where $S$ is the set of support vector indices, and $K$ is the kernel function. 
                         By using nonlinear kernels, we can model nonlinear decision boundaries. 
                         Common choices are the polynomial kernel and the radial basis kernel.""", mathjax=True),

            dcc.Markdown(r"""In scikit learn, the polynomial kernel of degree $d$ and coef0 $r$ is 
                         $K(x_i, x_{i'}) = (r + \gamma \sum_{j=1}^p x_{ij} x_{i'j})^d$""", mathjax=True),

            dcc.Markdown(r"""The radial kernel with parameter $\gamma$ is 
                         $K(x_i, x_{i'}) = \exp(-\gamma \sum_{j=1}^p (x_{ij} - x_{i'j})^2)$""", mathjax=True),

            dcc.Markdown(r"""$\gamma$ controls the effect of distant observations on the classification of a point.
                          If $\gamma$ is large, then points that are far away from the test point have a smaller impact on the
                          classification of the test point. This means that only nearby points affect the classification,
                          resulting in more complex decision boundaries. On the other hand, if $\gamma$ is small, then
                          we have smoother decision boundaries.""", mathjax=True),

            # dataset selection dropdown
            dcc.Dropdown(id='dataset-dropdown', options=[{'label': name, 'value': name} for name in datasets.keys()],
                         value='scaled_numeric', placeholder="Select Dataset"),
            
            # feature selection dropdown
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in datasets['scaled_numeric']['features']],
                value=datasets['scaled_numeric']['features'],  # Default to all features of the selected dataset
                multi=True,
                placeholder="Select Features"
            ),

            # kernel selection dropdown 
            html.Label("Kernel:"),
            dcc.Dropdown(id='svm-kernel-dropdown', options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Polynomial', 'value': 'poly'},
                {'label': 'RBF', 'value': 'rbf'}
            ], value='linear'),

            # regularization c slider 
            html.Label("C:"),
            dcc.Slider(id='svm-c-slider', min=-3, max=2, step=0.1, value=0,
                       marks={i: str(10**i) for i in range(-3, 3)}),

            # degree slider for poly kernel 
            html.Label("Degree (poly kernel only):"),
            dcc.Slider(
                id='svm-degree-slider',
                min=1, max=5, step=1, value=2,
                marks={i: str(i) for i in range(1, 6)}
            ),

            # gamma slider for poly and rbf kernel 
            html.Label("Gamma (poly and rbf kernel):"),
            dcc.Slider(
                id='svm-gamma-slider',
                min=-3, max=2, step=0.1, value=-1,
                marks={i: str(10**i) for i in range(-3, 3)}
            ),
            
            # decision boundary training set 
            html.Div('Decision Boundary and Margins on Training Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='svm-train-decision'),

            # decision boundary test set 
            html.Div('Decision Boundary and Margins on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='svm-test-decision'),

            # training set confusion matrix 
            html.Div('Training Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='svm-train-conf-mat'),

            # test set confusion matrix 
            html.Div('Test Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='svm-test-conf-mat'),   

            dcc.Markdown("""It's best to choose hyperparameters by cross validation. In the case where there are
                         many hyperparameters to optimize, this can take a while. Select the option below
                         to enable grid search."""),

            # kernel for grid search 
            html.Label("Kernel:"),
            dcc.Dropdown(id='svm-kernel-dropdown-2', options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Polynomial', 'value': 'poly'},
                {'label': 'RBF', 'value': 'rbf'}
            ], value='linear'),

            # grid search selection dropdown 
            html.Label("Perform Grid Search:"),
            dcc.Dropdown(
                id='grid-search-dropdown',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',  
            ),

            # grid search decision boundary on training set 
            html.Div('Grid Search Decision Boundary and Margins on Training Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='svm-train-grid-search'),   

            # grid search decision boundary on test set 
            html.Div('Grid Search Decision Boundary and Margins on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='svm-test-grid-search'),   

        ])
    
    elif tab_name == 'dtree':
        return html.Div([

            dcc.Markdown("""Decision trees work by creating splits that separate the predictor space into 
                         multiple regions. In the case of classification trees, we assign observations in the same 
                         region to the majority class of that region. At each step when building the tree, 
                         we split the predictor space (1st split only) or a previously established region of the predictor 
                         space into two regions based 
                         on a single feature and some specified criteria. In the case of classification trees, 
                         this could be to find the split that minimizes the classification error, or the Gini 
                         index or cross-entropy, which are measures of impurity."""),

            dcc.Markdown("""The benefits of decision trees are that they are simple, easy to interpret, 
                         and make for intuitive visualizations. The downsides are that they cannot capture 
                         boundaries that are not a result of recursive binary splitting well, and other 
                         supervised learning techniques typically outperform them."""),

            dcc.Dropdown(id='dataset-dropdown', options=[{'label': name, 'value': name} for name in datasets.keys()],
                         value='original', placeholder="Select Dataset"),

            # feature selection dropdown 
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in datasets['original']['features']],
                value=datasets['original']['features'], 
                multi=True,
                placeholder="Select Features"
            ),

            # impurity metric dropdown
            dcc.Dropdown(
                id='impurity-dropdown',
                options=[
                    {'label': 'Gini', 'value': 'gini'},
                    {'label': 'Cross-Entropy', 'value': 'entropy'},
                ],
                value='gini', 
                placeholder="Select Impurity Metric"
            ),

            # impurity metric plot 
            html.Div('Impurity Metric', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='impurity-plot'),

            # unpruned decision tree 
            html.Div('Unpruned Decision Tree', style={'textAlign': 'center', 'padding': '10px'}),
            html.Img(id='dtree-plot'), 

            dcc.Markdown("""When building a tree, we can build the tree to be deep and have many nodes. However, 
                         this is not what we want since it means that our model is likely overfitting. We can prevent 
                         this by either choosing a max depth for the tree or growing a large tree and pruning it 
                         afterward. For the latter method, we choose a cost for the number of terminal nodes in the tree 
                         so that there is a penalty for complexity. We prune the nodes or splits if the reduction 
                         in the chosen criteria (Classification error, Gini, Cross-entropy) is not greater 
                         than the cost. For each cost, there is an optimal subtree (for a cost of 0, we keep 
                         the unpruned tree). We can choose the cost through cross-validation."""),

            # pruning method dropdown 
            html.Label("Pruning method:"),
            dcc.Dropdown(
                id='pruning-method-dropdown',
                options=[
                    {'label': 'Max Depth', 'value': 'max_depth'},
                    {'label': 'Cost Complexity Pruning', 'value': 'ccp'},
                ],
                value='ccp', 
            ),

            # grid search dropdown 
            html.Label("Grid search:"),
            dcc.Dropdown(
                id='dtree-cross-val-dropdown',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'},
                ],
                value='no', 
            ),

            # tree depth slider
            html.Label("Max depth (for max depth, non grid search):"),
            dcc.Slider(id='tree-depth-slider', min=1, max=5, step=1, value=3,
                                marks={i: str(i) for i in range(1, 6)}),
                        
            # node cost slider
            html.Label("Node cost (for cost complexity, non grid search):"),
            dcc.Slider(id='node-cost-slider', min=0, max=.15, step=.01, value=0.02,
                                ),

            # pruned decision tree 
            html.Div('Pruned Decision Tree', style={'textAlign': 'center', 'padding': '10px'}),
            html.Img(id='dtree-plot-pruned'), 

            # training decision boundary
            html.Div('Pruned Tree Decision Boundary on Training Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='dtree-decision'), 

            # test decision boundary
            html.Div('Pruned Tree Decision Boundary on Test Set', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='dtree-test-decision'), 

            # training confusion matrix 
            html.Div('Pruned Tree Training Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='dtree-conf-mat'), 

            # test confusion matrix 
            html.Div('Pruned Tree Test Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Graph(id='dtree-test-conf-mat'), 
        ])


    elif tab_name == 'tree_ensemble':
        return html.Div([

            html.Div('Bagging', style={'textAlign': 'center', 'padding': '10px'}),            

            dcc.Markdown("""An improvement we can make to decision trees is bagging. With bagging, 
                         we build many decision trees (weak learners) and aggregate their predictions together. 
                         Typically, the trees are grown to be very large to overfit the data. In the case of 
                         classification, we can take the majority prediction as the final prediction. The benefit 
                         of bagging is that since we are averaging the predictions of multiple trees, we are reducing 
                         the variance of our predictions. Thus, increasing the number of trees does not lead to overfitting, 
                         but rather decreases the variance, reducing the risk of overfitting."""),
                         
            dcc.Markdown(r"""To build the trees, it would be best to have multiple 
                         training sets. However, this is not a common circumstance. Intead, what we do is take 
                         bootstrap samples that are the same size as our original training set. This has the 
                         advantage that for each bootstrap sample, about $\frac{1}{e}$ of the observations do not 
                         make it into the sample, giving us a natural out-of-bag validation sample. Then for each 
                         training observation, we can use the trees for which the observation is in the out-of-bag 
                         sample to predict the class and average the predictions together to obtain a validation accuracy.""", mathjax=True),

            html.Div('Random Forest', style={'textAlign': 'center', 'padding': '10px'}),      

            dcc.Markdown("""One potential problem with bagging is that the decision trees can be very similar 
                         since the most important predictors are likely to be the same for each sample. One solution 
                         is the random forest model, which selects only a subset of predictors to include in each tree. 
                         This uncorrelates the trees, which further reduces the variance compared to bagging."""),

            dcc.Markdown("""Another benefit of random forests is that they output feature importance scores. Each 
                         feature receives a score based on how much it reduced the impurity criteria throughout the 
                         multiple decision trees that the model built."""),

            html.Div('Boosting', style={'textAlign': 'center', 'padding': '10px'}),

            dcc.Markdown("""Another tree-based method is boosting. The algorithm for boosting is very different from 
                         bagging and random forest. Instead of fitting decision trees hard to bootstrap samples, 
                         boosting uses only the original training sample and builds trees iteratively, building off the 
                         residuals or misclassifications of the last tree. As such, boosting learns slowly. Another 
                         difference between boosting and bagging or random forest is that it is common for 
                         boosting to build stumps (trees of max depth 1). Unlike bagging or random forest, it is possible for the model
                         to overfit if we build too many trees."""),

            dcc.Dropdown(id='dataset-dropdown', options=[{'label': name, 'value': name} for name in datasets.keys()],
                         value='original', placeholder="Select Dataset"),

            # feature selection dropdown
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in datasets['original']['features']],
                value=datasets['original']['features'],  # Default to all features of the selected dataset
                multi=True,
                placeholder="Select Features"
            ),

            # impurity metric dropdown 
            html.Label("Impurity Metric:"),
            dcc.Dropdown(
                id='impurity-dropdown',
                options=[
                    {'label': 'Gini', 'value': 'gini'},
                    {'label': 'Cross-Entropy', 'value': 'entropy'},
                ],
                value='gini', 
                placeholder="Select Impurity Metric"
            ),

            # ensemble model dropdown
            html.Label("Ensemble Model:"),
            dcc.Dropdown(
                id='ensemble-dropdown',
                options=[
                    {'label': 'Bagging', 'value': 'bagging'},
                    {'label': 'Random Forest', 'value': 'random_forest'},
                    {'label': 'Boosting', 'value': 'boosting'},
                ],
                value='random_forest', 
            ),

        # number of estimators slider 
        html.Label("Number of Estimators:"),
            dcc.Slider(id='num-estimators-slider', min=20, max=500, step=20, value=100),

        # learning rate slider
        html.Label("Learning Rate (Boosting):"),
        dcc.Slider(id='learning-rate-slider', min=-3, max=2, step=0.1, value=0,
                       marks={i: str(10**i) for i in range(-3, 3)}),
    
        # training decision boundary
        html.Div('Decision Boundary on Training Set', style={'textAlign': 'center', 'padding': '10px'}),   
        dcc.Graph(id='ensemble-decision'), 

        # test decision boundary
        html.Div('Decision Boundary on Test Set', style={'textAlign': 'center', 'padding': '10px'}),   
        dcc.Graph(id='ensemble-test-decision'), 

        # training confusion matrix 
        html.Div('Training Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),   
        dcc.Graph(id='ensemble-conf-mat'),

        # out of bag confusion matrix 
        html.Div('Out of Bag Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),   
        dcc.Graph(id='ensemble-oob-conf-mat'),

        # test confusion matrix 
        html.Div('Test Set Confusion Matrix', style={'textAlign': 'center', 'padding': '10px'}),   
        dcc.Graph(id='ensemble-test-conf-mat'), 

        # random forest feature importance
        html.Div('Feature Importance Plot', style={'textAlign': 'center', 'padding': '10px'}),   
        dcc.Graph(id='ensemble-importance'), 
    
        ])




@app.callback(
    Output('feature-dropdown', 'options'),
    [Input('dataset-dropdown', 'value')]
)
def update_feature_options(selected_dataset):
    # Get the features based on the selected dataset
    features = datasets[selected_dataset]['features']
    return [{'label': feature, 'value': feature} for feature in features]


@app.callback(
    Output('feature-dropdown', 'value'),
    [Input('dataset-dropdown', 'value')]
)
def update_feature_value(selected_dataset):
    # Set the default value to all features of the selected dataset
    return datasets[selected_dataset]['features']









@app.callback(
    [Output('pca-plot-train', 'figure'),
     Output('pca-plot-test', 'figure'),
     Output('pca-scree-plot', 'figure'), 
     Output('pca-feature-magnitude-plot', 'figure'),
     Output('pca-feature-mag-desc', 'children')],
    [Input('pca-components-slider', 'value')]
)
def update_pca_plot(n_components):
    data_dict = datasets['pca']
    X_train = data_dict['X_train'].iloc[:, :n_components]
    y_train = data_dict['y_train']
    X_test = data_dict['X_test'].iloc[:, :n_components]
    y_test = data_dict['y_test']
    features = data_dict['features']

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    if n_components in [2,3]: 
        train_plot = pca_funcs.pca_plot_func(pca, n_components, False, X_train, y_train, 
                                             X_test, y_test, features, artist_train, artist_test,
                                             song_train, song_test)
        test_plot = pca_funcs.pca_plot_func(pca, n_components, True, X_train, y_train, 
                                             X_test, y_test, features, artist_train, artist_test,
                                             song_train, song_test)

        pca_scree_plot = pca_funcs.pca_scree(pca)
        pca_feature_magnitude_plot = pca_funcs.pca_feature_magnitude(pca, n_components, scaled_features + unscaled_features)
        if n_components == 2: 
            feature_desc = """Looking at only the first two principal components, the most important feature 
            in terms of magnitude is energy, which contributes positively to both the first and second principal 
            components, with a greater contribution to the first. Loudness is similar to energy. Acousticness 
            contributes negatively to both the first and second principal components, with a greater negative 
            contribution to the first."""
        elif n_components == 3: 
            feature_desc = """Looking at the first three principal components, the most important feature in terms 
            of magnitude is energy, which contributes positively to the first and second principal components and 
            negatively to the third principal component. Energy has the greatest contribution to the first principal 
            component. Loudness is similar to energy, but with a much larger negative contribution to the third 
            principal component. Acousticness contributes negatively to both the first and second principal 
            components, with a greater negative contribution to the first. Acousticness does not contribute much 
            to the third principal component."""
        return train_plot, test_plot, pca_scree_plot, pca_feature_magnitude_plot, feature_desc
    


@app.callback(
    [Output('logistic-p-values', 'figure'),
     Output('logistic-train-decision', 'figure'),
     Output('logistic-test-decision', 'figure'),
     Output('logistic-drop-in-dev-text', 'children'),
     Output('logistic-train-conf-mat', 'figure'),
     Output('logistic-test-conf-mat', 'figure'),
     Output('logistic-confidence-plot', 'figure'),
     Output('logistic-confidence-text', 'children'),
     Output('logistic-odds-confidence-text', 'children')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-dropdown', 'value')]
)
def update_logistic_plot(selected_dataset, selected_features):

    # add intercept terms for logistic model 
    X_train = sm.add_constant(datasets[selected_dataset]['X_train'][selected_features])
    y_train = datasets[selected_dataset]['y_train']
    X_test = sm.add_constant(datasets[selected_dataset]['X_test'][selected_features])
    y_test = datasets[selected_dataset]['y_test']

    model = sm.Logit(y_train, X_train).fit(disp=False)

    p_values_plot = logistic_funcs.p_values_plot(model)

    train_decision = logistic_funcs.logistic_decision_boundary(model, selected_features, False,
                                                               X_train, y_train, X_test, y_test,
                                                               song_train, song_test, artist_train, artist_test)
    test_decision = logistic_funcs.logistic_decision_boundary(model, selected_features, True,
                                                               X_train, y_train, X_test, y_test,
                                                               song_train, song_test, artist_train, artist_test)
    deviance_text = logistic_funcs.deviance_text_func(model, selected_features, X_train, y_train)

    y_train_probas = model.predict(X_train)
    y_train_pred = (y_train_probas > 0.5).astype(int).to_numpy()
    train_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, y_train_pred)

    y_test_probas = model.predict(X_test)
    y_test_pred = (y_test_probas > 0.5).astype(int).to_numpy()
    test_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_test_pred)

    log_conf = logistic_funcs.logistic_confidence(X_train, model)

    conf_int = model.conf_int(alpha=0.05)
    conf_interp = "" 
    for i, feature in enumerate(conf_int.index.tolist()): 
        lower = conf_int.iloc[i][0]
        upper = conf_int.iloc[i][1]
        conf_interp = conf_interp + logistic_funcs.interpret_interval(lower, upper, feature, scaled_features, unscaled_features)

    odds_conf_interp = "" 
    for i, feature in enumerate(conf_int.index.tolist()): 
        lower = conf_int.iloc[i][0]
        upper = conf_int.iloc[i][1]
        odds_conf_interp = odds_conf_interp + logistic_funcs.interpret_odds_interval(lower, upper, feature, scaled_features, unscaled_features)

    return p_values_plot, train_decision, test_decision, deviance_text, train_conf_mat, test_conf_mat, log_conf, conf_interp, odds_conf_interp



@app.callback(
    [Output('knn-train-decision', 'figure'),
     Output('knn-test-decision', 'figure'),
     Output('knn-train-conf-mat', 'figure'),
     Output('knn-test-conf-mat', 'figure'),
     Output('knn-cross-val-plot', 'figure'),
     Output('knn-cross-val-text', 'children'),
     Output('knn-cv-conf-mat', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('knn-k-slider', 'value')]
)
def update_knn_plot(selected_dataset, selected_features, k):

    X_train = datasets[selected_dataset]['X_train'][selected_features]
    y_train = datasets[selected_dataset]['y_train']
    X_test = datasets[selected_dataset]['X_test'][selected_features]
    y_test = datasets[selected_dataset]['y_test']

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    knn_decision_train = plotly_funcs.plotly_decision_boundary(model, selected_features, False, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)
    knn_decision_test = plotly_funcs.plotly_decision_boundary(model, selected_features, True, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)

    y_train_pred = model.predict(X_train)
    train_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_test_pred)

    cross_val_plot, opt_k = knn_funcs.n_neighbors_cv(X_train, y_train)

    cross_val_text = f"""Based on the cross-validation accuracy, the optimal number of neighbors is {opt_k}.
    We summarize the test set performance of the model with k={opt_k} in the table below."""

    model = KNeighborsClassifier(n_neighbors=opt_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_pred)


    return knn_decision_train, knn_decision_test, train_conf_mat, test_conf_mat, cross_val_plot, cross_val_text, cv_conf_mat






@app.callback(
    [Output('lda-posteriors', 'figure'),
     Output('lda-train-decision', 'figure'),
     Output('lda-test-decision', 'figure'),
     Output('lda-train-conf-mat', 'figure'),
     Output('lda-test-conf-mat', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('discriminant-dropdown', 'value')]
)
def update_discriminant_plot(selected_dataset, selected_features, discriminant):

    X_train = datasets[selected_dataset]['X_train'][selected_features]
    y_train = datasets[selected_dataset]['y_train']
    X_test = datasets[selected_dataset]['X_test'][selected_features]
    y_test = datasets[selected_dataset]['y_test']

    priors = [1 - sum(y_train) / len(y_train), sum(y_train) / len(y_train)]

    model, gauss_plot = discriminant_funcs.posteriors_plot(discriminant, selected_features, priors, X_train, y_train)
    train_decision = plotly_funcs.plotly_decision_boundary(model, selected_features, False, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)
    test_decision = plotly_funcs.plotly_decision_boundary(model, selected_features, True, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)

    y_train_pred = model.predict(X_train)
    train_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, y_train_pred)
    y_test_pred = model.predict(X_test)
    test_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_test_pred)

    return gauss_plot, train_decision, test_decision, train_conf_mat, test_conf_mat



@app.callback(
    [Output('svm-train-decision', 'figure'),
     Output('svm-test-decision', 'figure'),
     Output('svm-train-conf-mat', 'figure'),
     Output('svm-test-conf-mat', 'figure'),
     Output('svm-train-grid-search', 'figure'),
     Output('svm-test-grid-search', 'figure'),],
    [Input('dataset-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('svm-kernel-dropdown', 'value'),
     Input('svm-c-slider', 'value'),
     Input('svm-degree-slider', 'value'),
     Input('svm-gamma-slider', 'value'),
     Input('svm-kernel-dropdown-2', 'value'),
     Input('grid-search-dropdown', 'value')]
)
def update_svm_plot(selected_dataset, selected_features, kernel, c_value, degree, gamma, kernel_grid, grid_search_opt):

    X_train = datasets[selected_dataset]['X_train'][selected_features]
    y_train = datasets[selected_dataset]['y_train']
    X_test = datasets[selected_dataset]['X_test'][selected_features]
    y_test = datasets[selected_dataset]['y_test']

    c_value = 10**c_value 
    gamma = 10**gamma

    if kernel == 'linear': 
        model = SVC(kernel='linear', C=c_value)
    elif kernel == 'poly': 
        model = SVC(kernel='poly', C=c_value, degree=degree, gamma=gamma)
    elif kernel == 'rbf': 
        model = SVC(kernel='rbf', gamma=gamma)

    model.fit(X_train, y_train) 

    train_boundary = svm_funcs.plotly_decision_boundary_with_margins(model, selected_features, False, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)
    test_boundary = svm_funcs.plotly_decision_boundary_with_margins(model, selected_features, True, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)

    y_train_pred = model.predict(X_train) 
    train_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, y_train_pred)

    y_test_pred = model.predict(X_test) 
    test_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_test_pred)

    cv_train_decision, cv_test_decision = svm_funcs.svm_grid_search(grid_search_opt, X_train, y_train,
                                                                    X_test, y_test, kernel_grid, selected_features,
                                                                    song_train, song_test, artist_train, artist_test)

    return train_boundary, test_boundary, train_conf_mat, test_conf_mat, cv_train_decision, cv_test_decision


@app.callback(
    [Output('impurity-plot', 'figure'),
     Output('dtree-plot', 'src'),
     Output('dtree-plot-pruned', 'src'),
     Output('dtree-decision', 'figure'),
     Output('dtree-test-decision', 'figure'),
     Output('dtree-conf-mat', 'figure'),
     Output('dtree-test-conf-mat', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('impurity-dropdown', 'value'), 
     Input('pruning-method-dropdown', 'value'),
     Input('dtree-cross-val-dropdown', 'value'),
     Input('tree-depth-slider', 'value'),
     Input('node-cost-slider', 'value'),]
)
def update_dtree_plot(selected_dataset, selected_features, impurity, pruning_method, dtree_cross_val, max_depth, node_cost):
    X_train = datasets[selected_dataset]['X_train'][selected_features]
    y_train = datasets[selected_dataset]['y_train']
    X_test = datasets[selected_dataset]['X_test'][selected_features]
    y_test = datasets[selected_dataset]['y_test']

    impurity_graph = dtree_funcs.plot_impurity_metric(impurity)

    dtree = DecisionTreeClassifier(random_state=0, criterion=impurity)
    dtree.fit(X_train, y_train)
    unpruned_plot = dtree_funcs.plot_decision_tree(dtree, selected_features)
    pruned_dtree = dtree_funcs.prune_tree(pruning_method, dtree_cross_val, impurity, max_depth, node_cost, X_train, y_train)
    pruned_dtree.fit(X_train, y_train)
    pruned_plot = dtree_funcs.plot_decision_tree(pruned_dtree, selected_features)
        
    train_boundary = plotly_funcs.plotly_decision_boundary(pruned_dtree, selected_features, False, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)
    test_boundary = plotly_funcs.plotly_decision_boundary(pruned_dtree, selected_features, True, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)
    
    y_train_pred = pruned_dtree.predict(X_train)
    train_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, y_train_pred)
    
    y_test_pred = pruned_dtree.predict(X_test)
    test_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_test_pred)

    return impurity_graph, unpruned_plot, pruned_plot, train_boundary, test_boundary, train_conf_mat, test_conf_mat


@app.callback(
    [Output('ensemble-decision', 'figure'),
     Output('ensemble-test-decision', 'figure'),
     Output('ensemble-conf-mat', 'figure'),
     Output('ensemble-oob-conf-mat', 'figure'),
     Output('ensemble-test-conf-mat', 'figure'),
     Output('ensemble-importance', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('impurity-dropdown', 'value'), 
     Input('ensemble-dropdown', 'value'), 
     Input('num-estimators-slider', 'value'),
     Input('learning-rate-slider', 'value'),
     ]
)
def update_ensemble_plot(selected_dataset, selected_features, impurity, ensemble, num_estimators, learning_rate):

    learning_rate = 10 ** learning_rate

    X_train = datasets[selected_dataset]['X_train'][selected_features]
    y_train = datasets[selected_dataset]['y_train']
    X_test = datasets[selected_dataset]['X_test'][selected_features]
    y_test = datasets[selected_dataset]['y_test']

    if ensemble == 'bagging': 
        dtree = DecisionTreeClassifier(random_state=0, criterion=impurity)
        model = BaggingClassifier(estimator=dtree, n_estimators=num_estimators, random_state=0, oob_score=True)

    elif ensemble == 'random_forest': 
        model = RandomForestClassifier(n_estimators=num_estimators, random_state=0, criterion=impurity, oob_score=True)

    elif ensemble == 'boosting': 
        tree_stump = DecisionTreeClassifier(max_depth=1, random_state=0, criterion=impurity)
        model = AdaBoostClassifier(estimator=tree_stump, n_estimators=num_estimators, random_state=0,
                                     algorithm='SAMME', learning_rate=learning_rate)
    model.fit(X_train, y_train)

    train_decision = plotly_funcs.plotly_decision_boundary(model, selected_features, False, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)
    test_decision = plotly_funcs.plotly_decision_boundary(model, selected_features, True, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test)

    y_train_pred = model.predict(X_train)
    train_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    test_conf_mat = plotly_funcs.plotly_confusion_matrix(y_test, y_test_pred)

    # out of bag confusion matrix 
    if ensemble == 'bagging' or ensemble == 'random_forest': 
        oob_probabilities = model.oob_decision_function_
        oob_predictions = np.argmax(oob_probabilities, axis=1)  
        oob_conf_mat = plotly_funcs.plotly_confusion_matrix(y_train, oob_predictions)
    
    else: 
        oob_conf_mat = {'data': [], 'layout': {'title': 'Out of Bag Confusion Matrix (Bagging or Random Forest)',
                                            'xaxis': {'title': 'Feature 1'}, 'yaxis': {'title': 'Feature 2'},
                                            'annotations': [{'text': 'Select Bagging or Random Forest to show OOB Confusion Matrix.',
                                                            'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]}}

    importance_plot = ensemble_funcs.plot_feature_importances(ensemble, model, selected_features)

    return train_decision, test_decision, train_conf_mat, oob_conf_mat, test_conf_mat, importance_plot 






if __name__ == '__main__':
    app.run_server(debug=True, port=8090)

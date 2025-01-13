from sklearn.metrics import accuracy_score
import numpy as np 
import statsmodels.api as sm
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import chi2

def logistic_decision_boundary(model, selected_features, test, X_train, y_train, X_test, y_test, song_train, song_test, artist_train, artist_test): 
    if len(selected_features) != 2:
        decision_fig =  {'data': [], 'layout': {'title': 'Decision Boundary (requires 2 features)',
                                        'xaxis': {'title': 'Feature 1'}, 'yaxis': {'title': 'Feature 2'},
                                        'annotations': [{'text': 'Select exactly 2 features to display the decision boundary.',
                                                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]}}
        return decision_fig

    if test: 
        # limits for the plot 
        x_min, x_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
        y_min, y_max = X_train.iloc[:, 2].min() - 1, X_train.iloc[:, 2].max() + 1

        # create a grid of points to predict on
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = sm.add_constant(np.c_[xx.ravel(), yy.ravel()])
        probs = model.predict(grid).reshape(xx.shape)

        # add intercept term to test set 
        X_test_const = sm.add_constant(X_test)

        # predict on test set to obtain accuracy
        y_pred_prob = model.predict(X_test_const) 
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        decision_fig = go.Figure(data=[
            # decision boundary
            go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=probs, colorscale='RdBu', contours={'showlabels': True}),
            # points 
            go.Scatter(
                x=X_test.iloc[:, 1],
                y=X_test.iloc[:, 2],
                mode='markers', 
                marker={'color': y_test, 'symbol': 'circle', 'line': {'width': 1}},
                hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artist_test), np.array(song_test))],)
        ], layout={'title': f'Decision Boundary (Test Accuracy = {round(accuracy,3)})', 'xaxis': {'title': selected_features[0]}, 'yaxis': {'title': selected_features[1]}})

    
    else: 
        # limits for the plot
        x_min, x_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
        y_min, y_max = X_train.iloc[:, 2].min() - 1, X_train.iloc[:, 2].max() + 1

        # create a grid of points to predict on
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = sm.add_constant(np.c_[xx.ravel(), yy.ravel()])
        probs = model.predict(grid).reshape(xx.shape)

        # add intercept term to training set 
        X_train_const = sm.add_constant(X_train)

        # predict on training set to obtain accuracy
        y_pred_prob = model.predict(X_train_const) 
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_train, y_pred)

        decision_fig = go.Figure(data=[
            # decision boundary
            go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=probs, colorscale='RdBu', contours={'showlabels': True}),
            # points
            go.Scatter(x=X_train.iloc[:, 1], 
                        y=X_train.iloc[:, 2], 
                        mode='markers', 
                        marker={'color': y_train, 'symbol': 'circle', 'line': {'width': 1}},
                        # hovertext=np.array(song_train),
                        hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artist_train), np.array(song_train))],)
        ], layout={'title': f'Decision Boundary (Training Accuracy = {round(accuracy,3)})', 'xaxis': {'title': selected_features[0]}, 'yaxis': {'title': selected_features[1]}})
    
    return decision_fig


def logistic_confidence(X_train, model): 

    # coefficients (point estimates) and confidence intervals
    coefficients = model.params
    conf = model.conf_int()

    fig = go.Figure()

    # loop through coefficients to add lines for each predictor
    for i, (coef, ci_lower, ci_upper) in enumerate(zip(coefficients, conf[0], conf[1])):
        # horizontal lines for confidence interval
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[i, i],
            mode='lines',
            line=dict(color='black'),
            name=f"{X_train.columns[i]} Coeff"
        ))
        # point estimates
        fig.add_trace(go.Scatter(
            x=[coef],
            y=[i],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f"{X_train.columns[i]} Coeff",
            showlegend=False
        ))

    fig.update_layout(
        title="Logistic Regression Coefficients with 95% Confidence Intervals",
        xaxis=dict(title="Coefficient", showgrid=True),
        yaxis=dict(title="Predictor", tickvals=np.arange(len(X_train.columns)), ticktext=X_train.columns),
        showlegend=False
    )

    return fig 


def p_values_plot(model): 
    # get coefficients and p values from model
    coef = model.params
    p_values = model.pvalues
    std_err = model.bse

    summary_df = pd.DataFrame({
        'coef': coef,
        'p_value': p_values,
        'std_err': std_err
    })
    summary_df = summary_df.sort_values(by='p_value', ascending=True)

    summary_fig = go.Figure(data=[
        go.Bar(
            # features 
            x=summary_df.index, 
            # coefficients
            y=summary_df['coef'],  
            marker=dict(color=summary_df['p_value'], colorscale='RdBu', cmin=0, cmax=1, colorbar=dict(title='P-value')),
            hovertext=[
                f"Feature: {index}<br>Coefficient: {coef:.4f}<br>P-value: {p_val:.4f}<br>Std Err: {se:.4f}"
                for index, coef, p_val, se in zip(summary_df.index, summary_df['coef'], summary_df['p_value'], summary_df['std_err'])
            ],
            hoverinfo='text'
        )
    ])

    summary_fig.update_layout(
        title="Logistic Regression Coefficients Colored by P-value",
        xaxis_title="Features",
        yaxis_title="Coefficient Value",
        showlegend=False
    )

    return summary_fig




def deviance_text_func(model, selected_features, X_train, y_train):

    # we won't make a reduced model if there are only 2 features selected 
    if len(selected_features) <= 2: 
        # compare full model to null model 
        full_deviance = -2 * model.llf  
        null_deviance = -2 * model.llnull  
        deviance_difference = null_deviance - full_deviance
        df_test = len(selected_features) 

        # compare with chi^2 with df_test degrees of freedom
        p_value = chi2.sf(null_deviance - full_deviance, df_test)

        # at least one coefficient in full model is significant 
        if p_value < 0.05: 
            deviance_text = f"""The deviance for the full model with the selected features is {round(full_deviance,3)}.
                        The deviance for the null model with only the intercept is {round(null_deviance,3)}.
                        The difference between the null deviance and the full model deviance is {round(deviance_difference,3)}.
                        Since we have {df_test} parameter(s), we compare the deviance difference to a chi-square distribution with {df_test} degrees of freedom.
                        This gives us a p-value of {round(p_value,5)}. Since the p-value is small (less than 0.05), there is sufficient
                        evidence to conclude that at least one of the coefficients is nonzero. 
                        Select more than two features to compare the selected model to a reduced model."""
        # insufficient evidence that at least one coefficient in full model is significant
        else: 
            deviance_text = f"""The deviance for the full model with the selected features is {round(full_deviance,3)}.
                        The deviance for the null model with only the intercept is {round(null_deviance,3)}.
                        The difference between the null deviance and the full model deviance is {round(deviance_difference,3)}.
                        Since we have {df_test} parameter(s), we compare the deviance difference to a chi-square distribution with {df_test} degrees of freedom.
                        This gives us a p-value of {round(p_value,5)}. Since the p-value is large (greater than 0.05), there is not sufficient
                        evidence to conclude that at least one of the coefficients is nonzero.
                        Select more than two features to compare the selected model to a reduced model."""         

    # we will make a reduced model with 2 features if there are at least 3 features selected 
    else: 
        # compare full model to null model 
        full_deviance = -2 * model.llf  
        null_deviance = -2 * model.llnull  
        deviance_difference = null_deviance - full_deviance
        df_test = len(selected_features) 

        # compare with chi^2 with df_test degrees of freedom
        p_value = chi2.sf(null_deviance - full_deviance, df_test)

        # we can use p values as a guide for which features might be important
        p_values = model.pvalues

        # subset of features - get features with lowest pvalues
        sorted_p_values = p_values.sort_values()
        sorted_feature_names = sorted_p_values.index.tolist()
        sorted_feature_names.remove('const')
        red_features = sorted_feature_names[:2]

        # train a model with the reduced set of features
        X_train_red = X_train[red_features]  
        X_train_red = sm.add_constant(X_train_red)  
        red_model = sm.Logit(y_train, X_train_red).fit(disp=False)

        # drop in deviance test comparing full model to reduced model 
        red_deviance = -2 * red_model.llf
        drop_in_dev = red_deviance - full_deviance
        df_test2 = len(selected_features) - len(red_features)

        # compare with chi^2 with df degrees of freedom
        p_value2 = chi2.sf(red_deviance - full_deviance, df_test2)

        # at least one coefficient in full model is significant 
        if p_value < 0.05: 
            # at least one coefficient in full model not in reduced model is significant 
            if p_value2 < 0.05: 
                deviance_text = f"""The deviance for the full model with the selected features is {round(full_deviance,3)}.
                                    The deviance for the null model with only the intercept is {round(null_deviance,3)}.
                                    The difference between the null deviance and the full model deviance is {round(deviance_difference,3)}.
                                    Since we have {df_test} parameter(s), we compare the deviance difference to a chi-square distribution with {df_test} degrees of freedom.
                                    This gives us a p-value of {round(p_value,5)}. Since the p-value is small (less than 0.05), there is sufficient
                                    evidence to conclude that at least one of the coefficients is nonzero. 
                                    Looking at a reduced model with only the two predictors in the selected model with the lowest p-values,
                                    the reduced model deviance is {round(red_deviance,3)}. The drop in deviance by adding the additional predictors
                                    in the full model is {round(drop_in_dev,3)}. Since there are {df_test2} new parameter(s) in the full model that are not
                                    in the reduced model, we compare the deviance difference to a chi-square distribution with {df_test2} degrees of freedom.
                                    This gives us a p-value of {round(p_value2, 5)}. Since the p-value is small (less than 0.05), there is sufficient
                                    evidence to conclude that at least one of the coefficients in the full model that is not present in the reduced model is nonzero."""
            # insufficient evidence that at least one coefficient in full model not in reduced model is significant 
            else: 
                deviance_text = f"""The deviance for the full model with the selected features is {round(full_deviance,3)}.
                                    The deviance for the null model with only the intercept is {round(null_deviance,3)}.
                                    The difference between the null deviance and the full model deviance is {round(deviance_difference,3)}.
                                    Since we have {df_test} parameter(s), we compare the deviance difference to a chi-square distribution with {df_test} degrees of freedom.
                                    This gives us a p-value of {round(p_value,5)}. Since the p-value is small (less than 0.05), there is sufficient
                                    evidence to conclude that at least one of the coefficients is nonzero. 
                                    Looking at a reduced model with only the two predictors in the selected model with the lowest p-values,
                                    the reduced model deviance is {round(red_deviance,3)}. The drop in deviance by adding the additional predictors
                                    in the full model is {round(drop_in_dev,3)}. Since there are {df_test2} new parameter(s) in the full model that are not
                                    in the reduced model, we compare the deviance difference to a chi-square distribution with {df_test2} degrees of freedom.
                                    This gives us a p-value of {round(p_value2, 5)}. Since the p-value is large (greater than 0.05), there is not sufficient
                                    evidence to conclude that at least one of the coefficients in the full model that is not present in the reduced model is nonzero."""
        
        # insufficient evidence that at least one coefficient in full model is significant
        elif p_value > 0.05: 
            deviance_text = f"""The deviance for the full model with the selected features is {round(full_deviance,3)}.
                        The deviance for the null model with only the intercept is {round(null_deviance,3)}.
                        The difference between the null deviance and the full model deviance is {round(deviance_difference,3)}.
                        Since we have {df_test} parameter(s), we compare the deviance difference to a chi-square distribution with {df_test} degrees of freedom.
                        This gives us a p-value of {round(p_value,5)}. Since the p-value is large (greater than 0.05), there is not sufficient
                        evidence to conclude that at least one of the coefficients is nonzero.
                        Since there is insufficient evidence that at least one of the coefficients is nonzero,
                        there is no need to check whether a subset of the coefficients is nonzero."""   

    return deviance_text


def interpret_interval(lower, upper, feature, scaled_features, unscaled_features): 
    if feature in scaled_features or feature in unscaled_features: 
        if lower < 0 and upper < 0: 
            keyword = 'decrease'
        elif lower > 0 and upper > 0:
            keyword = 'increase'
        else: 
            keyword = 'change'
        return f"""Controlling for the other predictors, we are 95% confident that a unit increase
            in {feature} will {keyword} the log-odds of being in the idol group category by between {round(lower,3)} and {round(upper,3)}.\n"""

    else:
        if lower < 0 and upper < 0: 
            keyword = 'lower'
        elif lower > 0 and upper > 0:
            keyword = 'higher'
        else: 
            keyword = 'different'

        if feature == 'gender_male': 
            return f"""Controlling for the other predictors, we are 95% confident that the log odds
            of being in the idol group category is {keyword} for male artists than non-male artists by
                between {round(lower,3)} to {round(upper,3)}.\n"""
        elif feature == 'gender_mixed':
            return f"""Controlling for the other predictors, we are 95% confident that the log odds
            of being in the idol group category is {keyword} for mixed group artists than non-mixed group artists by
                between {round(lower,3)} to {round(upper,3)}.\n"""
        elif feature == 'mode':
            return f"""Controlling for the other predictors, we are 95% confident that the log odds
            of being in the idol group category is {keyword} for songs in major key than songs in minor key by
                between {round(lower,3)} to {round(upper,3)}.\n"""            
        else:
            return "" 


def interpret_odds_interval(lower, upper, feature, scaled_features, unscaled_features): 
    if feature in scaled_features : 
        if lower < 0 and upper < 0: 
            keyword = 'decrease'
        elif lower > 0 and upper > 0:
            keyword = 'increase'
        else: 
            keyword = 'change'
        return f"""Controlling for the other predictors, we are 95% confident that an increase 
            in {feature} by 0.1 will {keyword} the odds of being in the idol group category by a factor between {round(np.exp(lower * 0.1),3)} and {round(np.exp(upper * 0.1),3)}.\n"""

    elif feature in unscaled_features: 
        if lower < 0 and upper < 0: 
            keyword = 'decrease'
        elif lower > 0 and upper > 0:
            keyword = 'increase'
        else: 
            keyword = 'change'
        return f"""Controlling for the other predictors, we are 95% confident that an increase 
            in {feature} by 1 will {keyword} the odds of being in the idol group category by a factor between {round(np.exp(lower),3)} and {round(np.exp(upper),3)}.\n"""


    else:
        if lower < 0 and upper < 0: 
            keyword = 'lower'
        elif lower > 0 and upper > 0:
            keyword = 'higher'
        else: 
            keyword = 'different'

        if feature == 'gender_male': 
            return f"""Controlling for the other predictors, we are 95% confident that the odds
            of being in the idol group category is {keyword} for male artists than non-male artists by a factor
                between {round(np.exp(lower),3)} to {round(np.exp(upper),3)}.\n"""
        elif feature == 'gender_mixed':
            return f"""Controlling for the other predictors, we are 95% confident that the odds
            of being in the idol group category is {keyword} for mixed group artists than non-mixed group artists by a factor
                between {round(np.exp(lower),3)} to {round(np.exp(upper),3)}.\n"""
        elif feature == 'mode':
            return f"""Controlling for the other predictors, we are 95% confident that the odds
            of being in the idol group category is {keyword} for songs in major key than songs in minor key by a factor
                between {round(np.exp(lower),3)} to {round(np.exp(upper),3)}.\n"""            
        else:
            return "" 

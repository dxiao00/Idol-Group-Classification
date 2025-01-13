
import numpy as np 
import plotly.graph_objects as go

def pca_plot_func(pca, n_components, test, X_train, y_train, X_test, y_test, features, artist_train, artist_test, song_train, song_test): 
    
    loading_vectors = pca.components_

    # 2d plot for 2 components 
    if n_components == 2: 
        # if test, plot test points
        if test: 
            scatter_traces = []
            for label in np.unique(y_test):
                mask = y_test == label  
                scatter_traces.append(go.Scatter(
                    x=X_test[mask, 0],
                    y=X_test[mask, 1],
                    mode='markers',
                    marker=dict(size=5, opacity=1),
                    hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artist_test)[mask], np.array(song_test)[mask])],
                    hoverinfo="text",
                    name=f"{'idol group' if label==1 else 'non idol group'}",  
                ))
        # else, plot training points
        else: 
            scatter_traces = []
            for label in np.unique(y_train):
                mask = y_train == label 
                scatter_traces.append(go.Scatter(
                    x=X_train[mask, 0],
                    y=X_train[mask, 1],
                    mode='markers',
                    marker=dict(size=5, opacity=1),
                    hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artist_train)[mask], np.array(song_train)[mask])],
                    hoverinfo="text",
                    name=f"{'idol group' if label==1 else 'non idol group'}", 
                ))

        # vectors for the loading components (by feature)
        vectors = []
        for i, feature in enumerate(features):
            # vectors start at the origin and end at (x_end, y_end)
            x_end, y_end = loading_vectors[0, i], loading_vectors[1, i]
            vectors.append(
                go.Scatter(
                    x=[0, x_end],
                    y=[0, y_end],
                    mode='lines+text',
                    line=dict(color='black', width=2),
                    text=[None, feature],  
                    textposition="top center",
                    name=f"{feature} Contribution",  
                    showlegend=False  
                )
            )

        # combine scatter traces and vectors
        fig = go.Figure(data=scatter_traces + vectors)
        fig.update_layout(
            title="PCA Biplot",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            showlegend=True,  
            width=800,
            height=600
        )

        return fig 


    # 3d plot for 3 components
    elif n_components == 3: 
        reduced_loading_vectors_3 = loading_vectors[:3, :]

        scatter_traces_3d = []
        if test: 
            # iterate unique labels to color the classes differently
            for label in np.unique(y_test):
                mask = y_test == label  
                scatter_traces_3d.append(go.Scatter3d(
                    x=X_test[mask, 0],
                    y=X_test[mask, 1],
                    z=X_test[mask, 2],
                    mode='markers',
                    marker=dict(size=3, opacity=0.5),
                    hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artist_test)[mask], np.array(song_test)[mask])],
                    hoverinfo="text",
                    name=f"{'idol group' if label==1 else 'non idol group'}",  
                ))
        else: 
            for label in np.unique(y_train):
                mask = y_train == label  
                scatter_traces_3d.append(go.Scatter3d(
                    x=X_train[mask, 0],
                    y=X_train[mask, 1],
                    z=X_train[mask, 2],
                    mode='markers',
                    marker=dict(size=3, opacity=0.5),
                    hovertext=[f"Artist: {artist}<br>Song: {song}" for artist, song in zip(np.array(artist_train)[mask], np.array(song_train)[mask])],
                    hoverinfo="text",
                    name=f"{'idol group' if label==1 else 'non idol group'}",  
                ))

        # vectors for the loading components (by feature)
        vectors = []
        for i in range(len(reduced_loading_vectors_3[0])):
            x_start, y_start, z_start = 0, 0, 0
            x_end, y_end, z_end = (
                reduced_loading_vectors_3[0][i],
                reduced_loading_vectors_3[1][i],
                reduced_loading_vectors_3[2][i],
            )
            vectors.append(
                go.Scatter3d(
                    x=[x_start, x_end],
                    y=[y_start, y_end],
                    z=[z_start, z_end],
                    mode='lines+text',
                    line=dict(color='black', width=2),
                    text=[None, features[i]],
                    textposition="top center",
                    showlegend=False,  
                    name=f"Vector {i}"
                )
            )

        # combine the scatter plot and vectors
        fig = go.Figure(data=scatter_traces_3d + vectors)
        fig.update_layout(
            scene=dict(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                zaxis_title="Principal Component 3"
            ),
            title="3D PCA",
            showlegend=True,  
        )

        return fig 




def pca_scree(pca):
    # cumulative variance explained (up to 1 for all features)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    comp_idx = np.arange(1, len(explained_variance_ratio) + 1)

    # bar plot for individual explained variance
    trace1 = go.Bar(
        x=comp_idx,
        y=explained_variance_ratio,
        name='Individual explained variance',
        opacity=0.5
    )

    # line plot for cumulative explained variance
    trace2 = go.Scatter(
        x=comp_idx,
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative explained variance'
    )

    layout = go.Layout(
        title='Scree Plot',
        xaxis=dict(
            title='Principal Component Index',
            tickmode='array',
            tickvals=comp_idx
        ),
        yaxis=dict(title='Explained Variance Ratio'),
        showlegend=True
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    return fig 



def pca_feature_magnitude(pca, n_components, features): 

    loading_vectors = pca.components_
    reduced_loading_vectors = loading_vectors[:n_components, :]
    explained_variance_ratio = pca.explained_variance_ratio_
    

    # weighted (by pca component explained variance) magnitudes for each feature
    if n_components == 2: 
        # this is a vector of magnitudes (each element is the magnitude for a feature)
        magnitudes = np.sqrt((reduced_loading_vectors[0] * explained_variance_ratio[0])**2 + (reduced_loading_vectors[1] * explained_variance_ratio[1])**2)

        # sort features by magnitude
        sorted_indices = np.argsort(magnitudes)[::-1]  
        sorted_magnitudes = magnitudes[sorted_indices]
        sorted_features = np.array(features)[sorted_indices]
        pca1_contributions = reduced_loading_vectors[0][sorted_indices]  
        pca2_contributions = reduced_loading_vectors[1][sorted_indices]  

    
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_features,
            y=sorted_magnitudes,
            marker=dict(color='royalblue'),
            hovertext=[f"PCA1: {pca1:.3f}<br>PCA2: {pca2:.3f}" for pca1, pca2 in zip(pca1_contributions, pca2_contributions)],
            hoverinfo='text',  
        ))

    # weighted (by pca component explained variance) magnitudes for each feature
    elif n_components == 3: 
         # this is a vector of magnitudes (each element is the magnitude for a feature)
        magnitudes = np.sqrt((reduced_loading_vectors[0] * explained_variance_ratio[0])**2 + (reduced_loading_vectors[1] * explained_variance_ratio[1])**2 + 
        (reduced_loading_vectors[2] * explained_variance_ratio[2])**2)


        # sort features by magnitude
        sorted_indices = np.argsort(magnitudes)[::-1]  
        sorted_magnitudes = magnitudes[sorted_indices]
        sorted_features = np.array(features)[sorted_indices]
        pca1_contributions = reduced_loading_vectors[0][sorted_indices]  
        pca2_contributions = reduced_loading_vectors[1][sorted_indices]  
        pca3_contributions = reduced_loading_vectors[2][sorted_indices] 


        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_features,
            y=sorted_magnitudes,
            marker=dict(color='royalblue'),
            hovertext=[f"PCA1: {pca1:.3f}<br>PCA2: {pca2:.3f}<br>PCA3: {pca3:.3f}" for pca1, pca2, pca3 in zip(pca1_contributions, pca2_contributions, pca3_contributions)],
            hoverinfo='text',  
        ))  


    fig.update_layout(
        title='(Variance Weighted) Magnitude of Features in Reduced Space',
        xaxis_title='Features',
        yaxis_title='Magnitude',
        showlegend=False
    )

    return fig 

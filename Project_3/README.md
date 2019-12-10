Link to complete project 3 Github repository: 
    https://github.com/alanbalugu/DataScience/edit/master/Project_3
    
    *if the .csv datasets are not populated in the .zip file, please use the above link to download the full datasets and add to this directory.
    
Additional Analysis:
    
    *see network analysis section and Network Analysis pdf file
    
Plotly visualizations:

    files required:
        merged_data.csv
        plotly_visualizations.py
        emission_distribution_plotly.py

    to run:
        open terminal and type "python3 plotly_visualizations.py"
        code will write several visualizations to html (except for stacked bar
        graph of chemical release types) and auto open in web browser

        open terminal and type "python3 emission_distribution_plotly.py"
        code will create stacked bar graph of chemical release types
        and auto open in web browser

    visualizations created:
        linear regression model for age-adjusted cancer rate (averaged per state
        over time) vs. average release estimate total (averaged per state over
        time) – one visualization with Alaska, one with Alaska removed
            with Alaska – index.html for website contained in linreg_alaska
            without Alaska – index.html for website contained in linreg_noalaska

        usa choropleth map showing average release estimate total (averaged over
        time) for each state – one visualization with Alaska, one with Alaska removed
            darker blue = higher release amounts
            lighter blue = lower release amounts
            with Alaska – index.html for website stored in usamap_chem_alaska
            without Alaska – index.html for website stored in usamap_chem_noalaska

        usa choropleth map showing age-adjusted cancer rate (averaged over time)
        for each state
            darker blue = higher cancer rate
            lighter blue = lower cancer rate
            index.html for website stored in usamap_cancer

        line graph with slider showing average release estimate totals over time,
        one line graph trace per state with state as slider – one visualization
        with Alaska, one with Alaska removed
            with Alaska – index.html for website stored in scatter_chem_alaska
            without Alaska – index.html for website stored in scatter_chem_noalaska

        line graph with slider showing age-adjusted cancer rates over time, one
        line graph trace per state with state as slider
            index.html for website stored in scatter_cancer

        bar graph with slider showing average release estimate totals per state
        over time, one bar graph trace per year with year as slider – one visualization
        with Alaska, one with Alaska removed
            with Alaska – index.html for website stored in bar_chem_alaska
            without Alaska – index.html for website stored in bar_chem_noalaska

        bar graph with slider showing age-adjusted cancer rates per state over time,
        one bar graph trace per year with year as slider
            index.html for website stored in bar_cancer

        usa choropleth map showing age-adjusted cancer rate (averaged over time)
        for each state
            darker blue = higher cancer rate
            lighter blue = lower cancer rate
            index.html for website stored in usamap_cancer

        stacked bar graph of chemical release types over time
            index.html for website stored in usamap_merged_clustering
        
Getting the Size of our Original Datasets (number of rows and columns) files required**:
    
        CDC_API.csv
        USCS_CancerTrends_OverTime_ByState.csv
        epa_data_state_chems_and_releases.csv
        combined_twitter_files.csv
        merged_data.csv
        get_size_of_datasets.py
        
        **If any of the above csv files are in a different directory, please
        make sure they are all in the same location as the get_size_of_datasets.py
        file before running the py file.**
        
    to run:
        1) Open Link_to_Large_EPA_Files and download epa_data_state_chems_and_releases.csv
        from the Google drive
        
            *we had to link to this file instead of including it directly because it was
            too large
        
        2) Open terminal and type "python3 get_size_of_datasets.py"
        
    outputs:
        prints number of rows and columns of each of the dataframes we worked with
        to the console (5 total dataframes, for each of the csv files)
        
        we used this file to get additional info for the "Data" section on our
        website
      

Making HeapMap of T-tests Between Regions for Average Cancer Rate, Line Graph of Cancer Rate Over Time, & Cancer Rate Correlations Over Time:
    
    requires:
        CdcClustering.py
        AssociationRuleMining.py
        CDC_regions_heatmaps.py
        USCS_CancerTrends_OverTime_ByState.csv
        
    run:
        CDC_regions_heatmaps.py
    
    outputs:
        1. heatmap of p-values from t-tests between regions for cancer rate time series
        2. line graph of the average cancer rate for each region
        3. heatmap of correlation coefficients between the average cancer rate of each regions
        

Run Machine Learning Classification Models:

    requires:
        merged_data2.csv
        Gaussian_RF_CDC.py
        kNearestNeighbors.py
            CDC_USCS_clustering.py
            CdcClustering.py
        decision_tree.py
        
    run:
        Gaussian_RF_CDC.py
        
        outputs:
            confusion matrices and results of Random Forrest and Gaussian Naive Bayes classifiers respectively.
            
    run:
        kNearestNeighbors.py
        
        outputs:
            confusion matrices and results of kNN classifier.
            ROC curve plot (saved as "ROC CURVE KNN.png")
            
    run:
        decision_tree.py
        
        outputs:
            confusion matrices and results of Decision Tree classifier.
            

Make Heatmap of Cancer Rate Correlations over Time Between States:

    requires:
        merged_data2.csv
        merged_heatmaps.py
        CdcClustering.py
        AssociationRuleMining.py
        
    run:
        merged_heatmaps.py
        
    outputs:
        Heatmap of linear regression correlations between each pair of states for the cancer rate over time.
        

Clustering Analysis on Merged Dataset:

    requires:
        merged_data2.csv
        merged_clustering.py
        CdcClustering.py
        AssociationRuleMining.py
        
    run:
        merged_clustering.py
        
    outputs:
        Scatterplot for KMeans clustering with cluster labels for states as colors. (saved as "KMeans Clustering.png")
        Scatterplot for Hierarchical clustering with cluster labels for states as colors. (saved as "Hierarchical Clustering.png")
        Choropleth for Hiearchical clustering with cluster labels for state as colors. (saved as "usamap_Hierarchical_Clusters_(n_=_6),_Cluster_Avg_Format_=_(pollution,_cancer).html")
        Dendrogram for hierarchical clustering. (saved as "hierarchical clustering dendrogram.png")
        
        
Network Analysis on Merged Dataset:
    
        First make the network:
            
            requires:
                merged_data2.csv
                CdcClustering.py
                AssociationRuleMining.py
                merged_heatmaps.py
                merged_clustering.py
                make_merged_network.py
                
            run:
                make_merged_network.py
                
            outputs:
                network_df.csv
                final_network.csv  (this is the edge and edge weights for the final network)
                
        Complete network analysis and visualizations:
        
            requires:
                final_network.csv
                merged_network_analysis.py
                
            run:
                merged_network_analysis.py
                
            outputs:
                1. Whole network plot
                2. Distribution of degree for the nodes
                3. Degree plot of each state and its degree
                4. Whole network plot colored by degree centrality
                5. Whole network plot colored by betweenness centrality
                6. While network plot colored by partitioning
                7. Degree plot of each state and its degree colored by the partitioning
                
                Various statistics and characteristics about the network
                
Website_Pages_HTML_Files.zip

    contains a zip file with HTML files for each page on our website
                
        
        
        
    

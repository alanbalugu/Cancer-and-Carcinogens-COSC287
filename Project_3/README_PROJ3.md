Plotly visualizations:

    files required:
        merged_data.csv
        plotly_visualizations.py

    to run:
        open terminal and type "python3 plotly_visualizations.py"
        code will write several visualizations to html and auto open in web browser

    visualizations created:
        linear regression model for age-adjusted cancer rate (averaged per state
        over time) vs. average release estimate total (averaged per state over
        time) – one visualization with Alaska, one with Alaska removed

        usa choropleth map showing average release estimate total (averaged over
        time) for each state – one visualization with Alaska, one with Alaska removed
            darker blue = higher release amounts
            lighter blue = lower release amounts

        usa choropleth map showing age-adjusted cancer rate (averaged over time)
        for each state
            darker blue = higher cancer rate
            lighter blue = lower cancer rate

        line graph with slider showing average release estimate totals over time,
        one line graph trace per state with state as slider – one visualization
        with Alaska, one with Alaska removed

        line graph with slider showing age-adjusted cancer rates over time, one
        line graph trace per state with state as slider

        bar graph with slider showing average release estimate totals per state
        over time, one bar graph trace per year with year as slider – one visualization
        with Alaska, one with Alaska removed

        bar graph with slider showing age-adjusted cancer rates per state over time,
        one bar graph trace per year with year as slider
        
Getting the Size of our Original Datasets (number of rows and columns) files required**:
    
        CDC_API.csv
        USCS_CancerTrends_OverTime_ByState.csv
        epa_data_state_chems_and_releases.csv
        combined_twitter_files.csv
        merged_data.csv
        get_size_of_datasets.py
        
        **If any of the above csv files are in a different directory, please
        make sure they are all in the same location as the get_size_of_datasets.py
        file before running the py file.
        
    to run:
        open terminal and type "python3 get_size_of_datasets.py"
        
    outputs:
        prints number of rows and columns of each of the dataframes we worked with
        to the console (5 total dataframes, for each of the csv files)
        
        we used this file to get additional info for the "Data" section on our
        website
      

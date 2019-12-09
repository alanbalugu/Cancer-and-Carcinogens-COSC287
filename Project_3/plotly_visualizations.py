import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly
import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def makeSubset(dataFrame, varList):
    #make a new dataframe for the subset
    dataSubset = pd.DataFrame()
    #loop through the variables given, make subset frame col equal to dataFrame col
    for var in varList:
        dataSubset[var] = dataFrame[var]
    return dataSubset

def avgOverTime(dataFrame, stateAbbr, var):
    #find dataframe rows that correspond to the state abbr given
    rows = dataFrame.loc[dataFrame['STATE_ABBR'] == stateAbbr]
    #make a subset of the dataframe, containing one column related to that state's var data
    dataSubset = makeSubset(rows, [var])

    #loop through the subset, sum the data, keep count
    count = 0
    total = 0
    for i in dataSubset[var]:
        total += i
        count += 1

    #when the count isn't zero, return total/count = average
    if count != 0:
        return (total/count)
    #if count is zero, return -1 since no division by zero
    else:
        return -1

def categorize(value):
    #state abbr to region
    if value in ['CT', 'DE', 'FL', 'GA', 'IN', 'KY', 'ME', 'MI', 'MD', 'MA', 'PA', 'OH', 'WV','VA','NC', 'SC', 'NY', 'VT', 'NH', 'RI', 'DC','NJ']:
        return 'EAST'
    elif value in ['ND', 'MN', 'WI', 'SD', 'NE', 'IA', 'IL', 'KS', 'MO', 'TN', 'OK', 'AR', 'MS', 'AL', 'TX', 'LA']:
        return 'CENT'
    elif value in ['MT', 'ID', 'WY', 'UT', 'CO', 'AZ', 'NM']:
        return 'MONT'
    elif value in ['AK', 'HI']:
        return 'ALK/HAW'
    else:
        return 'PACF'

def makeLinReg(dataFrame, xVar, yVar, title, xLabel, yLabel):
    #first find linear regression of data and linear regression coefficient
    linReg = LinearRegression()
    linReg.fit(dataFrame[xVar].values.reshape(-1, 1), dataFrame[yVar])
    linCoef = linReg.score(dataFrame[xVar].values.reshape(-1, 1), dataFrame[yVar])
    linY = linReg.predict(dataFrame[xVar].values.reshape(-1, 1))

    #make arrays for variable categories
    eastX = []
    eastY = []
    centX = []
    centY = []
    montX = []
    montY = []
    alkhawX = []
    alkhawY = []
    pacfX = []
    pacfY = []
    index = 0

    #for each value in dataframe length
    for i in dataFrame[xVar]:
        #check the category of i
        if categorize(dataFrame.iloc[:,0][index]) == 'ALK/HAW':
            #make sure the value is not -1
            if dataFrame.iloc[:,1][index] != -1 or dataFrame.iloc[:,2][index] != -1:
                #append x and y variables to respective lists
                alkhawX.append(dataFrame.iloc[:,1][index])
                alkhawY.append(dataFrame.iloc[:,2][index])
        elif categorize(dataFrame.iloc[:,0][index]) == 'EAST':
            if dataFrame.iloc[:,1][index] != -1 or dataFrame.iloc[:,2][index] != -1:
                eastX.append(dataFrame.iloc[:,1][index])
                eastY.append(dataFrame.iloc[:,2][index])
        elif categorize(dataFrame.iloc[:,0][index]) == 'CENT':
            if dataFrame.iloc[:,1][index] != -1 or dataFrame.iloc[:,2][index] != -1:
                centX.append(dataFrame.iloc[:,1][index])
                centY.append(dataFrame.iloc[:,2][index])
        elif categorize(dataFrame.iloc[:,0][index]) == 'MONT':
            if dataFrame.iloc[:,1][index] != -1 or dataFrame.iloc[:,2][index] != -1:
                montX.append(dataFrame.iloc[:,1][index])
                montY.append(dataFrame.iloc[:,2][index])
        else:
            if dataFrame.iloc[:,1][index] != -1 or dataFrame.iloc[:,2][index] != -1:
                pacfX.append(dataFrame.iloc[:,1][index])
                pacfY.append(dataFrame.iloc[:,2][index])
        index+=1


    #make lists of min and max y-values
    yMin = [min(eastY), min(centY), min(montY), min(alkhawY), min(pacfY)]
    if -1 in yMin:
        yMin.remove(-1)
    yMax = [max(eastY), max(centY), max(montY), max(alkhawY), max(pacfY)]
    if -1 in yMax:
        yMax.remove(-1)

    #plot data as a scatter plot
    fig = go.Figure()
    #each region has its own trace
    fig.add_trace(
        go.Scatter(
            x = eastX,
            y = eastY,
            mode = 'markers',
            name='East'
        )
    )
    fig.add_trace(
        go.Scatter(
            x = centX,
            y = centY,
            mode = 'markers',
            name='Central'
        )
    )
    fig.add_trace(
        go.Scatter(
            x = montX,
            y = montY,
            mode = 'markers',
            name='Mountain'
        )
    )
    fig.add_trace(
        go.Scatter(
            x = alkhawX,
            y = alkhawY,
            mode = 'markers',
            name='Alaska/Hawaii'
        )
    )
    fig.add_trace(
        go.Scatter(
            x = pacfX,
            y = pacfY,
            mode = 'markers',
            name='Pacific'
        )
    )
    #make a trace for the linear regression line
    fig.add_trace(
        go.Scatter(
            x = dataFrame[xVar],
            y = linY,
            mode = 'lines',
            name='linCoef = '+str(round(linCoef, 3))
        )
    )
    #set the titles for figure, x-axis, y-axis
    fig.update_layout(
        title = title,
        xaxis=dict(
            title = xLabel
        ),
        yaxis=dict(
            title = yLabel,
            #set range to min and max y values -/+ 10 for padding
            range=(min(yMin)-10,max(yMax)+10)
        ),
    )
    #make a filename and save as html – opens automatically
    filename = 'linreg_'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)
    #return the linear regression coefficient for the data
    return linCoef

def usaMap(dataFrame, loc, var, color, title):
    #make a map of the usa that shows relative amounts of var
    #usaMap(dataSubset3, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Blues', 'Average Release Estimate Total Per State Over Time')

    fig = go.Figure(data=go.Choropleth(
        #set location equal to dataframe column (for example: dataFrame['STATE_ABBR'])
        locations=dataFrame[loc],
        #set data equal to datafram column corresponding to variable passed as arg
        z = dataFrame[var],
        #set location mode to United States
        locationmode = 'USA-states',
        #set colorscale to color scheme passed as arg
        colorscale = color,
    ))

    fig.update_layout(
        #set the title to the title passed as arg
        title_text = title,
        #set the scope to usa so that fig only shows United States map
        geo_scope='usa',
    )

    #make filename and write the figure to html – opens automatically
    filename = 'usamap_'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

def makeSliderScatter(dataFrame, sliderVar, sliders, xVar, yVar, title, xLabel, yLabel):
    #make a new figure
    fig = go.Figure()
    #go through the data that should be on the slider bar
    for s in sliders:
        #find rows corresponding to slider value
        rows = dataFrame.loc[dataFrame[sliderVar] == s]
        #make a subset of the dataframe with columns for the sliderVar, xVar, yVar
        varList = [sliderVar, xVar, yVar]
        dataSubset = makeSubset(rows, varList)
        #add a scatterplot trace with datasubset values
        fig.add_trace(
            go.Scatter(
                x = dataSubset[xVar],
                y = dataSubset[yVar],
                mode = 'lines',
            )
        )

    #set all of the traces' visibility to false
    for i in range(len(sliders)):
        fig.data[i].visible=False
    #set the first trace's visibility to true so that it shows up as the initial trace
    fig.data[0].visible=True

    #make an array to hold data for the steps
    steps = []
    #use s as an index to loop through len of slider variables
    for s in range(len(sliders)):
        #make new step
        step = dict(
            method="restyle",
            args=["visible", [False] * len(sliders)],
            label=str(sliders[s])
        )
        step["args"][1][s]=True
        #add to steps array
        steps.append(step)
    #set up sliders with sliderVar label
    sliders = [dict(
        active=10,
        currentvalue={"prefix": sliderVar+": "},
        pad={"t": 50},
        steps=steps
    )]
    #set titles for fig, x-axis, y-axis; set y-axis range; add sliders
    fig.update_layout(
        title = title,
        xaxis=dict(
            title = xLabel
        ),
        yaxis=dict(
            title = yLabel,
            range=(min(dataFrame[yVar]),max(dataFrame[yVar]))
        ),
        sliders=sliders,
        font=dict(
            size=8
        )
    )
    #make filename and write to html – opens automatically
    filename = 'scatter_'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

def makeSliderBar(dataFrame, sliderVar, sliders, xVar, yVar, title, xLabel, yLabel):
    fig = go.Figure()
    #go through the data that should be on the slider bar
    for s in sliders:
        #find rows corresponding to slider value
        rows = dataFrame.loc[dataFrame[sliderVar] == s]
        #make a subset of the dataframe with columns for the sliderVar, xVar, yVar
        varList = ['YEAR', sliderVar, xVar, yVar]
        dataSubset = makeSubset(rows, varList)
        #add a bar graph trace with datasubset values
        fig.add_trace(
            go.Bar(
                x = dataSubset[xVar],
                y = dataSubset[yVar],
            )
        )
    #set all of the traces' visibility to false
    for i in range(len(sliders)):
        fig.data[i].visible=False
    #set the first trace's visibility to true so that it shows up as the initial trace
    fig.data[0].visible=True
    #make an array to hold data for the steps
    steps = []
    #use s as an index to loop through len of slider variables
    for s in range(len(sliders)):
        #make new step
        step = dict(
            method="restyle",
            args=["visible", [False] * len(sliders)],
            label=str(sliders[s])
        )
        step["args"][1][s]=True
        #add to steps array
        steps.append(step)
    #set up sliders with sliderVar label
    sliders = [dict(
        active=10,
        currentvalue={"prefix": sliderVar+": "},
        pad={"t": 50},
        steps=steps
    )]
    #set titles for fig, x-axis, y-axis; set y-axis range; add sliders
    fig.update_layout(
        title = title,
        xaxis=dict(
            title = xLabel
        ),
        yaxis=dict(
            title = yLabel,
            range=(min(dataFrame[yVar]),max(dataFrame[yVar])),
        ),
        sliders=sliders,
        font=dict(
            size=8
        )
    )
    #make filename and write to html – opens automatically
    filename = 'bar'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

def main():
    #read data from cleaned epa files into pandas dataframe
    dataFrame = pd.read_csv("merged_data.csv", sep=',', encoding='latin1')
    #remove territories
    territories = ['AS', 'FM', 'GU', 'MH', 'MP', 'PR', 'PW', 'VI', 'UM']
    for t in territories:
        dataFrame = dataFrame[dataFrame['STATE_ABBR'] != t]
    #find a list of unique state abbreviations and a list of unique years
    stateList = dataFrame['STATE_ABBR'].unique()
    yearList = dataFrame['YEAR'].unique()

    #make subset of relevant data for all states including Alaska
    varList = ['STATE_ABBR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE', 'YEAR']
    subsetAK = makeSubset(dataFrame, varList)
    subsetAK.dropna(inplace=True)
    #make subset of relevant data for all states excluding Alaska
    rows = dataFrame.loc[dataFrame['STATE_ABBR'] != 'AK']
    subsetNoAK = makeSubset(rows, varList)
    subsetNoAK.dropna(inplace=True)

    #make subset of total release estimates, cancer rates averaged per state over time including Alaska
    avgCancer = []
    avgChems = []
    for state in stateList:
        avgCancer.append(avgOverTime(subsetAK, state, 'AGE_ADJUSTED_CANCER_RATE'))
        avgChems.append(avgOverTime(subsetAK, state, 'AVG_REL_EST_TOTAL'))
    subsetAKAvg = pd.DataFrame({'STATE_ABBR': stateList, 'AVG_REL_EST_TOTAL': avgChems, 'AGE_ADJUSTED_CANCER_RATE': avgCancer})
    #make subset of total release estimates, cancer rates averaged per state over time excluding Alaska
    avgChemsNoAK = []
    avgCancerNoAK = []
    for state in stateList[1:]:
        avgChemsNoAK.append(avgOverTime(subsetNoAK, state, 'AVG_REL_EST_TOTAL'))
        avgCancerNoAK.append(avgOverTime(subsetNoAK, state, 'AGE_ADJUSTED_CANCER_RATE'))
    subsetNoAKAvg = pd.DataFrame({'STATE_ABBR': stateList[1:], 'AVG_REL_EST_TOTAL': avgChemsNoAK, 'AGE_ADJUSTED_CANCER_RATE': avgCancerNoAK})

    #make linear regression models for age-adjusted cancer rate vs. avg rel est tot including and excluding Alaska
    makeLinReg(subsetAKAvg, 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate vs. Average Release Estimate Total', 'Average Chemical Release Estimate Total', 'Age-Adjusted Cancer Rate')
    makeLinReg(subsetNoAKAvg, 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate vs. Average Release Estimate Total (excluding Alaska)', 'Average Chemical Release Estimate Total', 'Age-Adjusted Cancer Rate')

    #make maps showing avg rel est tot for each state averaged over time (including and excluding Alaska)
    usaMap(subsetAKAvg, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Blues', 'Average Release Estimate Total Per State Over Time')
    usaMap(subsetNoAKAvg, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Blues', 'Average Release Estimate Total Per State Over Time (excluding Alaska)')
    #make  map showing age adjusted cancer rate for each state averaged over time
    usaMap(subsetAKAvg, 'STATE_ABBR', 'AGE_ADJUSTED_CANCER_RATE', 'Blues', 'Age Adjusted Cancer Rate Per State Over Time')

    #make scatter plots showing avg rel est tot for each state over time, with state as slider (including and excluding Alaska)
    makeSliderScatter(subsetAK, 'STATE_ABBR', stateList, 'YEAR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount Over Time Per State', 'Year', 'Average Chemical Release Estimate Total')
    makeSliderScatter(subsetNoAK, 'STATE_ABBR', stateList[1:], 'YEAR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount Over Time Per State – Alaska excluded', 'Year', 'Average Chemical Release Estimate Total')
    #make scatter plot showing age adjusted cancer rate for each state over time, with state as slider
    makeSliderScatter(subsetAK, 'STATE_ABBR', stateList, 'YEAR', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate Over Time Per State', 'Year', 'Age-Adjusted Cancer Rate')

    #make bar graphs showing avg rel est tot for each state over time, with year as slider (including and excluding Alaska)
    makeSliderBar(subsetAK, 'YEAR', yearList, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount for States Over Time', 'Year', 'Average Chemical Release Estimate Total')
    makeSliderBar(subsetNoAK, 'YEAR', yearList, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount for States Over Time – Alaska excluded', 'Year', 'Average Chemical Release Estimate Total')
    #make bar graph showing age adjusted cancer rate for each state over time, with year as slider
    makeSliderBar(subsetAK, 'YEAR', yearList, 'STATE_ABBR', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate for States Over Time', 'Year', 'Age-Adjusted Cancer Rate')

if __name__ == '__main__':
    main()

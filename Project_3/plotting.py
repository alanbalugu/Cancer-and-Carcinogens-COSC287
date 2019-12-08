import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import plotly
import chart_studio.plotly as py
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pprint import pprint

def makeSubset(dataFrame, varList):
    #make a new dataframe for the subset
    dataSubset = pd.DataFrame()
    #loop through the variables given, make subset frame col equal to dataFrame col
    for var in varList:
        dataSubset[var] = dataFrame[var]
    return dataSubset

def usaMap(dataFrame, loc, var, color, title):
    print(dataFrame[loc])
    print(dataFrame[var])
    fig = go.Figure(data=go.Choropleth(
        locations=dataFrame[loc],
        z = dataFrame[var],
        locationmode = 'USA-states',
        colorscale = color,
    ))

    fig.update_layout(
        title_text = title,
        geo_scope='usa',
    )

    filename = 'usamap_'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

def equiWidthBinning(dataFrame, var, numBins):
    #puts data into bins of equal width
    minVal = dataFrame[var].min() - 1
    maxVal = dataFrame[var].max() + 1

    step = (maxVal - minVal) / numBins
    bins =  np.arange(minVal, maxVal + step, step)

    equiWidthBins = np.digitize(dataFrame[var], bins)

    binCounts = np.bincount(equiWidthBins)
    print("\n\nBins for " + var + " are: \n ", equiWidthBins)
    print("\nBin count is ", binCounts)

    return dataFrame[var]

def makeHistogram(dataFrame, var, title):
    #plot the data as a histogram
    binnedData = equiWidthBinning(dataFrame, var, 10)
    plt.figure(1)
    binnedData.hist()
    plt.title(var + ' Distribution')
    plt.xlabel(var)
    plt.savefig('epa_'+ var + '_histogram.png')
    plt.show()

def avgOverTime(dataFrame, stateAbbr, var):
    rows = dataFrame.loc[dataFrame['STATE_ABBR'] == stateAbbr]
    toMap = [var]
    dataSubset = makeSubset(rows, toMap)

    count = 0
    total = 0
    for i in dataSubset[var]:
        total = total + i
        count = count + 1

    if count != 0:
        return (total/count)
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

    #for each year
    for i in dataFrame[xVar]:
        if categorize(dataFrame.iloc[:,0][index]) == 'ALK/HAW':
            if dataFrame.iloc[:,1][index] != -1 or dataFrame.iloc[:,2][index] != -1:
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

    index = 0

    yMin = [min(eastY), min(centY), min(montY), min(alkhawY), min(pacfY)]
    if -1 in yMin:
        yMin.remove(-1)
    yMax = [max(eastY), max(centY), max(montY), max(alkhawY), max(pacfY)]
    if -1 in yMax:
        yMax.remove(-1)
    #plot data as a scatter plot
    fig = go.Figure()
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
    fig.add_trace(
        go.Scatter(
            x = dataFrame[xVar],
            y = linY,
            mode = 'lines',
            name='linCoef = '+str(round(linCoef, 3))
        )
    )

    fig.update_layout(
        title = title,
        xaxis=dict(
            title = xLabel
        ),
        yaxis=dict(
            title = yLabel,
            range=(min(yMin)-10,max(yMax)+10)
        ),
    )

    filename = 'linreg_'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

    #return the linear regression coefficient for the data
    return linCoef

def makeSliderScatter(dataFrame, sliderVar, sliders, xVar, yVar, title, xLabel, yLabel):
    fig = go.Figure()
    for s in sliders:
        rows = dataFrame.loc[dataFrame[sliderVar] == s]
        varList = ['YEAR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE']
        dataSubset = makeSubset(rows, varList)
        fig.add_trace(
            go.Scatter(
                x = dataSubset[xVar],
                y = dataSubset[yVar],
                mode = 'lines',
            )
        )

    for i in range(len(sliders)):
        fig.data[i].visible=False
    fig.data[0].visible=True

    steps = []
    for s in range(len(sliders)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(sliders)],
            label=str(sliders[s])
        )
        step["args"][1][s]=True
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

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

    filename = 'scatter_'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

def makeSliderBar(dataFrame, sliderVar, sliders, xVar, yVar, title, xLabel, yLabel):
    fig = go.Figure()
    for s in sliders:
        rows = dataFrame.loc[dataFrame[sliderVar] == s]
        varList = ['YEAR', 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE']
        dataSubset = makeSubset(rows, varList)
        fig.add_trace(
            go.Bar(
                x = dataSubset[xVar],
                y = dataSubset[yVar],
            )
        )

    for i in range(len(sliders)):
        fig.data[i].visible=False
    fig.data[0].visible=True

    steps = []
    for s in range(len(sliders)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(sliders)],
            label=str(sliders[s])
        )
        step["args"][1][s]=True
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

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

    filename = 'bar'+title.replace(' ','_')+'.html'
    fig.write_html(filename, auto_open=True)

def main():
    #read data from cleaned epa files into pandas dataframe
    dataFrame = pd.read_csv("merged_data.csv", sep=',', encoding='latin1')
    territories = ['AS', 'FM', 'GU', 'MH', 'MP', 'PR', 'PW', 'VI', 'UM']
    for t in territories:
        dataFrame = dataFrame[dataFrame['STATE_ABBR'] != t]
    rows = dataFrame.loc[dataFrame['YEAR'] == 2016]
    toMap = ['STATE_ABBR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE']
    dataSubset = makeSubset(rows, toMap)
    stateList = dataFrame['STATE_ABBR'].unique()
    yearList = dataFrame['YEAR'].unique()
    rows2 = dataFrame.loc[dataFrame['STATE_ABBR'] != 'AK']
    toMap2 = ['STATE_ABBR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE', 'YEAR']
    dataSubset2 = makeSubset(rows2, toMap2)
    year=2016

    linRegVars = ['STATE_ABBR', 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE']
    dataSubset3 = makeSubset(dataFrame, linRegVars)
    dataSubset3.dropna(inplace=True)

    avgCancer = []
    avgChems = []
    for state in stateList:
        avgCancer.append(avgOverTime(dataSubset3, state, 'AGE_ADJUSTED_CANCER_RATE'))
        avgChems.append(avgOverTime(dataSubset3, state, 'AVG_REL_EST_TOTAL'))

    dataSubset3 = pd.DataFrame({'STATE_ABBR': stateList, 'AVG_REL_EST_TOTAL': avgChems, 'AGE_ADJUSTED_CANCER_RATE': avgCancer})
    makeLinReg(dataSubset3, 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate vs. Average Release Estimate Total', 'Average Chemical Release Estimate Total', 'Age-Adjusted Cancer Rate')

    dataSubset4 = dataSubset3[dataSubset3['STATE_ABBR'] != 'AK']
    modStateList = stateList[1:]
    modAvgCancer = []
    modAvgChems = []
    for state in modStateList:
        modAvgCancer.append(avgOverTime(dataSubset4, state, 'AGE_ADJUSTED_CANCER_RATE'))
        modAvgChems.append(avgOverTime(dataSubset4, state, 'AVG_REL_EST_TOTAL'))

    dataSubset4 = pd.DataFrame({'STATE_ABBR': modStateList, 'AVG_REL_EST_TOTAL': modAvgChems, 'AGE_ADJUSTED_CANCER_RATE': modAvgCancer})
    makeLinReg(dataSubset4, 'AVG_REL_EST_TOTAL', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate vs. Average Release Estimate Total (excluding Alaska)', 'Average Chemical Release Estimate Total', 'Age-Adjusted Cancer Rate')

    #makeHistogram(dataFrame, 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate Distribution')
    #makeHistogram(dataFrame, 'AVG_REL_EST_TOTAL', 'Average Release Estimate Total Distribution')
    usaMap(dataSubset3, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Blues', 'Average Release Estimate Total Per State Over Time')
    usaMap(dataSubset3, 'STATE_ABBR', 'AGE_ADJUSTED_CANCER_RATE', 'Blues', 'Age Adjusted Cancer Rate Per State Over Time')
    usaMap(dataSubset4, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Blues', 'Average Release Estimate Total Per State Over Time (excluding Alaska)')
    usaMap(dataSubset4, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Blues', 'Average Release Estimate Total Per State Over Time (excluding Alaska)')

    makeSliderScatter(dataFrame, 'STATE_ABBR', stateList, 'YEAR', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate Over Time Per State', 'Year', 'Age-Adjusted Cancer Rate')
    makeSliderScatter(dataFrame, 'STATE_ABBR', stateList, 'YEAR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount Over Time Per State', 'Year', 'Average Chemical Release Estimate Total')
    #makeSliderScatter(dataSubset2, 'STATE_ABBR', stateList[1:], 'YEAR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount Over Time Per State – Alaska excluded')


    makeSliderBar(dataFrame, 'YEAR', yearList, 'STATE_ABBR', 'AGE_ADJUSTED_CANCER_RATE', 'Age-Adjusted Cancer Rate for States Over Time', 'Year', 'Age-Adjusted Cancer Rate')
    makeSliderBar(dataFrame, 'YEAR', yearList, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount for States Over Time', 'Year', 'Average Chemical Release Estimate Total')
    #makeSliderBar(dataSubset2, 'YEAR', yearList, 'STATE_ABBR', 'AVG_REL_EST_TOTAL', 'Average Chemical Release Amount for States Over Time – Alaska excluded')

if __name__ == '__main__':
    main()

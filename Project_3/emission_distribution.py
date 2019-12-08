import plotly.express as px
import pandas as pd

import plotly

import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objs as go

# gapminder = px.data.gapminder().query("country=='Canada'")
# fig = px.line(gapminder, x="year", y="lifeExp", title='Life expectancy in Canada')
# fig.show()

RELEASE_NAMES = ['Air - Stack', 'Air - not stack', 'Land treatment/application farming ',
				'Landfill - Other', 'On-site RCRA Subtitle C landfills', 'On-site surface impoundment',
				'Other On-site surface impoundment', 'Other On-site land', 'Publicly Owned Treatment Works - Metals',
				'Publicly Owned Treatment Works - Non Metals', 'Publicly Owned Treatment Works - Treatment',
				'RCRA Subtitle C On-site surface impoundment', 'Recycled', 'Underground injection On-site I',
				'Underground injection on-site V', 'Used for Energy Recovery', 'Waste Treatment', 'Water']

# plot land distribution over time
def stackedBarChart(data, col_names):
	# Source: https://plot.ly/python/bar-charts/
	x = data['YEAR']
	# Create a trace for a bar plot
	fig = go.Figure(go.Bar())
	for i in range(len(col_names)):
		fig.add_trace(go.Bar(x=x, y=data[col_names[i]], name=RELEASE_NAMES[i]))

	fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'}, title="Emission Distribution Over Time",yaxis_title="Total Emissions",xaxis_title="Year")

	fig.write_html('emission_distribution.html', auto_play=True)

def multiLineChart(data, col_names):
	# Source: https://plot.ly/python/bar-charts/
	x = data['YEAR']
	# Create a trace for a scatterplot
	fig = go.Figure()
	for i in range(len(col_names)):
		fig.add_trace(go.Scatter(x=x, y=data[col_names[i]], name=RELEASE_NAMES[i], mode='lines'))
	
	fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'}, title="Emission Distribution Over Time",yaxis_title="Total Emissions",xaxis_title="Year")

def main():
	# RELEASE_NAMES.sort()
	# print(RELEASE_NAMES)
	# exit()
	df = pd.read_csv('merged_data2.csv' , sep=',', encoding='latin1')
	df.drop(columns='AVG_REL_EST_TOTAL_ONSITE_RELEASE') 	# drop sum columns
	df.drop(columns='AVG_REL_EST_TOTAL_ON_OFFSITE_RELEASE') # drop sum columns
	release_col_names = df.columns[2:20]
	print(release_col_names)
	# release_types = [s.replace('AVG_REL_EST_','') for s in release_col_names]
	years = df.YEAR.unique()
	years = [int(x) for x in years]

	temp = pd.DataFrame(data={'YEAR':years})
	annual_total_pollution_by_release_type = pd.concat([temp, pd.DataFrame(columns=release_col_names)],sort=True)
	
	i = int(0)
	for year in years:
		for col_name in release_col_names:
			annual_total_pollution_by_release_type.at[i,col_name] = df[df['YEAR'] == int(year)][col_name].sum()

		i += 1

		
	annual_total_pollution_by_release_type.to_csv(r'emission_distribution.csv', index = False , header=True)

	stackedBarChart(annual_total_pollution_by_release_type, release_col_names)
	# multiLineChart(annual_total_pollution_by_release_type, release_col_names, release_types)



if __name__ == '__main__':
	main()
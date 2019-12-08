import plotly.express as px
import pandas as pd

import plotly

import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objs as go

# gapminder = px.data.gapminder().query("country=='Canada'")
# fig = px.line(gapminder, x="year", y="lifeExp", title='Life expectancy in Canada')
# fig.show()

# plot land distribution over time
def stackedBarChart(data, col_names, release_names):
	# Source: https://plot.ly/python/bar-charts/
	x = data['YEAR']
	# Create a trace for a scatterplot
	fig = go.Figure(go.Bar())
	for i in range(len(col_names)):
		fig.add_trace(go.Bar(x=x, y=data[col_names[i]], name=release_names[i]))

	fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'}, title="Emission Distribution Over Time",yaxis_title="Total Emissions",xaxis_title="Year")

	fig.write_html('emission_distribution.html', auto_play=True)



def multiLineChart(data, col_names, release_names):
	# Source: https://plot.ly/python/bar-charts/
	x = data['YEAR']
	# Create a trace for a scatterplot
	fig = go.Figure()
	for i in range(len(col_names)):
		fig.add_trace(go.Scatter(x=x, y=data[col_names[i]], name=release_names[i], mode='lines'))
	

	# myLayout = go.Layout(
	# 	title = "Forest Land Vs Emissions - Per Capita",
	# 	xaxis=dict(
	# 		title = 'Forest Land per Capita'
	# 	),
	# 	yaxis=dict(
	# 		title = 'Carbon Emissions per Capita'
	# 	)
	# )
	fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'}, title="Emission Distribution Over Time",yaxis_title="Total Emissions",xaxis_title="Year")

	


def main():
	df = pd.read_csv('merged_data2.csv' , sep=',', encoding='latin1')
	# print(df)
	# print(df.columns)
	release_col_names = df.columns[2:22]
	release_types = [s.replace('AVG_REL_EST_','') for s in release_col_names]
	# print(release_types)
	years = df.YEAR.unique()
	years = [int(x) for x in years]
	# annual_total_pollution_by_release_type = []

	temp = pd.DataFrame(data={'YEAR':years})
	annual_total_pollution_by_release_type = pd.concat([temp, pd.DataFrame(columns=release_col_names)],sort=False)
	# print(final)
	# exit(0)
	# print(df[df['YEAR'] == 1999]
	i = int(0)
	for year in years:
		# print('year = ', year)
		for col_name in release_col_names:
			# print('col_name = ', col_name)
			annual_total_pollution_by_release_type.at[i,col_name] = df[df['YEAR'] == int(year)][col_name].sum()

		i += 1

			# exit(0)
	# annual_total_pollution_by_release_type = df[2:22]
	# annual_total_
	# print(annual_total_pollution_by_release_type)
	annual_total_pollution_by_release_type.to_csv(r'release_sums.csv', index = False , header=True)
	stackedBarChart(annual_total_pollution_by_release_type, release_col_names, release_types)

	# multiLineChart(annual_total_pollution_by_release_type, release_col_names, release_types)



if __name__ == '__main__':
	main()
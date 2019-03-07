import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

import flask
import pandas as pd
import time
import os
#from log_to_csv import *
from mongo import *

PAGE_SIZE = 10

booleanDictionary = {'nearMiss':{True: 'TRUE', False: 'FALSE'}}

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/hello-world-stock.csv')
#collision_info = create_csv('/home/m/Documents/incluit/intel/OpenVino-For-SmartCity/build/log/*')
#col_df = pd.DataFrame.from_dict(collision_info)
#unique_ids_tuples = col_df.groupby(['ob1','ob2']).size().reset_index().drop(0,axis=1)
#print(unique_ids_tuples)
"""table_dicts = []
for row in unique_ids_tuples.itertuples():
    filter_df = col_df.loc[(col_df['ob1'] == row[1]) & (col_df['ob2'] == row[2])]
    column_frame = filter_df['frame']
    table_entry = {'ob1': row[1], 'ob2': row[2], 'min_f': column_frame.min(), 'max_f': column_frame.max()}
    table_dicts.append(table_entry)"""

"""df = pd.read_csv('log.csv')
df.dropna(inplace=True)
df.Id = df.Id.astype(int)"""

app = dash.Dash('app', server=server)

app.scripts.config.serve_locally = False
dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

"""events_l = events_list()
events_df = pd.DataFrame(events_l).replace(booleanDictionary).sort_values(by=['nearMiss','frame'],ascending=[False,True])
"""

app.layout = html.Div([
    html.Div([
        html.H1('Incluit - OpenVINO For SmartCity dashboard')
    ],className='row'),
    html.Div([
        html.Div([
            dcc.Graph(id='my-graph1')
        ], className="four columns")
        ,
        html.Div([
            html.H2('Collisions'),
            dt.DataTable(
                id='datatable-collision',
                columns=[   {"name": "Object N°", "id": "ob1"}, 
                            {"name": "Object N°", "id": "ob2"},
                            {"name": "Init frame", "id": "min_f"},
                            {"name": "End frame", "id": "max_f"},
                        ],
                row_selectable="single",
                selected_rows=[],
            ),
            html.H2('Events'),
            dt.DataTable(
                id='datatable-paging',
                columns=[   {"name": "Frame", "id": "frame"}, 
                            {"name": "ID", "id": "Id"},
                            {"name": "Class", "id": "class"},
                            {"name": "Near miss", "id": "nearMiss"},
                            {"name": "Event", "id": "event"},
                        ],
                row_selectable="single",
                selected_rows=[],
                pagination_settings={
                    'current_page': 0,
                    'page_size': PAGE_SIZE
                },
                pagination_mode='be'
            )
        ], className="four columns")
        ,
        html.Div([
            dcc.Graph(id='my-graph2')
        ], className="four columns"),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in millisecondsnear
            n_intervals=0
        )
    ], className="row")
],className="row")

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
@app.callback(
    Output('datatable-collision', 'data'),
    [   Input('datatable-collision', 'derived_virtual_selected_rows'),
        Input('interval-component', 'n_intervals')])
def update_collision_table(derived_virtual_selected_rows,n):
    cl = collision_list()
    table_dicts = []
    if(cl):
        col_df = pd.DataFrame.from_dict(cl)
        unique_ids_tuples = col_df.groupby(['ob1','ob2']).size().reset_index().drop(0,axis=1)
        for row in unique_ids_tuples.itertuples():
            filter_df = col_df.loc[(col_df['ob1'] == row[1]) & (col_df['ob2'] == row[2])]
            column_frame = filter_df['frame']
            table_entry = {'ob1': row[1], 'ob2': row[2], 'min_f': column_frame.min(), 'max_f': column_frame.max()}
            table_dicts.append(table_entry)
 
    return table_dicts

@app.callback(
    Output('datatable-paging', 'data'),
    [Input('datatable-paging', 'pagination_settings'),
    Input('interval-component', 'n_intervals')])
def update_events_table(pagination_settings,n):
    events_l = events_list()
    if(events_l):
        events_df = pd.DataFrame(events_l).replace(booleanDictionary).sort_values(by=['nearMiss','frame'],ascending=[False,False])
        return events_df.iloc[
            pagination_settings['current_page']*pagination_settings['page_size']:
            (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
        ].to_dict('rows')
    else:
        return []
@app.callback(Output('my-graph1', 'figure'),
                [Input('datatable-collision', 'derived_virtual_selected_rows'),
                Input('datatable-collision', 'data'),
                Input('interval-component', 'n_intervals')
                ])
                
def update_graph(derived_virtual_selected_rows,td,n):
    tl = tracker_list()
    df = pd.DataFrame(tl)
    if derived_virtual_selected_rows == None or derived_virtual_selected_rows == []:
        if len(td) != 0:
            object_id = td[0]['ob1']
            dff = df[df['Id'] == object_id]
            x_axis_range = [td[0]['min_f']-10,td[0]['max_f']+10]
            title = 'Object ID: ' + str(object_id)

        else:
            dff = pd.DataFrame([{'frame':None,'vel_x':None,'vel_y':None}])
            object_id = None
            title = None
            x_axis_range = [0,100]
    else:
        object_id = td[derived_virtual_selected_rows[0]]['ob1']
        dff = df[df['Id'] == object_id]
        x_axis_range = [td[derived_virtual_selected_rows[0]]['min_f']-10, td[derived_virtual_selected_rows[0]]['max_f']+10]
        title = 'Object ID: ' + str(object_id)

    return {
        'data': [{
            'x': dff.frame,
            'y': dff.vel_x,
            'line': {
                'width': 3,
                'shape': 'spline'
            },
            'name': 'Vel_x'
        },
        {
            'x': dff.frame,
            'y': dff.vel_y,
            'line': {
                'width': 3,
                'shape': 'spline'
            },
            'name': 'Vel_y'
        }
        ],
        'layout': {
            'title': title,
            'margin': {
                'l': 30,
                'r': 20,
                'b': 30,
                't': 35
            },
            
            'legend': {'x': 0, 'y': 1},
            'xaxis': {  'range': x_axis_range,
                        'uirevision': 'sarlanga',
            }
        }
    }

@app.callback(Output('my-graph2', 'figure'),
                [Input('datatable-collision', 'derived_virtual_selected_rows'),
                Input('datatable-collision', 'data'),
                Input('interval-component', 'n_intervals')])
                
def update_graph2(derived_virtual_selected_rows,td,n):
    tl = tracker_list()
    df = pd.DataFrame(tl)
    if derived_virtual_selected_rows == None or derived_virtual_selected_rows == []:
        if len(td) != 0:
            object_id = td[0]['ob2']
            dff = df[df['Id'] == object_id]
            x_axis_range = [td[0]['min_f']-10,td[0]['max_f']+10]
            title = 'Object ID: ' + str(object_id)
        else:
            dff = pd.DataFrame([{'frame':None,'vel_x':None,'vel_y':None}])
            object_id = None
            x_axis_range = [0,100]
            title = None
    else:
        object_id = td[derived_virtual_selected_rows[0]]['ob2']
        dff = df[df['Id'] == object_id]
        x_axis_range = [td[derived_virtual_selected_rows[0]]['min_f']-10, td[derived_virtual_selected_rows[0]]['max_f']+10]
        title = 'Object ID: ' + str(object_id)

    return {
        'data': [{
            'x': dff.frame,
            'y': dff.vel_x,
            'line': {
                'width': 3,
                'shape': 'spline'
            },
            'name': 'Vel_x'
        },
        {
            'x': dff.frame,
            'y': dff.vel_y,
            'line': {
                'width': 3,
                'shape': 'spline'
            },
            'name': 'Vel_y'
        }
        ],
        'layout': {
            'title': title,
            'margin': {
                'l': 30,
                'r': 20,
                'b': 30,
                't': 35
            },
            'legend': {'x': 0, 'y': 1},
            'xaxis': {  'range': x_axis_range,
                        'uirevision': 'sarlanga',

            }
        }
    }

if __name__ == '__main__':
    app.run_server(debug=True)
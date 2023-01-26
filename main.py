from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


class model_input(BaseModel):
    average_rank: float
    rank_difference: float
    point_difference: float
    is_stake: bool
    is_worldcup: bool


kickoff_model = pickle.load(open('Kickoff_model.sav', 'rb'))


@app.post('/match_prediction')
def match_pred(input_parameters: model_input):
    # input_data = input_parameters.json()
    # input_dict = json.loads(input_data)
    #
    # avg_rank = input_dict['average_rank']
    # rank_diff = input_dict['rank_difference']
    # point_diff = input_dict['point_difference']
    # stake = input_dict['is_stake']
    # world_cup = input_dict['is_worldcup']

    # input_list = [[input_parameters.average_rank, input_parameters.rank_difference, input_parameters.point_difference, input_parameters.is_stake, input_parameters.is_worldcup]]

    prediction = kickoff_model.predict_proba(np.array([[input_parameters.average_rank, input_parameters.rank_difference,
                                                        input_parameters.point_difference, input_parameters.is_stake,
                                                        input_parameters.is_worldcup]]))

    return {'prediction': prediction}
    # if prediction[0] > 0.5:
    #     return f'Home team win with precentge {prediction}'
    # elif prediction < 0.5:
    #     return prediction[0]
    # else:
    #     return 'Draw'
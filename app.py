import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json



app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")




# Load the model
model = load_model('static/best_model.h5', compile=False)


with open('static/teams.json') as file:
    team_mapping = json.load(file)
with open('static/grounds.json') as file:
    ground_mapping = json.load(file)





class MatchInput(BaseModel):
    team1: str
    team2: str
    ground: str

@app.get('/')
async def welcome(request: Request):
    return {"message": str(model.summary())}


@app.post("/predict/")
def predict_winner(match: MatchInput):
    # Encode the inputs
    # team1_code = team_encoder.transform([match.team1])[0]
    # team2_code = team_encoder.transform([match.team2])[0]
    # ground_code = ground_encoder.transform([match.ground])[0]
    # print(team1_code,team2_code,ground_code)

    team1_code = team_mapping.get(match.team1, -1)  # -1 or another default for unrecognized input
    team2_code = team_mapping.get(match.team2, -1)
    ground_code = ground_mapping.get(match.ground, -1)

    if team1_code == -1 or team2_code == -1 or ground_code == -1:
        return {"error": "Invalid team or ground name"}
    # team1_code=8
    # team2_code=1
    # ground_code=73
    # Make a prediction
    prediction = model.predict([[team1_code, team2_code, ground_code]])
    if prediction[0][team1_code] > prediction[0][team2_code]:
        winner= [name for name, code in team_mapping.items() if code == team1_code]
    else :
        winner= [name for name, code in team_mapping.items() if code == team2_code]
    return {"winner": winner}

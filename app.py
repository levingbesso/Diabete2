



from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import pickle
import uvicorn
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})



@app.post("/predict")
def predict(request: Request,
            age: float = Form(...),
            bmi: float = Form(...),
            bp: float = Form(...),
            s1: float = Form(...),
            s2: float = Form(...),
            s3: float = Form(...),
            s4: float = Form(...),
            s5: float = Form(...),
            s6: float = Form(...)):
    data = [age, bmi, bp, s1, s2, s3, s4, s5, s6]
    data = np.array(data).reshape(1, -1)
    scaled_data = scalar.transform(data)
    prediction = regmodel.predict(scaled_data)[0]
    return templates.TemplateResponse("home.html", {"request": request, "prediction_text": f"The diabete progression is {prediction:.2f}"})

if __name__ == "__main__":
    uvicorn.run(app)













